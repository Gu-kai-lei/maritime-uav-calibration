# Rare-Class Resolution-by-Exposure Control

## Registered question

The post-freeze audit found that `life_saving_appliances` is the rarest active detector-train class
(422 instances) and has a median effective size of only 5.50 x 4.83 pixels at 640. It also spans
visibly different contexts across frozen source roles. This control separates two actionable main
effects: input resolution and repeated exposure of rare-positive training images.

The complete registration is versioned in
[`configs/rare_class_factorial.yaml`](../configs/rare_class_factorial.yaml). Official validation is
forbidden for training, checkpoint selection, threshold selection, and stage gating.

## Stage design

| Cell | Resolution | Rare-positive exposure | Status |
|---|---:|---:|---|
| Natural-640 | 640 | 1x | Reused stable no-AMP baseline |
| Repeat4-640 | 640 | 4x total | Completed after resume; best epoch 45 |
| Natural-1280 | 1280 | 1x | Completed; best epoch 49 |
| Repeat4-1280 | 1280 | 4x total | Completed; best epoch 39 |

Repeat4 preserves all 6,251 unique detector-train images and repeats only the 323 images containing
the target. It adds 969 exposures per epoch, bringing effective target-instance exposure from 422
to 1,688, close to the natural 1,551 jetski instances. Because other classes can co-occur in those
images, this is an image-exposure intervention, not a pure class-loss weight.

## Source-preserving dataset wrapper

The first launch preflight correctly stopped before training when Ultralytics tried to rebuild the
official-data `train.cache` after the training-list hash changed. The official data was not modified.
The active experiment instead uses an ignored local wrapper:

- an image-directory junction provides read-only access to official images;
- 8,930 text labels are copied into ignored `artifacts/`;
- both train and detector-dev lists point through the wrapper;
- Ultralytics cache files are created only inside ignored `artifacts/`.

This isolates generated cache state and preserves the official dataset snapshot.

## Reproduce the repeated list

Create the ignored wrapper without writing to the official dataset:

```powershell
.\scripts\prepare_isolated_yolo_wrapper.ps1 `
  -SourceImages D:\dataset\SeaDronesSee_ODv2\compressed\images\train `
  -SourceLabels D:\dataset\SeaDronesSee_ODv2\compressed\labels\train `
  -OutputRoot artifacts\rare_class_factorial\dataset
```

Then rebuild the lists and manifest:

```powershell
python scripts\prepare_rare_positive_sampling.py `
  --annotations artifacts\partitions\seed20260803\instances_detector_train.json `
  --source-yaml D:\dataset\SeaDronesSee_ODv2\yolo\seadronessee_detector.yaml `
  --source-train-list D:\dataset\SeaDronesSee_ODv2\yolo\lists\detector_train.txt `
  --source-val-list D:\dataset\SeaDronesSee_ODv2\yolo\lists\detector_dev.txt `
  --output-val-list artifacts\rare_class_factorial\detector_dev_isolated.txt `
  --output-image-root artifacts\rare_class_factorial\dataset\images\train `
  --target life_saving_appliances `
  --repeat-total 4 `
  --output-list artifacts\rare_class_factorial\detector_train_rare_repeat4.txt `
  --output-yaml artifacts\rare_class_factorial\seadronessee_detector_rare_repeat4.yaml `
  --manifest artifacts\rare_class_factorial\sampling_manifest.json
```

## Active Repeat4-640 command

The 2%-data, one-epoch isolated launch gate completed without CUDA, label, or non-finite failures.
Performance from that preflight is not evidence. The active full run uses:

```powershell
python scripts\train_yolo.py `
  --data artifacts\rare_class_factorial\seadronessee_detector_rare_repeat4.yaml `
  --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16 `
  --device 0 --workers 4 --project runs\detect `
  --name sds_v2_yolov8n_640_seed20260803_rare4_noamp `
  --seed 20260803 --patience 20 --no-amp `
  --optimizer auto --lr0 0.01 --nbs 64 --fraction 1.0
```

The checkpoint remains selected by the ordinary detector-dev fitness. Target-specific scores and
the two additional frozen-role diagnostics are computed only after the selected checkpoint exists.

## Power-interruption recovery

An unintended system shutdown occurred after 15 complete epochs while epoch 16 was only partially
processed. `results.csv` contains exactly 15 completed rows and both training logs had zero stderr
bytes. The partial epoch is not treated as evidence.

Before resuming, `last.pt` was audited locally. It stores epoch index 14, the optimizer state is
present, all 3,022,223 floating model values and 6,013,007 floating optimizer values are finite, and
the checkpoint SHA-256 is
`a7d66ea42f61217db2972b26163bb30e0af69e954de1b9a77042e3b21a0cb284`. The resume audit is retained
under ignored `artifacts/rare_class_factorial/resume_checkpoint_audit.json`.

The run then resumed in place through Ultralytics `resume=True`. The checkpoint restored the
original optimizer, scheduler, epoch, seed, `amp=false`, `batch=16`, `nbs=64`, data paths, and the
100-epoch/20-patience stopping contract. The resume log explicitly reports continuation from epoch
16 rather than a new run. Resume support is implemented in `scripts/train_yolo.py`, and the
checkpoint gate is reproducible with `scripts/inspect_resumable_checkpoint.py`.

## Results

Repeat4-640 stopped naturally after 65 completed epochs under the frozen 20-epoch patience rule;
the detector-dev-selected checkpoint is epoch 45. The requested and effective batch were both 16,
`nbs=64`, AMP remained disabled, and both training stderr files are empty. The selected checkpoint
SHA-256 is
`3a093b43231a736141179c702ca5ad9aaaf0be93e8f1b955c316374311f3bf61`. Audits found zero
non-finite values among 3,022,223 floating values in each of `best.pt` and `last.pt`. The completed
checkpoints are optimizer-stripped release checkpoints and therefore are intentionally reported as
non-resumable; the pre-resume epoch-15 checkpoint audit above remains the recovery evidence.

At the frozen detector-dev evaluation settings, repeated exposure did not recover target AP, but it
changed the low-confidence localization behavior:

| Detector-dev metric | Natural-640 | Repeat4-640 | Delta |
|---|---:|---:|---:|
| Aggregate mAP50-95 | 0.27981 | 0.27392 | -0.00589 (-2.10%) |
| Aggregate mAP50 | 0.54068 | 0.52799 | -0.01269 |
| `life_saving_appliances` AP50-95 | 0.00000 | 0.00000 | 0.00000 |
| Target matches at conf 0.001 / IoU 0.50 | 0 / 64 | 0 / 64 | 0 |
| Maximum target score | 0.00880 | 0.40440 | +0.39559 |

The post-freeze, one-time diagnostics were not used to select a checkpoint or threshold. On
calibration-fit, target localization increased from 2/364 to 14/364; on policy-tune it increased
from 0/73 to 2/73. Across the three frozen roles, Repeat4 therefore increased matched target
annotations from 2/501 to 16/501 while keeping aggregate detector-dev mAP50-95 loss within the
registered 10% relative tolerance. This supports a useful repeated-exposure main effect on proposal
localization, but not a solved detector: detector-dev target AP and IoU-0.50 recall remain zero.

Natural-1280 completed 69 epochs under the same early-stopping rule; its detector-dev-selected
checkpoint is epoch 49. Repeat4-1280 completed 59 epochs and selected epoch 39. Both 1280 cells used
effective batch 4 with `nbs=64`, AMP remained disabled, stderr is empty, and audits found zero
non-finite model values in `best.pt` and `last.pt`. The Repeat4-1280 selected-checkpoint SHA-256 is
`3c96178bab23b6ea17964ff4913364be9edbe0f68b9e4710e677c494a10bc822`.

The strict four-cell detector-dev comparison is:

| Detector-dev metric | Natural-640 | Repeat4-640 | Natural-1280 | Repeat4-1280 |
|---|---:|---:|---:|---:|
| Precision | 0.83603 | 0.60983 | 0.87141 | 0.85655 |
| Recall | 0.52121 | 0.52075 | 0.63568 | 0.61283 |
| mAP50 | 0.54068 | 0.52799 | 0.64844 | 0.63650 |
| mAP50-95 | 0.27981 | 0.27392 | 0.36570 | 0.35690 |

For aggregate mAP50-95, the exposure effect is -0.00589 at 640 and -0.00880 at 1280. The
resolution effect is +0.08589 under natural exposure and +0.08298 under Repeat4. The registered
difference-in-differences is -0.00291. Thus resolution is the dominant, robust main effect, while
repeated exposure carries a small aggregate cost at both resolutions and does not positively
interact with resolution.

| Class AP50-95 | Natural-640 | Repeat4-640 | Natural-1280 | Repeat4-1280 |
|---|---:|---:|---:|---:|
| swimmer | 0.16230 | 0.15641 | 0.26995 | 0.27018 |
| boat | 0.53350 | 0.52207 | 0.58836 | 0.58550 |
| jetski | 0.27650 | 0.26841 | 0.41041 | 0.40126 |
| life_saving_appliances | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| buoy | 0.42675 | 0.42272 | 0.55980 | 0.52756 |

At confidence 0.001 and IoU 0.50, the one-time, non-selection role diagnostics are:

| Frozen role | Target GT | Natural-640 | Repeat4-640 | Natural-1280 | Repeat4-1280 |
|---|---:|---:|---:|---:|---:|
| detector-dev | 64 | 0 | 0 | 0 | 0 |
| calibration-fit | 364 | 2 | 14 | 64 | 72 |
| policy-tune | 73 | 0 | 2 | 1 | 1 |
| Total | 501 | 2 | 16 | 65 | 73 |

Across the three roles, target proposal recall is 0.0040, 0.0319, 0.1297, and 0.1457 for the four
cells respectively. The exposure effect is +0.02794 at 640 but only +0.01597 at 1280; the proposal-
recall difference-in-differences is -0.01198. Repeated exposure therefore adds some post-freeze
low-confidence localization, but the gain is sub-additive once resolution is increased. Most
importantly, all four cells still have zero detector-dev target AP50-95 and zero target proposal
recall. The factorial isolates an unresolved detector-dev target-domain failure rather than a solved
rare-class detector.

Official validation was never used for training, checkpoint selection, threshold selection, or
factorial analysis. Machine-readable four-cell results, full per-class effects, role diagnostics,
and the comparison figure are stored in
[`results/sensitivities/rare_class_factorial/`](../results/sensitivities/rare_class_factorial/),
with the final outputs in `rare_class_factorial_2x2.json` and
`rare_class_factorial_2x2.png`.

The follow-up [four-cell target error taxonomy](TARGET_ERROR_TAXONOMY_2X2.md) shows that no
target-class prediction overlaps any detector-dev target in any cell, while the few 1280 any-class
overlaps are low-IoU `boat` or `swimmer` boxes. This rules out more whole-frame repetition as the
next useful control.
