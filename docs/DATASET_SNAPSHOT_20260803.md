# SeaDronesSee OD v2 Dataset Snapshot — 2026-08-03

Official image access: obtain a current share token from the SeaDronesSee dataset host.
Set it locally as `SEADRONESSEE_SHARE_TOKEN` when using the optional WebDAV downloader;
do not add the token to Git or share it in an issue.

Only the compressed train and validation images are required for the first release. The official
test images and 117.4 GB uncompressed image tree are intentionally excluded.

## Annotation files

| Split | Bytes | SHA-256 |
|---|---:|---|
| train | 10,037,084 | `5706B042F6B6C50267ED06076B8C27594FF4DA76569EDBD99ED53ED8BFA1DB95` |
| validation | 1,711,247 | `9E965C832263E6B9B9DD299637833E5B93924ECF9A1780B16FBD7C53E4A63F5E` |

## Verified structure

| Split | Images | Boxes | Metadata-complete images |
|---|---:|---:|---:|
| train | 8,930 | 57,760 | 6,212 |
| validation | 1,547 | 9,630 | 1,093 |

Categories are `swimmer`, `boat`, `jetski`, `life_saving_appliances`, and `buoy`. Category ID 0
is reserved as `ignored` and has no ordinary training annotations.

Flight altitude and gimbal pitch are represented by two schema variants:

- `meta.altitude` and `meta.gimbal_pitch`;
- `meta.height_above_takeoff(meter)` and `meta.gimbal_pitch(degrees)`.

Top-level `height` is the image height in pixels and must never be used as UAV altitude.

The official training annotations contain 16 exact duplicate image/category/box records across
16 images (`9560`–`9572`, `13698`, `13831`, and `13832`). All occur in detector-train; detector-dev,
calibration-fit, policy-tune, and official validation contain none. Official source JSON is kept
unchanged. Ultralytics removes these exact duplicates from its training cache automatically, and
the warning is retained in the run log.

## Analysis populations

- Primary calibration analysis: metadata-complete images only.
- Sensitivity analysis: the complete official validation split for raw, global, and
  class-conditional variants that do not use flight metadata.
- Detector training: all eligible training images.

## Official split limitation

The 1,093 metadata-complete validation images cover 31 acquisition groups. Every one of these
groups also occurs in one of the four roles derived from the official training split. Consequently,
official validation remains a frozen benchmark evaluation but is not a held-out-video test. Any
source-role analysis is labelled diagnostic, and the repository does not claim cross-video
generalization from official-validation results.
