# Related Work and Claim Boundary

This project is not claiming that metadata-aware vision, multivariate detector calibration, or
altitude-aware maritime detection is new in isolation.

## Established work

- [SeaDronesSee (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022/papers/Varga_SeaDronesSee_A_Maritime_Benchmark_for_Detecting_Humans_in_Open_Water_WACV_2022_paper.pdf)
  introduced the maritime UAV benchmark and explicitly included altitude and viewing-angle
  metadata.
- [MaCVi 2023 challenge report](https://arxiv.org/abs/2211.13508) introduced Object Detection v2,
  expanded the classes and footage, and reported challenge methods and results.
- [Küppers et al., CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf)
  showed that detector confidence calibration can condition on localization and box-scale
  features rather than score alone.
- [Pathiraja et al., 2023](https://arxiv.org/abs/2306.08271) studied joint multiclass confidence
  and localization calibration as a train-time method.
- [Ahmed and Pizarro, 2025](https://arxiv.org/abs/2511.19728) used altitude-aware dynamic tiling
  for maritime small-object detection. Dynamic tiling is therefore outside this repository's
  novelty claim.

## Project-specific question

The testable question is narrower: on SeaDronesSee ODv2, does post-hoc calibration that adds UAV
altitude and gimbal pitch to score, class, and predicted scale improve reliability and recall under
a fixed false-positive-per-image budget, especially for high-altitude small targets?

Potential contributions remain conditional until the frozen evaluation is run:

1. a source-disjoint, four-role protocol that prevents adjacent-frame and model-selection leakage;
2. an interpretable ablation from raw score through class-, altitude-, angle-, and scale-aware
   logistic calibration;
3. operating thresholds chosen on policy-tune and transferred unchanged to official validation;
4. metadata, scale, and class slices with grouped uncertainty estimates.

Negative or null results are publishable project outcomes and must not be hidden by changing the
protocol after inspecting official-validation results.
