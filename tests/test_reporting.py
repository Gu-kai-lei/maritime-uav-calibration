from __future__ import annotations

from scripts.build_public_results import detector_summary, first_metric_collapse, parse_training_log
from scripts.compare_calibration_sensitivity import interval_status
from scripts.evaluate_detector_resolution import class_metric_rows
from scripts.analyze_stability_control import stability_gate
from scripts.compare_detector_replays import compare_results_tables, find_collapse_epoch
from scripts.train_yolo import best_epoch_summary
import pandas as pd


def test_parse_training_log_extracts_batch_stop_and_final_metrics(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\x1b[34mAutoBatch:\x1b[0m Using batch-size 16 for CUDA:0\n"
        "EarlyStopping: Training stopped early as no improvement observed. "
        "Best results observed at epoch 8, best model saved as best.pt.\n"
        "28 epochs completed in 1.119 hours.\n"
        "all 893 5437 0.774 0.486 0.483 0.238\n"
        "life_saving_appliances 64 64 1 0 0 0\n",
        encoding="utf-8",
    )

    result = parse_training_log(log)

    assert result["effective_batch_size"] == 16
    assert result["best_epoch"] == 8
    assert result["completed_epochs"] == 28
    assert result["training_hours"] == 1.119
    assert result["detector_dev_metrics"]["all"]["map50_95"] == 0.238
    assert result["detector_dev_metrics"]["life_saving_appliances"]["recall"] == 0.0


def test_best_epoch_summary_strips_ultralytics_csv_headers(tmp_path) -> None:
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), "
        "metrics/mAP50-95(B)\n"
        "0,0.6,0.4,0.3,0.1\n"
        "1,0.7,0.5,0.4,0.2\n",
        encoding="utf-8",
    )

    summary = best_epoch_summary(results)

    assert summary == {
        "epoch": 1,
        "precision": 0.7,
        "recall": 0.5,
        "map50": 0.4,
        "map50_95": 0.2,
    }


def test_replay_comparison_ignores_runtime_but_requires_exact_metrics() -> None:
    primary = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "time": [10.0, 20.0, 30.0],
            "metrics/mAP50-95(B)": [0.1, 0.2, 0.01],
            "train/box_loss": [2.0, 1.5, 4.0],
        }
    )
    replay = primary.copy()
    replay["time"] = [8.0, 16.0, 24.0]

    comparison = compare_results_tables(primary, replay)

    assert comparison["all_fields_exact"]
    assert "time" not in comparison["fields_compared"]
    assert find_collapse_epoch(primary) == 3


def test_stability_gate_requires_no_collapse_finite_weights_and_empty_stderr() -> None:
    finite = {"nonfinite_values": 0}

    assert stability_gate(None, finite, finite, 0)["passed"]
    assert not stability_gate(9, finite, finite, 0)["passed"]
    assert not stability_gate(None, {"nonfinite_values": 1}, finite, 0)["passed"]
    assert not stability_gate(None, finite, finite, 20)["passed"]


def test_public_detector_summary_handles_a_fixed_batch_stable_run(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        """{
  "model_initialization_sha256": "model-hash",
  "best_weights_sha256": "weights-hash",
  "data_yaml_sha256": "data-hash",
  "imgsz": 640,
  "seed": 7,
  "epochs": 100,
  "epochs_completed": 3,
  "batch_requested": 16,
  "batch_effective": 16,
  "patience": 20,
  "amp": false,
  "optimizer": "auto",
  "python": "3.12",
  "platform": "Windows",
  "torch": "2.6",
  "ultralytics": "8.4",
  "cuda_available": true,
  "gpu": "test GPU"
}\n""",
        encoding="utf-8",
    )
    results = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "metrics/precision(B)": [0.5, 0.6, 0.59],
            "metrics/recall(B)": [0.4, 0.5, 0.49],
            "metrics/mAP50(B)": [0.3, 0.4, 0.39],
            "metrics/mAP50-95(B)": [0.1, 0.2, 0.19],
        }
    )
    results.to_csv(run / "results.csv", index=False)
    log = tmp_path / "train.log"
    log.write_text(
        "3 epochs completed in 0.1 hours.\n"
        "all 10 20 0.6 0.5 0.4 0.2\n",
        encoding="utf-8",
    )

    summary = detector_summary(run, log)

    assert summary["requested_batch"] == 16
    assert summary["effective_batch_size"] == 16
    assert summary["amp"] is False
    assert summary["metric_collapse_epoch"] is None
    assert "No metric-collapse event" in summary["training_stability_note"]
    assert first_metric_collapse(results) is None


def test_interval_status_requires_the_entire_interval_on_one_side_of_zero() -> None:
    assert interval_status(-0.02, -0.001) == "better_than_raw"
    assert interval_status(0.001, 0.02) == "worse_than_raw"
    assert interval_status(-0.01, 0.01) == "inconclusive"


def test_class_metric_rows_preserves_class_indices_and_instance_counts() -> None:
    rows = class_metric_rows(
        {0: "swimmer", 2: "jetski"},
        pd.Series([0, 2]).to_numpy(),
        pd.Series([0.7, 0.8]).to_numpy(),
        pd.Series([0.5, 0.6]).to_numpy(),
        pd.Series([0.4, 0.5]).to_numpy(),
        pd.Series([0.2, 0.3]).to_numpy(),
        pd.Series([10, 0, 4]).to_numpy(),
    )

    assert rows["swimmer"]["instances"] == 10
    assert rows["jetski"]["instances"] == 4
    assert rows["jetski"]["map50_95"] == 0.3
