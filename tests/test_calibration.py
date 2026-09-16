import numpy as np
import pandas as pd

from maritime_calibration.calibration import fit_calibrator


def test_metadata_calibrator_fits_and_predicts_probabilities():
    rng = np.random.default_rng(20260803)
    score = rng.uniform(0.05, 0.95, size=80)
    altitude = rng.uniform(10, 180, size=80)
    relative_area = rng.uniform(1e-5, 0.02, size=80)
    logits = np.log(score / (1 - score)) - 0.006 * altitude
    labels = (logits + rng.normal(0, 0.8, size=80) > 0).astype(int)
    table = pd.DataFrame(
        {
            "score": score,
            "altitude": altitude,
            "gimbal_pitch": rng.uniform(10, 85, size=80),
            "relative_area": relative_area,
            "category_id": rng.integers(1, 4, size=80),
            "is_tp": labels,
        }
    )
    calibrator = fit_calibrator(table, "full_metadata")
    probabilities = calibrator.predict(table)
    assert probabilities.shape == (80,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
