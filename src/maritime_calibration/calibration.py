from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VARIANT_SPECS = {
    "global_logistic": {"numeric": ["score_logit"], "categorical": []},
    "class_logistic": {"numeric": ["score_logit"], "categorical": ["category_id"]},
    "altitude": {
        "numeric": ["score_logit", "log_altitude"],
        "categorical": ["category_id"],
    },
    "altitude_gimbal": {
        "numeric": ["score_logit", "log_altitude", "gimbal_sin", "gimbal_cos"],
        "categorical": ["category_id"],
    },
    "full_metadata": {
        "numeric": [
            "score_logit",
            "log_altitude",
            "gimbal_sin",
            "gimbal_cos",
            "log_relative_area",
        ],
        "categorical": ["category_id"],
    },
}
FEATURE_VARIANTS = tuple(VARIANT_SPECS)


def add_derived_features(table: pd.DataFrame) -> pd.DataFrame:
    required = {"score", "relative_area", "altitude", "gimbal_pitch"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")

    output = table.copy()
    score = output["score"].astype(float).clip(1e-6, 1 - 1e-6)
    output["score_logit"] = np.log(score / (1 - score))
    altitude = pd.to_numeric(output["altitude"], errors="coerce")
    output["log_altitude"] = np.log1p(altitude.clip(lower=0))
    angle = np.deg2rad(pd.to_numeric(output["gimbal_pitch"], errors="coerce"))
    output["gimbal_sin"] = np.sin(angle)
    output["gimbal_cos"] = np.cos(angle)
    relative_area = pd.to_numeric(output["relative_area"], errors="coerce").clip(lower=1e-12)
    output["log_relative_area"] = np.log(relative_area)
    return output


@dataclass
class DetectionCalibrator:
    variant: str
    model: Pipeline

    @property
    def features(self) -> list[str]:
        spec = VARIANT_SPECS[self.variant]
        return [*spec["numeric"], *spec["categorical"]]

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        featured = add_derived_features(table)
        return self.model.predict_proba(featured[self.features])[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def fit_calibrator(table: pd.DataFrame, variant: str) -> DetectionCalibrator:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"unknown variant={variant}; choose from {sorted(VARIANT_SPECS)}")
    if "is_tp" not in table:
        raise ValueError("calibration table must contain is_tp")
    labels = table["is_tp"].astype(int)
    if labels.nunique() < 2:
        raise ValueError("calibration requires both correct and incorrect detections")

    featured = add_derived_features(table)
    spec = VARIANT_SPECS[variant]
    numeric_features = spec["numeric"]
    categorical_features = spec["categorical"]
    features = [*numeric_features, *categorical_features]
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    transformers = [("numeric", numeric, numeric_features)]
    if categorical_features:
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical, categorical_features))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight=None, random_state=20260803),
            ),
        ]
    )
    model.fit(featured[features], labels)
    return DetectionCalibrator(variant=variant, model=model)
