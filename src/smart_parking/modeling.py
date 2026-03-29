from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .config import RANDOM_STATE, RF_MODEL_PATH, XGB_MODEL_PATH
from .evaluation import metrics_row
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


os.environ.setdefault("OMP_NUM_THREADS", "1")


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_time: pd.Timestamp


def temporal_train_test_split(model_df: pd.DataFrame, train_ratio: float = 0.8) -> SplitBundle:
    unique_times = sorted(model_df["target_time_slot"].unique())
    split_index = max(1, int(len(unique_times) * train_ratio))
    split_time = pd.Timestamp(unique_times[split_index - 1])
    train_df = model_df[model_df["target_time_slot"] <= split_time].copy()
    test_df = model_df[model_df["target_time_slot"] > split_time].copy()
    return SplitBundle(train_df=train_df, test_df=test_df, split_time=split_time)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _make_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=350,
                    max_depth=14,
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _make_xgb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=350,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_rf_pipeline() -> Pipeline:
    return _make_rf_pipeline()


def make_xgb_pipeline() -> Pipeline:
    return _make_xgb_pipeline()


def _calibration_split(train_df: pd.DataFrame, ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_times = sorted(train_df["target_time_slot"].unique())
    calibration_count = max(1, int(len(unique_times) * ratio))
    calibration_times = set(unique_times[-calibration_count:])
    core_train_df = train_df[~train_df["target_time_slot"].isin(calibration_times)].copy()
    calibration_df = train_df[train_df["target_time_slot"].isin(calibration_times)].copy()
    return core_train_df, calibration_df


def _prediction_frame(test_df: pd.DataFrame, model_name: str, predictions: np.ndarray) -> pd.DataFrame:
    frame = test_df[
        [
            "system_code",
            "time_slot",
            "target_time_slot",
            "capacity",
            "available_spaces",
            "target_available_spaces_1h",
            "utilization",
            "target_utilization_1h",
        ]
    ].copy()
    frame["model"] = model_name
    frame["predicted_utilization_1h"] = predictions.clip(0, 1)
    frame["predicted_available_spaces_1h"] = (frame["capacity"] * (1 - frame["predicted_utilization_1h"])).round(0)
    return frame


def train_classical_models(split: SplitBundle) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_df = split.train_df
    test_df = split.test_df
    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["target_utilization_1h"]

    core_train_df, calibration_df = _calibration_split(train_df)
    x_core = core_train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_core = core_train_df["target_utilization_1h"]
    x_calibration = calibration_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_calibration = calibration_df["target_utilization_1h"]

    baseline_pred = test_df["utilization"].to_numpy()

    rf_eval_pipeline = _make_rf_pipeline()
    rf_eval_pipeline.fit(x_core, y_core)
    rf_calibration_pred = rf_eval_pipeline.predict(x_calibration)
    rf_pred = rf_eval_pipeline.predict(x_test)

    xgb_eval_pipeline = _make_xgb_pipeline()
    xgb_eval_pipeline.fit(x_core, y_core)
    xgb_calibration_pred = xgb_eval_pipeline.predict(x_calibration)
    xgb_pred = xgb_eval_pipeline.predict(x_test)

    rf_calibration_rmse = float(np.sqrt(np.mean((y_calibration.to_numpy() - rf_calibration_pred) ** 2)))
    xgb_calibration_rmse = float(np.sqrt(np.mean((y_calibration.to_numpy() - xgb_calibration_pred) ** 2)))
    inverse_errors = np.array([1 / max(rf_calibration_rmse, 1e-6), 1 / max(xgb_calibration_rmse, 1e-6)])
    ensemble_weights = inverse_errors / inverse_errors.sum()
    ensemble_calibration_pred = (
        ensemble_weights[0] * rf_calibration_pred + ensemble_weights[1] * xgb_calibration_pred
    )
    ensemble_pred = ensemble_weights[0] * rf_pred + ensemble_weights[1] * xgb_pred

    xgb_interval_radius = float(np.quantile(np.abs(y_calibration.to_numpy() - xgb_calibration_pred), 0.9))
    ensemble_interval_radius = float(np.quantile(np.abs(y_calibration.to_numpy() - ensemble_calibration_pred), 0.9))

    rf_pipeline = clone(rf_eval_pipeline)
    rf_pipeline.fit(x_train, train_df["target_utilization_1h"])
    joblib.dump(rf_pipeline, RF_MODEL_PATH)

    xgb_pipeline = clone(xgb_eval_pipeline)
    xgb_pipeline.fit(x_train, train_df["target_utilization_1h"])
    joblib.dump(xgb_pipeline, XGB_MODEL_PATH)

    metrics = pd.concat(
        [
            metrics_row("Persistence Baseline", y_test, baseline_pred),
            metrics_row("Random Forest", y_test, rf_pred),
            metrics_row("XGBoost", y_test, xgb_pred),
            metrics_row("Weighted Ensemble", y_test, ensemble_pred),
        ],
        ignore_index=True,
    )

    prediction_frames = []
    for model_name, predictions in [
        ("Persistence Baseline", baseline_pred),
        ("Random Forest", rf_pred),
        ("XGBoost", xgb_pred),
        ("Weighted Ensemble", ensemble_pred),
    ]:
        frame = _prediction_frame(test_df, model_name, predictions)
        if model_name == "XGBoost":
            frame["prediction_interval_lower"] = (frame["predicted_utilization_1h"] - xgb_interval_radius).clip(0, 1)
            frame["prediction_interval_upper"] = (frame["predicted_utilization_1h"] + xgb_interval_radius).clip(0, 1)
            frame["uncertainty_width"] = frame["prediction_interval_upper"] - frame["prediction_interval_lower"]
        elif model_name == "Weighted Ensemble":
            frame["prediction_interval_lower"] = (
                frame["predicted_utilization_1h"] - ensemble_interval_radius
            ).clip(0, 1)
            frame["prediction_interval_upper"] = (
                frame["predicted_utilization_1h"] + ensemble_interval_radius
            ).clip(0, 1)
            frame["uncertainty_width"] = frame["prediction_interval_upper"] - frame["prediction_interval_lower"]
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    models = {
        "random_forest": rf_pipeline,
        "xgboost": xgb_pipeline,
        "random_forest_eval": rf_eval_pipeline,
        "xgboost_eval": xgb_eval_pipeline,
        "ensemble_weights": ensemble_weights.tolist(),
        "xgb_interval_radius": xgb_interval_radius,
        "ensemble_interval_radius": ensemble_interval_radius,
    }
    return metrics, predictions, models


def train_multi_horizon_xgboost(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_specs = [
        ("30 minutes", "target_utilization_30m", "target_time_slot_30m"),
        ("1 hour", "target_utilization_1h", "target_time_slot"),
        ("2 hours", "target_utilization_2h", "target_time_slot_2h"),
    ]
    metrics_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon_name, target_col, target_time_col in horizon_specs:
        current_df = model_df.dropna(subset=[target_col, target_time_col]).copy()
        unique_times = sorted(current_df[target_time_col].unique())
        split_index = max(1, int(len(unique_times) * 0.8))
        split_time = pd.Timestamp(unique_times[split_index - 1])
        train_df = current_df[current_df[target_time_col] <= split_time].copy()
        test_df = current_df[current_df[target_time_col] > split_time].copy()

        model = _make_xgb_pipeline()
        model.fit(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[target_col])
        predictions = model.predict(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]).clip(0, 1)

        current_metrics = metrics_row(f"XGBoost {horizon_name}", test_df[target_col], predictions)
        current_metrics["forecast_horizon"] = horizon_name
        current_metrics["target_column"] = target_col
        metrics_frames.append(current_metrics)

        current_predictions = test_df[["system_code", "time_slot", target_time_col, "capacity", target_col]].copy()
        current_predictions["forecast_horizon"] = horizon_name
        current_predictions["predicted_utilization"] = predictions
        current_predictions["predicted_available_spaces"] = (
            current_predictions["capacity"] * (1 - current_predictions["predicted_utilization"])
        ).round(0)
        prediction_frames.append(current_predictions)

    return pd.concat(metrics_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)
