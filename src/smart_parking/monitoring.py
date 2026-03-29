from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BACKTEST_METRICS_PATH, BACKTEST_PREDICTIONS_PATH, DRIFT_REPORT_PATH
from .evaluation import metrics_row
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .modeling import SplitBundle, make_xgb_pipeline


DRIFT_FEATURES = [
    "utilization",
    "queue_length",
    "parking_stress_index",
    "capacity_pressure",
    "neighbor_utilization_mean",
    "lag_utilization_30m",
    "rolling_utilization_6",
]


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    if reference.empty or current.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0
    reference_bins = pd.cut(reference, bins=cut_points, include_lowest=True)
    current_bins = pd.cut(current, bins=cut_points, include_lowest=True)
    ref_dist = reference_bins.value_counts(normalize=True, sort=False)
    cur_dist = current_bins.value_counts(normalize=True, sort=False).reindex(ref_dist.index, fill_value=0)
    ref_dist = ref_dist.clip(lower=1e-6)
    cur_dist = cur_dist.clip(lower=1e-6)
    return float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())


def build_drift_report(model_df: pd.DataFrame, split_time: pd.Timestamp) -> dict:
    train_df = model_df[model_df["target_time_slot"] <= split_time].copy()
    recent_df = model_df[model_df["target_time_slot"] > split_time].copy()

    feature_reports = []
    for feature in DRIFT_FEATURES:
        psi = _psi(train_df[feature], recent_df[feature])
        if psi >= 0.25:
            level = "high"
        elif psi >= 0.1:
            level = "moderate"
        else:
            level = "stable"
        feature_reports.append(
            {
                "feature": feature,
                "psi": round(psi, 6),
                "drift_level": level,
                "train_mean": round(float(train_df[feature].mean()), 6),
                "recent_mean": round(float(recent_df[feature].mean()), 6),
            }
        )

    report = {
        "train_period_end": split_time.isoformat(),
        "reference_rows": int(len(train_df)),
        "recent_rows": int(len(recent_df)),
        "features": sorted(feature_reports, key=lambda item: item["psi"], reverse=True),
    }
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


@dataclass
class BacktestWindow:
    train_end_time: pd.Timestamp
    test_start_time: pd.Timestamp
    test_end_time: pd.Timestamp


def _window_plan(unique_times: list[pd.Timestamp], eval_size: int = 120, windows: int = 3) -> list[BacktestWindow]:
    if len(unique_times) <= eval_size + 50:
        return []
    planned: list[BacktestWindow] = []
    for index in range(windows, 0, -1):
        test_end_idx = len(unique_times) - (index - 1) * eval_size
        test_start_idx = max(1, test_end_idx - eval_size)
        train_end_idx = test_start_idx - 1
        if train_end_idx < 100:
            continue
        planned.append(
            BacktestWindow(
                train_end_time=pd.Timestamp(unique_times[train_end_idx]),
                test_start_time=pd.Timestamp(unique_times[test_start_idx]),
                test_end_time=pd.Timestamp(unique_times[test_end_idx - 1]),
            )
        )
    return planned


def run_rolling_backtest(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_times = sorted(pd.to_datetime(model_df["target_time_slot"].unique()))
    windows = _window_plan(unique_times)
    metrics_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for window_number, window in enumerate(windows, start=1):
        train_df = model_df[model_df["target_time_slot"] <= window.train_end_time].copy()
        test_df = model_df[
            (model_df["target_time_slot"] >= window.test_start_time)
            & (model_df["target_time_slot"] <= window.test_end_time)
        ].copy()
        if train_df.empty or test_df.empty:
            continue

        model = make_xgb_pipeline()
        model.fit(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df["target_utilization_1h"])
        predictions = model.predict(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]).clip(0, 1)
        metrics = metrics_row("Rolling XGBoost", test_df["target_utilization_1h"], predictions, split_label="backtest")
        metrics["window"] = window_number
        metrics["train_end_time"] = window.train_end_time
        metrics["test_start_time"] = window.test_start_time
        metrics["test_end_time"] = window.test_end_time
        metrics_frames.append(metrics)

        baseline_predictions = test_df["utilization"].to_numpy()
        baseline_metrics = metrics_row(
            "Rolling Persistence", test_df["target_utilization_1h"], baseline_predictions, split_label="backtest"
        )
        baseline_metrics["window"] = window_number
        baseline_metrics["train_end_time"] = window.train_end_time
        baseline_metrics["test_start_time"] = window.test_start_time
        baseline_metrics["test_end_time"] = window.test_end_time
        metrics_frames.append(baseline_metrics)

        prediction_frame = test_df[
            ["system_code", "time_slot", "target_time_slot", "target_utilization_1h", "capacity"]
        ].copy()
        prediction_frame["window"] = window_number
        prediction_frame["model"] = "Rolling XGBoost"
        prediction_frame["predicted_utilization_1h"] = predictions
        prediction_frames.append(prediction_frame)

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics_df.to_csv(BACKTEST_METRICS_PATH, index=False)
    predictions_df.to_csv(BACKTEST_PREDICTIONS_PATH, index=False)
    return metrics_df, predictions_df
