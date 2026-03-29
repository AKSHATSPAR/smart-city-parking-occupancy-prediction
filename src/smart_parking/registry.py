from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import MODEL_REGISTRY_PATH, RF_MODEL_PATH, XGB_MODEL_PATH, LSTM_MODEL_PATH


def write_model_registry(
    metrics_df: pd.DataFrame,
    split_time: pd.Timestamp,
    feature_count: int,
    ensemble_weights: list[float],
    drift_report: dict,
    backtest_metrics_df: pd.DataFrame,
) -> None:
    backtest_records = []
    if not backtest_metrics_df.empty:
        normalized = backtest_metrics_df.copy()
        for column in normalized.columns:
            if pd.api.types.is_datetime64_any_dtype(normalized[column]):
                normalized[column] = normalized[column].astype(str)
        backtest_records = normalized.to_dict(orient="records")

    registry = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_model": metrics_df.sort_values("rmse").iloc[0]["model"],
        "train_test_split_time": split_time.isoformat(),
        "feature_count": int(feature_count),
        "artifacts": {
            "xgboost_model": str(XGB_MODEL_PATH),
            "random_forest_model": str(RF_MODEL_PATH),
            "lstm_model": str(LSTM_MODEL_PATH),
        },
        "metrics": metrics_df.to_dict(orient="records"),
        "ensemble_weights": {"random_forest": float(ensemble_weights[0]), "xgboost": float(ensemble_weights[1])},
        "drift_summary": drift_report["features"][:5],
        "backtest_summary": backtest_records,
    }
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
