from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .config import UTILIZATION_BINS, UTILIZATION_LABELS


def _bands(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.cut(
        pd.Series(values),
        bins=UTILIZATION_BINS,
        labels=UTILIZATION_LABELS,
        include_lowest=True,
        ordered=True,
    ).astype(str)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100),
        "band_accuracy": float(accuracy_score(_bands(y_true), _bands(y_pred))),
        "weighted_f1": float(f1_score(_bands(y_true), _bands(y_pred), average="weighted")),
    }


def metrics_row(model_name: str, y_true, y_pred, split_label: str = "test") -> pd.DataFrame:
    metrics = regression_metrics(y_true, y_pred)
    metrics["model"] = model_name
    metrics["split"] = split_label
    return pd.DataFrame([metrics])

