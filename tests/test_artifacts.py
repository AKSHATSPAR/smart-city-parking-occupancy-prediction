from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_parking.config import (
    BACKTEST_METRICS_PATH,
    DRIFT_REPORT_PATH,
    MODEL_METRICS_PATH,
    MODEL_REGISTRY_PATH,
    MULTI_HORIZON_METRICS_PATH,
    RECOMMENDATIONS_PATH,
)


def test_core_artifacts_exist():
    for path in [
        MODEL_METRICS_PATH,
        MULTI_HORIZON_METRICS_PATH,
        RECOMMENDATIONS_PATH,
        BACKTEST_METRICS_PATH,
        DRIFT_REPORT_PATH,
        MODEL_REGISTRY_PATH,
    ]:
        assert path.exists(), f"Missing artifact: {path}"


def test_registry_and_drift_payloads_are_valid():
    registry = json.loads(MODEL_REGISTRY_PATH.read_text())
    drift = json.loads(DRIFT_REPORT_PATH.read_text())
    assert "champion_model" in registry
    assert "features" in drift
    assert len(drift["features"]) >= 1


def test_multi_horizon_metrics_have_expected_horizons():
    metrics = pd.read_csv(MULTI_HORIZON_METRICS_PATH)
    horizons = set(metrics["forecast_horizon"].tolist())
    assert {"30 minutes", "1 hour", "2 hours"}.issubset(horizons)
