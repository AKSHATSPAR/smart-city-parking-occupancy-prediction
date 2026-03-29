from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "parkingStream_2.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DB_DIR = ARTIFACTS_DIR / "db"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "parking_cleaned.csv"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "parking_model_dataset.csv"
LOCATION_PROFILE_PATH = PROCESSED_DATA_DIR / "location_profiles.csv"
SPATIAL_NEIGHBOR_PATH = PROCESSED_DATA_DIR / "spatial_neighbor_graph.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
LATEST_FORECAST_PATH = REPORTS_DIR / "latest_forecast.csv"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
MULTI_HORIZON_METRICS_PATH = REPORTS_DIR / "multi_horizon_metrics.csv"
MULTI_HORIZON_PREDICTIONS_PATH = REPORTS_DIR / "multi_horizon_predictions.csv"
RECOMMENDATIONS_PATH = REPORTS_DIR / "parking_recommendations.csv"
ANOMALIES_PATH = REPORTS_DIR / "demand_anomalies.csv"
BACKTEST_METRICS_PATH = REPORTS_DIR / "rolling_backtest_metrics.csv"
BACKTEST_PREDICTIONS_PATH = REPORTS_DIR / "rolling_backtest_predictions.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
MODEL_REGISTRY_PATH = REPORTS_DIR / "model_registry.json"
SUMMARY_JSON_PATH = REPORTS_DIR / "project_summary.json"
SUMMARY_MD_PATH = REPORTS_DIR / "project_summary.md"
SQLITE_DB_PATH = DB_DIR / "smart_parking.db"
LIVE_OPS_DB_PATH = DB_DIR / "live_ops.db"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_pipeline.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_forecaster.pt"

TIME_SLOT_FREQUENCY = "30min"
FORECAST_HORIZON_STEPS = 2
SEQUENCE_LENGTH = 6
RANDOM_STATE = 42
NEIGHBOR_COUNT = 3

UTILIZATION_BINS = [0.0, 0.4, 0.7, 0.9, 1.01]
UTILIZATION_LABELS = ["Low", "Moderate", "High", "Critical"]
