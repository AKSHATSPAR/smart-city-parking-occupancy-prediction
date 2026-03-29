from __future__ import annotations

import pandas as pd

from .analytics import (
    build_location_profiles,
    build_recommendations,
    cluster_location_profiles,
    detect_operational_anomalies,
)
from .config import (
    ANOMALIES_PATH,
    BACKTEST_METRICS_PATH,
    CLEAN_DATA_PATH,
    DRIFT_REPORT_PATH,
    FEATURE_DATA_PATH,
    LATEST_FORECAST_PATH,
    LOCATION_PROFILE_PATH,
    MODEL_METRICS_PATH,
    MODEL_REGISTRY_PATH,
    MULTI_HORIZON_METRICS_PATH,
    MULTI_HORIZON_PREDICTIONS_PATH,
    RECOMMENDATIONS_PATH,
    SPATIAL_NEIGHBOR_PATH,
    TEST_PREDICTIONS_PATH,
)
from .data import build_clean_dataset, dataset_summary, load_raw_dataset
from .database import build_sqlite_database
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model_dataset
from .lstm import build_sequence_bundle, train_lstm_forecaster
from .monitoring import build_drift_report, run_rolling_backtest
from .modeling import temporal_train_test_split, train_classical_models, train_multi_horizon_xgboost
from .registry import write_model_registry
from .reporting import write_summary_report
from .visualization import (
    plot_actual_vs_predicted,
    plot_anomalies,
    plot_clusters,
    plot_feature_importance,
    plot_model_comparison,
    plot_multi_horizon_metrics,
    plot_queue_relationship,
    plot_traffic_relationship,
    plot_uncertainty_profile,
    plot_utilization_heatmap,
)


def run_pipeline() -> dict[str, str]:
    raw_df = load_raw_dataset()
    clean_df = build_clean_dataset(raw_df)

    model_df, neighbor_graph = build_model_dataset(clean_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    model_df.to_csv(FEATURE_DATA_PATH, index=False)
    neighbor_graph.to_csv(SPATIAL_NEIGHBOR_PATH, index=False)

    split = temporal_train_test_split(model_df, train_ratio=0.8)
    classical_metrics, classical_predictions, models = train_classical_models(split)

    sequence_bundle = build_sequence_bundle(clean_df, split.split_time)
    lstm_metrics, lstm_predictions = train_lstm_forecaster(sequence_bundle)

    metrics_df = pd.concat([classical_metrics, lstm_metrics], ignore_index=True).sort_values("rmse")
    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)

    predictions_df = pd.concat([classical_predictions, lstm_predictions], ignore_index=True)
    predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    multi_horizon_metrics, multi_horizon_predictions = train_multi_horizon_xgboost(model_df)
    multi_horizon_metrics.to_csv(MULTI_HORIZON_METRICS_PATH, index=False)
    multi_horizon_predictions.to_csv(MULTI_HORIZON_PREDICTIONS_PATH, index=False)

    location_profiles = build_location_profiles(clean_df)
    clustered_profiles, cluster_summary = cluster_location_profiles(location_profiles)
    clustered_profiles.to_csv(LOCATION_PROFILE_PATH, index=False)

    latest_rows = (
        model_df.sort_values("time_slot")
        .groupby("system_code", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    xgb_pipeline = models["xgboost"]
    rf_pipeline = models["random_forest"]
    latest_features = latest_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    latest_rows["predicted_utilization_1h"] = xgb_pipeline.predict(latest_features)
    latest_rows["rf_predicted_utilization_1h"] = rf_pipeline.predict(latest_features)
    latest_rows["ensemble_predicted_utilization_1h"] = (
        models["ensemble_weights"][0] * latest_rows["rf_predicted_utilization_1h"]
        + models["ensemble_weights"][1] * latest_rows["predicted_utilization_1h"]
    )
    latest_rows["prediction_interval_lower"] = (
        latest_rows["ensemble_predicted_utilization_1h"] - models["ensemble_interval_radius"]
    ).clip(0, 1)
    latest_rows["prediction_interval_upper"] = (
        latest_rows["ensemble_predicted_utilization_1h"] + models["ensemble_interval_radius"]
    ).clip(0, 1)
    latest_rows["uncertainty_width"] = latest_rows["prediction_interval_upper"] - latest_rows["prediction_interval_lower"]
    latest_rows["predicted_available_spaces_1h"] = (
        latest_rows["capacity"] * (1 - latest_rows["ensemble_predicted_utilization_1h"].clip(0, 1))
    ).round(0)
    latest_rows["predicted_utilization_1h"] = latest_rows["ensemble_predicted_utilization_1h"]
    latest_forecast = latest_rows[
        [
            "system_code",
            "time_slot",
            "target_time_slot",
            "capacity",
            "utilization",
            "capacity_pressure",
            "predicted_utilization_1h",
            "predicted_available_spaces_1h",
            "prediction_interval_lower",
            "prediction_interval_upper",
            "uncertainty_width",
        ]
    ].copy()
    latest_forecast.to_csv(LATEST_FORECAST_PATH, index=False)

    recommendations_df = build_recommendations(latest_rows)
    recommendations_df.to_csv(RECOMMENDATIONS_PATH, index=False)

    anomalies_df = detect_operational_anomalies(model_df)
    anomalies_df.to_csv(ANOMALIES_PATH, index=False)

    drift_report = build_drift_report(model_df, split.split_time)
    backtest_metrics_df, _backtest_predictions_df = run_rolling_backtest(model_df)
    write_model_registry(
        metrics_df=metrics_df,
        split_time=split.split_time,
        feature_count=len(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
        ensemble_weights=models["ensemble_weights"],
        drift_report=drift_report,
        backtest_metrics_df=backtest_metrics_df,
    )

    plot_utilization_heatmap(clean_df)
    plot_traffic_relationship(clean_df)
    plot_queue_relationship(clean_df)
    plot_clusters(clustered_profiles)
    plot_model_comparison(metrics_df)
    plot_actual_vs_predicted(predictions_df, "XGBoost", "xgboost_actual_vs_predicted.png")
    plot_actual_vs_predicted(predictions_df, "LSTM", "lstm_actual_vs_predicted.png")
    plot_actual_vs_predicted(predictions_df, "Weighted Ensemble", "ensemble_actual_vs_predicted.png")
    plot_feature_importance(xgb_pipeline)
    plot_multi_horizon_metrics(multi_horizon_metrics)
    plot_uncertainty_profile(predictions_df)
    plot_anomalies(anomalies_df)

    build_sqlite_database(
        raw_df=raw_df,
        clean_df=clean_df,
        model_df=model_df,
        neighbor_graph=neighbor_graph,
        location_profiles=clustered_profiles,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        multi_horizon_metrics_df=multi_horizon_metrics,
        multi_horizon_predictions_df=multi_horizon_predictions,
        recommendations_df=recommendations_df,
        anomalies_df=anomalies_df,
        backtest_metrics_df=backtest_metrics_df,
    )

    summary = dataset_summary(clean_df)
    write_summary_report(summary, metrics_df, split.split_time, cluster_summary, multi_horizon_metrics, recommendations_df)

    return {
        "clean_data_path": str(CLEAN_DATA_PATH),
        "feature_data_path": str(FEATURE_DATA_PATH),
        "metrics_path": str(MODEL_METRICS_PATH),
        "predictions_path": str(TEST_PREDICTIONS_PATH),
        "latest_forecast_path": str(LATEST_FORECAST_PATH),
        "location_profile_path": str(LOCATION_PROFILE_PATH),
        "multi_horizon_metrics_path": str(MULTI_HORIZON_METRICS_PATH),
        "recommendations_path": str(RECOMMENDATIONS_PATH),
        "anomalies_path": str(ANOMALIES_PATH),
        "backtest_metrics_path": str(BACKTEST_METRICS_PATH),
        "drift_report_path": str(DRIFT_REPORT_PATH),
        "model_registry_path": str(MODEL_REGISTRY_PATH),
    }
