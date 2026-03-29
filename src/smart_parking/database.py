from __future__ import annotations

import sqlite3

import pandas as pd

from .config import SQLITE_DB_PATH


def build_sqlite_database(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    model_df: pd.DataFrame,
    neighbor_graph: pd.DataFrame,
    location_profiles: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    multi_horizon_metrics_df: pd.DataFrame,
    multi_horizon_predictions_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    backtest_metrics_df: pd.DataFrame,
) -> None:
    with sqlite3.connect(SQLITE_DB_PATH) as connection:
        raw_df.to_sql("raw_parking", connection, if_exists="replace", index=False)
        clean_df.to_sql("cleaned_parking", connection, if_exists="replace", index=False)
        model_df.to_sql("model_dataset", connection, if_exists="replace", index=False)
        neighbor_graph.to_sql("spatial_neighbor_graph", connection, if_exists="replace", index=False)
        location_profiles.to_sql("location_profiles", connection, if_exists="replace", index=False)
        metrics_df.to_sql("model_metrics", connection, if_exists="replace", index=False)
        predictions_df.to_sql("test_predictions", connection, if_exists="replace", index=False)
        multi_horizon_metrics_df.to_sql("multi_horizon_metrics", connection, if_exists="replace", index=False)
        multi_horizon_predictions_df.to_sql("multi_horizon_predictions", connection, if_exists="replace", index=False)
        recommendations_df.to_sql("parking_recommendations", connection, if_exists="replace", index=False)
        anomalies_df.to_sql("demand_anomalies", connection, if_exists="replace", index=False)
        backtest_metrics_df.to_sql("rolling_backtest_metrics", connection, if_exists="replace", index=False)

        connection.executescript(
            """
            DROP VIEW IF EXISTS vw_daily_system_summary;
            CREATE VIEW vw_daily_system_summary AS
            SELECT
                system_code,
                date(time_slot) AS summary_date,
                AVG(utilization) AS avg_utilization,
                MAX(utilization) AS max_utilization,
                AVG(queue_length) AS avg_queue_length,
                SUM(occupancy_anomaly_flag) AS anomaly_rows
            FROM cleaned_parking
            GROUP BY system_code, date(time_slot);

            DROP VIEW IF EXISTS vw_peak_periods;
            CREATE VIEW vw_peak_periods AS
            SELECT
                system_code,
                hour,
                AVG(utilization) AS avg_utilization,
                AVG(queue_length) AS avg_queue_length
            FROM cleaned_parking
            GROUP BY system_code, hour
            ORDER BY avg_utilization DESC, avg_queue_length DESC;

            DROP VIEW IF EXISTS vw_recommended_parking;
            CREATE VIEW vw_recommended_parking AS
            SELECT
                recommendation_rank,
                system_code,
                predicted_available_spaces_1h,
                predicted_utilization_1h,
                risk_band,
                balanced_score
            FROM parking_recommendations
            ORDER BY recommendation_rank ASC, balanced_score DESC;

            DROP VIEW IF EXISTS vw_multi_horizon_summary;
            CREATE VIEW vw_multi_horizon_summary AS
            SELECT
                forecast_horizon,
                AVG(rmse) AS avg_rmse,
                AVG(mae) AS avg_mae,
                AVG(band_accuracy) AS avg_band_accuracy
            FROM multi_horizon_metrics
            GROUP BY forecast_horizon;

            DROP VIEW IF EXISTS vw_backtest_summary;
            CREATE VIEW vw_backtest_summary AS
            SELECT
                model,
                AVG(rmse) AS avg_rmse,
                AVG(mae) AS avg_mae,
                AVG(band_accuracy) AS avg_band_accuracy
            FROM rolling_backtest_metrics
            GROUP BY model;
            """
        )
