from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import SUMMARY_JSON_PATH, SUMMARY_MD_PATH


def write_summary_report(
    dataset_summary: dict,
    metrics_df: pd.DataFrame,
    split_time: pd.Timestamp,
    cluster_summary: dict[str, float],
    multi_horizon_metrics_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> None:
    best_model = metrics_df.sort_values("rmse").iloc[0]
    best_recommendation = recommendations_df.sort_values("recommendation_rank").iloc[0]
    report = {
        "dataset_summary": dataset_summary,
        "split_time": split_time.isoformat(),
        "best_model": best_model["model"],
        "best_rmse": float(best_model["rmse"]),
        "best_mae": float(best_model["mae"]),
        "cluster_summary": cluster_summary,
        "multi_horizon_best_rmse": {
            row["forecast_horizon"]: float(row["rmse"]) for _, row in multi_horizon_metrics_df.iterrows()
        },
        "top_recommendation": best_recommendation["system_code"],
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(report, indent=2))

    multi_horizon_lines = [
        f"- {row['forecast_horizon']}: RMSE {row['rmse']:.4f}, MAE {row['mae']:.4f}, band accuracy {row['band_accuracy']:.4f}"
        for _, row in multi_horizon_metrics_df.sort_values("rmse").iterrows()
    ]
    lines = [
        "# Smart Parking Project Summary",
        "",
        "## Dataset Snapshot",
        f"- Records after cleaning: {dataset_summary['rows']}",
        f"- Parking systems: {dataset_summary['systems']}",
        f"- Time span: {dataset_summary['date_start']} to {dataset_summary['date_end']}",
        f"- Occupancy anomalies capped during cleaning: {dataset_summary['anomaly_rows']}",
        "",
        "## Model Outcome",
        f"- Time-based train/test split boundary: {split_time}",
        f"- Best model by RMSE: {best_model['model']}",
        f"- RMSE: {best_model['rmse']:.4f}",
        f"- MAE: {best_model['mae']:.4f}",
        f"- Utilization-band accuracy: {best_model['band_accuracy']:.4f}",
        "",
        "## Clustering Insight",
        f"- Optimal cluster count: {int(cluster_summary['optimal_clusters'])}",
        f"- Silhouette score: {cluster_summary['silhouette_score']:.4f}",
        f"- PCA explained variance: {cluster_summary['explained_variance_ratio_sum']:.4f}",
        "",
        "## Multi-Horizon Forecasting",
        *multi_horizon_lines,
        "",
        "## Recommendation Snapshot",
        f"- Top recommended parking system right now: {best_recommendation['system_code']}",
        f"- Predicted free spaces in 1 hour: {int(best_recommendation['predicted_available_spaces_1h'])}",
        f"- Risk band: {best_recommendation['risk_band']}",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines))
