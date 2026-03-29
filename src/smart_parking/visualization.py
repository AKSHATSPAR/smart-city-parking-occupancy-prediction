from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import PLOTS_DIR


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", palette="crest")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14


def plot_utilization_heatmap(clean_df: pd.DataFrame) -> None:
    _apply_style()
    pivot = (
        clean_df.groupby(["system_code", "hour"])["utilization"]
        .mean()
        .reset_index()
        .pivot(index="system_code", columns="hour", values="utilization")
        .sort_index()
    )
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.2)
    plt.title("Average Parking Utilization by Location and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Parking System")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "utilization_heatmap.png", dpi=200)
    plt.close()


def plot_traffic_relationship(clean_df: pd.DataFrame) -> None:
    _apply_style()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=clean_df, x="traffic_condition_nearby", y="utilization", order=["low", "average", "high"])
    plt.title("Utilization Distribution by Nearby Traffic Condition")
    plt.xlabel("Traffic Condition")
    plt.ylabel("Utilization")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "traffic_vs_utilization.png", dpi=200)
    plt.close()


def plot_queue_relationship(clean_df: pd.DataFrame) -> None:
    _apply_style()
    sampled = clean_df.sample(min(len(clean_df), 3000), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sampled, x="queue_length", y="utilization", hue="traffic_condition_nearby", alpha=0.7)
    plt.title("Queue Length vs Utilization")
    plt.xlabel("Queue Length")
    plt.ylabel("Utilization")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "queue_vs_utilization.png", dpi=200)
    plt.close()


def plot_clusters(location_profiles: pd.DataFrame) -> None:
    _apply_style()
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=location_profiles,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        size="mean_utilization",
        palette="Set2",
        sizes=(80, 250),
    )
    for _, row in location_profiles.iterrows():
        plt.text(row["pca_1"] + 0.03, row["pca_2"] + 0.03, row["system_code"], fontsize=8)
    plt.title("Parking Location Clusters (PCA Projection)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "location_clusters.png", dpi=220)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    _apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=metrics_df, x="model", y="rmse", hue="model", ax=axes[0], palette="mako", legend=False)
    axes[0].set_title("Model RMSE Comparison")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=metrics_df,
        x="model",
        y="band_accuracy",
        hue="model",
        ax=axes[1],
        palette="viridis",
        legend=False,
    )
    axes[1].set_title("Utilization Band Accuracy")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Accuracy")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=220)
    plt.close()


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, model_name: str, output_name: str) -> None:
    _apply_style()
    current = predictions_df[predictions_df["model"] == model_name].copy()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=current,
        x="target_utilization_1h",
        y="predicted_utilization_1h",
        hue="system_code",
        s=35,
        alpha=0.7,
        legend=False,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Actual Utilization")
    plt.ylabel("Predicted Utilization")
    plt.title(f"{model_name}: Actual vs Predicted Utilization")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=220)
    plt.close()


def plot_feature_importance(xgb_pipeline) -> None:
    _apply_style()
    model = xgb_pipeline.named_steps["model"]
    feature_names = xgb_pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    top = importances.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=top, x="importance", y="feature", hue="feature", palette="rocket", legend=False)
    plt.title("Top XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgboost_feature_importance.png", dpi=220)
    plt.close()


def plot_multi_horizon_metrics(metrics_df: pd.DataFrame) -> None:
    _apply_style()
    plt.figure(figsize=(11, 6))
    sns.barplot(data=metrics_df, x="forecast_horizon", y="rmse", hue="forecast_horizon", palette="flare", legend=False)
    plt.title("XGBoost RMSE Across Forecast Horizons")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "multi_horizon_rmse.png", dpi=220)
    plt.close()


def plot_uncertainty_profile(predictions_df: pd.DataFrame) -> None:
    _apply_style()
    current = predictions_df[predictions_df["model"].isin(["XGBoost", "Weighted Ensemble"])].copy()
    current = current.dropna(subset=["uncertainty_width"])
    if current.empty:
        return
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=current, x="model", y="uncertainty_width", hue="model", palette="cubehelix", legend=False)
    plt.title("Prediction Interval Width by Advanced Model")
    plt.xlabel("")
    plt.ylabel("Interval Width")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "forecast_uncertainty.png", dpi=220)
    plt.close()


def plot_anomalies(anomalies_df: pd.DataFrame) -> None:
    _apply_style()
    if anomalies_df.empty:
        return
    sampled = anomalies_df.head(300).copy()
    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=sampled,
        x="queue_length",
        y="utilization",
        size="anomaly_score",
        hue="system_code",
        alpha=0.75,
    )
    plt.title("Top Operational Parking Anomalies")
    plt.xlabel("Queue Length")
    plt.ylabel("Utilization")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_anomalies.png", dpi=220)
    plt.close()
