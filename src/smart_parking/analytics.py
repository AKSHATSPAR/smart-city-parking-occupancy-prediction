from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def build_location_profiles(clean_df: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        clean_df.groupby("system_code", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            capacity=("capacity", "max"),
            mean_occupancy=("occupancy_clean", "mean"),
            max_occupancy=("occupancy_clean", "max"),
            mean_utilization=("utilization", "mean"),
            std_utilization=("utilization", "std"),
            mean_queue_length=("queue_length", "mean"),
            high_traffic_ratio=("traffic_level", lambda x: (x == 2).mean()),
            special_day_ratio=("is_special_day", "mean"),
            average_stress=("parking_stress_index", "mean"),
        )
        .fillna(0)
    )
    return profiles


def cluster_location_profiles(profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    feature_cols = [
        "latitude",
        "longitude",
        "capacity",
        "mean_occupancy",
        "max_occupancy",
        "mean_utilization",
        "std_utilization",
        "mean_queue_length",
        "high_traffic_ratio",
        "special_day_ratio",
        "average_stress",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles[feature_cols])

    best_k = 3
    best_score = -1.0
    for candidate_k in range(2, min(6, len(profiles))):
        model = KMeans(n_clusters=candidate_k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = score
            best_k = candidate_k

    final_model = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    labels = final_model.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)

    result = profiles.copy()
    result["cluster"] = labels
    result["pca_1"] = coords[:, 0]
    result["pca_2"] = coords[:, 1]
    summary = {
        "optimal_clusters": float(best_k),
        "silhouette_score": float(best_score),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return result, summary


def detect_operational_anomalies(clean_df: pd.DataFrame) -> pd.DataFrame:
    anomaly_features = clean_df[
        [
            "utilization",
            "queue_length",
            "traffic_level",
            "capacity_pressure",
            "parking_stress_index",
            "neighbor_utilization_mean",
            "spatial_utilization_gap",
        ]
    ].fillna(0)
    model = IsolationForest(contamination=0.03, random_state=RANDOM_STATE)
    scores = model.fit_predict(anomaly_features)
    output = clean_df.copy()
    output["anomaly_flag"] = (scores == -1).astype(int)
    output["anomaly_score"] = -model.score_samples(anomaly_features)
    output["anomaly_reason"] = np.select(
        [
            (output["queue_length"] >= 8) & (output["utilization"] >= 0.85),
            output["spatial_utilization_gap"] >= 0.2,
            output["capacity_pressure"] >= 0.02,
        ],
        [
            "High queue and near-full utilization",
            "Higher congestion than neighboring parking systems",
            "Queue pressure is unusually high for this capacity",
        ],
        default="Unusual demand pattern detected",
    )
    return output[output["anomaly_flag"] == 1].sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def build_recommendations(latest_forecast_df: pd.DataFrame) -> pd.DataFrame:
    latest = latest_forecast_df.copy()
    latest["prediction_interval_lower"] = latest["prediction_interval_lower"].fillna(latest["predicted_utilization_1h"])
    latest["prediction_interval_upper"] = latest["prediction_interval_upper"].fillna(latest["predicted_utilization_1h"])
    latest["uncertainty_width"] = latest["prediction_interval_upper"] - latest["prediction_interval_lower"]
    latest["predicted_free_ratio"] = latest["predicted_available_spaces_1h"] / latest["capacity"].clip(lower=1)

    latest["balanced_score"] = (
        0.45 * latest["predicted_free_ratio"]
        + 0.25 * (1 - latest["predicted_utilization_1h"])
        + 0.20 * (1 - latest["capacity_pressure"].clip(0, 1))
        + 0.10 * (1 - latest["uncertainty_width"].clip(0, 1))
    )
    latest["low_risk_score"] = 1 - latest["prediction_interval_upper"]
    latest["low_congestion_score"] = (
        0.6 * (1 - latest["capacity_pressure"].clip(0, 1)) + 0.4 * (1 - latest["predicted_utilization_1h"])
    )
    latest["max_availability_score"] = latest["predicted_free_ratio"]
    latest["risk_band"] = pd.cut(
        latest["prediction_interval_upper"],
        bins=[0.0, 0.55, 0.75, 0.9, 1.01],
        labels=["Low", "Moderate", "High", "Critical"],
        include_lowest=True,
        ordered=True,
    ).astype(str)
    latest["recommendation_rank"] = latest["balanced_score"].rank(ascending=False, method="dense").astype(int)
    latest = latest.sort_values(["recommendation_rank", "predicted_available_spaces_1h"], ascending=[True, False])
    return latest.reset_index(drop=True)
