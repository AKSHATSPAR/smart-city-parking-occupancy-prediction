from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FORECAST_HORIZON_STEPS, NEIGHBOR_COUNT, UTILIZATION_BINS, UTILIZATION_LABELS


NUMERIC_FEATURES = [
    "capacity",
    "latitude",
    "longitude",
    "occupancy_clean",
    "available_spaces",
    "utilization",
    "queue_length",
    "is_special_day",
    "is_weekend",
    "hour",
    "minute",
    "day_of_week",
    "slot_index",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "traffic_level",
    "vehicle_priority",
    "session_progress",
    "capacity_pressure",
    "parking_stress_index",
    "neighbor_utilization_mean",
    "neighbor_queue_mean",
    "neighbor_stress_mean",
    "spatial_utilization_gap",
    "mean_neighbor_distance",
    "lag_occupancy_30m",
    "lag_occupancy_60m",
    "lag_utilization_30m",
    "lag_utilization_60m",
    "lag_utilization_120m",
    "lag_queue_30m",
    "lag_utilization_prev_day",
    "lag_queue_prev_day",
    "lag_stress_prev_day",
    "rolling_utilization_3",
    "rolling_utilization_6",
    "rolling_queue_3",
    "rolling_queue_6",
    "rolling_stress_3",
    "utilization_delta_30m",
    "utilization_delta_60m",
]

CATEGORICAL_FEATURES = ["system_code", "vehicle_type", "traffic_condition_nearby", "day_name"]


def build_spatial_neighbor_graph(clean_df: pd.DataFrame, neighbor_count: int = NEIGHBOR_COUNT) -> pd.DataFrame:
    systems = clean_df[["system_code", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    records: list[dict[str, float | str]] = []

    for _, source in systems.iterrows():
        candidates = systems[systems["system_code"] != source["system_code"]].copy()
        candidates["distance"] = np.sqrt(
            (candidates["latitude"] - source["latitude"]) ** 2 + (candidates["longitude"] - source["longitude"]) ** 2
        )
        candidates = candidates.nsmallest(neighbor_count, "distance").copy()
        inverse_distance = 1 / candidates["distance"].clip(lower=1e-6)
        candidates["weight"] = inverse_distance / inverse_distance.sum()

        for _, neighbor in candidates.iterrows():
            records.append(
                {
                    "system_code": source["system_code"],
                    "neighbor_code": neighbor["system_code"],
                    "distance": float(neighbor["distance"]),
                    "weight": float(neighbor["weight"]),
                }
            )

    return pd.DataFrame(records)


def _build_spatial_context(clean_df: pd.DataFrame, neighbor_graph: pd.DataFrame) -> pd.DataFrame:
    neighbor_snapshot = clean_df[
        ["system_code", "time_slot", "utilization", "queue_length", "parking_stress_index"]
    ].rename(
        columns={
            "system_code": "neighbor_code",
            "utilization": "neighbor_utilization",
            "queue_length": "neighbor_queue_length",
            "parking_stress_index": "neighbor_stress",
        }
    )
    joined = neighbor_graph.merge(neighbor_snapshot, on="neighbor_code", how="left")
    joined["weighted_neighbor_utilization"] = joined["weight"] * joined["neighbor_utilization"]
    joined["weighted_neighbor_queue"] = joined["weight"] * joined["neighbor_queue_length"]
    joined["weighted_neighbor_stress"] = joined["weight"] * joined["neighbor_stress"]
    context = (
        joined.groupby(["system_code", "time_slot"], as_index=False)
        .agg(
            neighbor_utilization_mean=("weighted_neighbor_utilization", "sum"),
            neighbor_queue_mean=("weighted_neighbor_queue", "sum"),
            neighbor_stress_mean=("weighted_neighbor_stress", "sum"),
            mean_neighbor_distance=("distance", "mean"),
        )
        .fillna(0)
    )
    return context


def build_model_dataset(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = clean_df.copy()
    neighbor_graph = build_spatial_neighbor_graph(enriched)
    spatial_context = _build_spatial_context(enriched, neighbor_graph)
    enriched = enriched.merge(spatial_context, on=["system_code", "time_slot"], how="left")

    daily_context = (
        enriched.sort_values(["system_code", "slot_index", "time_slot"])
        .groupby(["system_code", "slot_index"])[["utilization", "queue_length", "parking_stress_index"]]
        .shift(1)
        .rename(
            columns={
                "utilization": "lag_utilization_prev_day",
                "queue_length": "lag_queue_prev_day",
                "parking_stress_index": "lag_stress_prev_day",
            }
        )
    )
    enriched = pd.concat([enriched, daily_context], axis=1)
    enriched["spatial_utilization_gap"] = enriched["utilization"] - enriched["neighbor_utilization_mean"]

    frames: list[pd.DataFrame] = []

    for (_, _), group in enriched.groupby(["system_code", "session_id"], sort=True):
        current = group.sort_values("time_slot").copy()
        current["lag_occupancy_30m"] = current["occupancy_clean"].shift(1)
        current["lag_occupancy_60m"] = current["occupancy_clean"].shift(2)
        current["lag_utilization_30m"] = current["utilization"].shift(1)
        current["lag_utilization_60m"] = current["utilization"].shift(2)
        current["lag_utilization_120m"] = current["utilization"].shift(4)
        current["lag_queue_30m"] = current["queue_length"].shift(1)
        current["rolling_utilization_3"] = current["utilization"].shift(1).rolling(3, min_periods=1).mean()
        current["rolling_utilization_6"] = current["utilization"].shift(1).rolling(6, min_periods=1).mean()
        current["rolling_queue_3"] = current["queue_length"].shift(1).rolling(3, min_periods=1).mean()
        current["rolling_queue_6"] = current["queue_length"].shift(1).rolling(6, min_periods=1).mean()
        current["rolling_stress_3"] = current["parking_stress_index"].shift(1).rolling(3, min_periods=1).mean()
        current["utilization_delta_30m"] = current["utilization"] - current["lag_utilization_30m"]
        current["utilization_delta_60m"] = current["utilization"] - current["lag_utilization_60m"]

        current["target_time_slot"] = current["time_slot"].shift(-FORECAST_HORIZON_STEPS)
        current["target_occupancy_1h"] = current["occupancy_clean"].shift(-FORECAST_HORIZON_STEPS)
        current["target_available_spaces_1h"] = current["available_spaces"].shift(-FORECAST_HORIZON_STEPS)
        current["target_utilization_1h"] = current["utilization"].shift(-FORECAST_HORIZON_STEPS)
        current["target_stress_1h"] = current["parking_stress_index"].shift(-FORECAST_HORIZON_STEPS)
        current["target_time_slot_30m"] = current["time_slot"].shift(-1)
        current["target_utilization_30m"] = current["utilization"].shift(-1)
        current["target_available_spaces_30m"] = current["available_spaces"].shift(-1)
        current["target_time_slot_2h"] = current["time_slot"].shift(-4)
        current["target_utilization_2h"] = current["utilization"].shift(-4)
        current["target_available_spaces_2h"] = current["available_spaces"].shift(-4)
        frames.append(current)

    dataset = pd.concat(frames, ignore_index=True)
    dataset["future_demand_regime"] = pd.cut(
        dataset["target_stress_1h"],
        bins=[0.0, 0.35, 0.6, 0.8, 1.01],
        labels=["Stable", "Busy", "Congested", "Critical"],
        include_lowest=True,
        ordered=True,
    ).astype(str)
    dataset["current_utilization_band"] = pd.cut(
        dataset["utilization"],
        bins=UTILIZATION_BINS,
        labels=UTILIZATION_LABELS,
        include_lowest=True,
        ordered=True,
    ).astype(str)
    dataset["target_utilization_band_1h"] = pd.cut(
        dataset["target_utilization_1h"],
        bins=UTILIZATION_BINS,
        labels=UTILIZATION_LABELS,
        include_lowest=True,
        ordered=True,
    ).astype(str)

    required = [
        "lag_occupancy_30m",
        "lag_occupancy_60m",
        "lag_utilization_30m",
        "lag_utilization_60m",
        "lag_utilization_120m",
        "lag_queue_30m",
        "neighbor_utilization_mean",
        "neighbor_queue_mean",
        "neighbor_stress_mean",
        "lag_utilization_prev_day",
        "lag_queue_prev_day",
        "lag_stress_prev_day",
        "target_utilization_1h",
        "target_time_slot",
    ]
    dataset = dataset.dropna(subset=required).reset_index(drop=True)
    return dataset, neighbor_graph
