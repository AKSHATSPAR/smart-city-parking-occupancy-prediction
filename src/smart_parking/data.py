from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, TIME_SLOT_FREQUENCY


RENAME_MAP = {
    "ID": "record_id",
    "SystemCodeNumber": "system_code",
    "Capacity": "capacity",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Occupancy": "occupancy",
    "VehicleType": "vehicle_type",
    "TrafficConditionNearby": "traffic_condition_nearby",
    "QueueLength": "queue_length",
    "IsSpecialDay": "is_special_day",
    "Timestamp": "timestamp",
}

TRAFFIC_MAP = {"low": 0, "average": 1, "high": 2}
VEHICLE_MAP = {"cycle": 0, "bike": 1, "car": 2, "truck": 3}


def _mode_or_first(values: pd.Series) -> Any:
    modes = values.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return values.iloc[0]


def load_raw_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"]).rename(columns=RENAME_MAP)
    df = df.sort_values(["system_code", "timestamp"]).reset_index(drop=True)
    return df


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["time_slot"] = cleaned["timestamp"].dt.round(TIME_SLOT_FREQUENCY)

    aggregated = (
        cleaned.groupby(["system_code", "time_slot"], as_index=False)
        .agg(
            record_id=("record_id", "min"),
            timestamp=("timestamp", "min"),
            capacity=("capacity", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            occupancy=("occupancy", "mean"),
            vehicle_type=("vehicle_type", _mode_or_first),
            traffic_condition_nearby=("traffic_condition_nearby", _mode_or_first),
            queue_length=("queue_length", "mean"),
            is_special_day=("is_special_day", "max"),
        )
        .sort_values(["system_code", "time_slot"])
        .reset_index(drop=True)
    )

    aggregated["occupancy_raw"] = aggregated["occupancy"]
    aggregated["occupancy_clean"] = aggregated[["occupancy", "capacity"]].min(axis=1).clip(lower=0)
    aggregated["occupancy_anomaly_flag"] = (aggregated["occupancy_raw"] > aggregated["capacity"]).astype(int)
    aggregated["queue_length"] = aggregated["queue_length"].clip(lower=0)
    aggregated["available_spaces"] = aggregated["capacity"] - aggregated["occupancy_clean"]
    aggregated["utilization"] = (aggregated["occupancy_clean"] / aggregated["capacity"]).clip(0, 1)

    aggregated["date"] = aggregated["time_slot"].dt.date.astype(str)
    aggregated["month"] = aggregated["time_slot"].dt.month
    aggregated["day"] = aggregated["time_slot"].dt.day
    aggregated["day_of_week"] = aggregated["time_slot"].dt.dayofweek
    aggregated["day_name"] = aggregated["time_slot"].dt.day_name()
    aggregated["is_weekend"] = aggregated["day_of_week"].isin([5, 6]).astype(int)
    aggregated["hour"] = aggregated["time_slot"].dt.hour
    aggregated["minute"] = aggregated["time_slot"].dt.minute
    aggregated["slot_index"] = aggregated["hour"] * 2 + (aggregated["minute"] // 30)
    aggregated["hour_sin"] = np.sin(2 * np.pi * aggregated["hour"] / 24)
    aggregated["hour_cos"] = np.cos(2 * np.pi * aggregated["hour"] / 24)
    aggregated["dow_sin"] = np.sin(2 * np.pi * aggregated["day_of_week"] / 7)
    aggregated["dow_cos"] = np.cos(2 * np.pi * aggregated["day_of_week"] / 7)

    aggregated["traffic_level"] = (
        aggregated["traffic_condition_nearby"].str.lower().map(TRAFFIC_MAP).fillna(1).astype(int)
    )
    aggregated["vehicle_priority"] = (
        aggregated["vehicle_type"].str.lower().map(VEHICLE_MAP).fillna(1).astype(int)
    )

    time_diff = aggregated.groupby("system_code")["time_slot"].diff()
    aggregated["is_new_session"] = time_diff.isna() | (time_diff != pd.Timedelta(TIME_SLOT_FREQUENCY))
    aggregated["session_id"] = aggregated.groupby("system_code")["is_new_session"].cumsum().astype(int)
    aggregated["minutes_from_previous"] = time_diff.dt.total_seconds().div(60).fillna(0)
    aggregated["session_step"] = aggregated.groupby(["system_code", "session_id"]).cumcount()
    aggregated["session_size"] = aggregated.groupby(["system_code", "session_id"])["record_id"].transform("count")
    aggregated["session_progress"] = aggregated["session_step"] / aggregated["session_size"].clip(lower=1)
    aggregated["capacity_pressure"] = aggregated["queue_length"] / aggregated["capacity"].clip(lower=1)
    aggregated["parking_stress_index"] = (
        0.7 * aggregated["utilization"] + 0.2 * aggregated["traffic_level"] / 2 + 0.1 * aggregated["capacity_pressure"]
    ).clip(0, 1)
    return aggregated


def dataset_summary(clean_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(clean_df.shape[0]),
        "columns": int(clean_df.shape[1]),
        "systems": int(clean_df["system_code"].nunique()),
        "date_start": clean_df["time_slot"].min().isoformat(),
        "date_end": clean_df["time_slot"].max().isoformat(),
        "anomaly_rows": int(clean_df["occupancy_anomaly_flag"].sum()),
        "avg_utilization": float(clean_df["utilization"].mean()),
        "avg_queue_length": float(clean_df["queue_length"].mean()),
    }
