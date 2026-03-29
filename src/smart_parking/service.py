from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from .config import (
    ANOMALIES_PATH,
    BACKTEST_METRICS_PATH,
    CLEAN_DATA_PATH,
    DRIFT_REPORT_PATH,
    FEATURE_DATA_PATH,
    LATEST_FORECAST_PATH,
    LIVE_OPS_DB_PATH,
    MODEL_METRICS_PATH,
    MODEL_REGISTRY_PATH,
    MULTI_HORIZON_METRICS_PATH,
    RECOMMENDATIONS_PATH,
    RF_MODEL_PATH,
    SQLITE_DB_PATH,
    SPATIAL_NEIGHBOR_PATH,
    XGB_MODEL_PATH,
)
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .live_ops import LiveOpsStore
from .simulator import DemoScenarioEngine


class WhatIfRequest(BaseModel):
    system_code: str
    queue_length: int | None = Field(default=None, ge=0, le=25)
    traffic_condition_nearby: str | None = Field(default=None)
    is_special_day: int | None = Field(default=None, ge=0, le=1)
    vehicle_type: str | None = Field(default=None)


class ObjectiveRequest(BaseModel):
    objective: str = "Balanced"
    limit: int = 10


class IngestObservationRequest(BaseModel):
    system_code: str
    occupancy: float = Field(ge=0)
    queue_length: int = Field(default=0, ge=0, le=30)
    traffic_condition_nearby: str = "low"
    is_special_day: int = Field(default=0, ge=0, le=1)
    timestamp: str | None = None
    vehicle_type: str | None = None
    capacity: float | None = Field(default=None, gt=0)
    latitude: float | None = None
    longitude: float | None = None


class DemoScenarioRequest(BaseModel):
    scenario_name: str
    steps: int = Field(default=3, ge=1, le=12)
    reset_first: bool = True


class DemoPlaybookRequest(BaseModel):
    playbook_name: str
    reset_first: bool = True


class DemoResetRequest(BaseModel):
    clear_jobs: bool = False


class SmartParkingService:
    def __init__(self) -> None:
        self.clean_df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["timestamp", "time_slot"])
        self.feature_df = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"])
        self.metrics_df = pd.read_csv(MODEL_METRICS_PATH)
        self.multi_horizon_df = pd.read_csv(MULTI_HORIZON_METRICS_PATH)
        self.latest_df = pd.read_csv(LATEST_FORECAST_PATH, parse_dates=["time_slot", "target_time_slot"])
        self.recommendations_df = pd.read_csv(RECOMMENDATIONS_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"])
        self.anomalies_df = pd.read_csv(ANOMALIES_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"])
        self.backtest_df = pd.read_csv(BACKTEST_METRICS_PATH, parse_dates=["train_end_time", "test_start_time", "test_end_time"])
        self.drift_report = json.loads(DRIFT_REPORT_PATH.read_text())
        self.registry = json.loads(MODEL_REGISTRY_PATH.read_text())
        self.neighbor_graph = pd.read_csv(SPATIAL_NEIGHBOR_PATH)
        self.xgb_model = joblib.load(XGB_MODEL_PATH)
        self.rf_model = joblib.load(RF_MODEL_PATH)
        self.store = LiveOpsStore()
        self.demo_engine = DemoScenarioEngine(self.recommendations_df, self.neighbor_graph)
        self.store.seed_live_forecasts(self.recommendations_df)

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict]:
        normalized = frame.copy()
        for column in normalized.columns:
            if pd.api.types.is_datetime64_any_dtype(normalized[column]):
                normalized[column] = normalized[column].astype(str)
        normalized = normalized.astype(object).where(pd.notnull(normalized), None)
        return normalized.to_dict(orient="records")

    def health(self) -> dict:
        live_obs = self.store.live_observations()
        return {
            "status": "ok",
            "systems": int(self.clean_df["system_code"].nunique()),
            "observations": int(len(self.clean_df)),
            "live_observations": int(len(live_obs)),
            "champion_model": self.registry["champion_model"],
        }

    def ready(self) -> dict:
        checks = {
            "xgboost_model_loaded": self.xgb_model is not None,
            "random_forest_model_loaded": self.rf_model is not None,
            "analytics_db_present": SQLITE_DB_PATH.exists(),
            "live_ops_db_present": LIVE_OPS_DB_PATH.exists(),
            "champion_model_available": bool(self.registry.get("champion_model")),
        }
        status = "ready" if all(checks.values()) else "degraded"
        return {
            "status": status,
            "checks": checks,
            "champion_model": self.registry.get("champion_model"),
        }

    def systems(self) -> list[str]:
        return sorted(self.clean_df["system_code"].unique().tolist())

    def _current_live_frame(self) -> pd.DataFrame:
        live_df = self.store.live_forecasts()
        if live_df.empty:
            live_df = self.recommendations_df.copy()
        live_df = live_df.copy()
        if "recommendation_rank" not in live_df.columns and "balanced_score" in live_df.columns:
            live_df = live_df.sort_values("balanced_score", ascending=False).reset_index(drop=True)
            live_df["recommendation_rank"] = live_df.index + 1
        return live_df

    def _drift_features_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.drift_report.get("features", []))

    @staticmethod
    def _drift_severity(max_psi: float) -> str:
        if max_psi >= 0.25:
            return "critical"
        if max_psi >= 0.10:
            return "warning"
        return "stable"

    def _latest_retrain_job(self) -> dict[str, Any] | None:
        jobs = self.store.retrain_jobs()
        return jobs[0] if jobs else None

    def latest_forecasts(self, limit: int = 20) -> list[dict]:
        live_df = self._current_live_frame()
        frame = (
            live_df.sort_values("predicted_available_spaces_1h", ascending=False)
            .head(limit)[
                [
                    "system_code",
                    "time_slot",
                    "predicted_available_spaces_1h",
                    "predicted_utilization_1h",
                    "prediction_interval_lower",
                    "prediction_interval_upper",
                    "uncertainty_width",
                    "risk_band",
                ]
            ]
            .copy()
        )
        return self._records(frame)

    def recommendations(self, objective: str = "Balanced", limit: int = 10) -> list[dict]:
        live_df = self._current_live_frame()
        score_map = {
            "Balanced": "balanced_score",
            "Maximum Availability": "max_availability_score",
            "Lowest Risk": "low_risk_score",
            "Lowest Congestion": "low_congestion_score",
        }
        score_col = score_map.get(objective, "balanced_score")
        live_df = live_df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
        live_df["recommendation_rank"] = live_df.index + 1
        frame = live_df.head(limit)[
            [
                "system_code",
                "time_slot",
                "latitude",
                "longitude",
                "predicted_available_spaces_1h",
                "predicted_utilization_1h",
                "prediction_interval_lower",
                "prediction_interval_upper",
                "risk_band",
                "balanced_score",
                "max_availability_score",
                "low_risk_score",
                "low_congestion_score",
                "recommendation_rank",
            ]
        ].copy()
        return self._records(frame)

    def anomalies(self, limit: int = 20) -> list[dict]:
        frame = self.anomalies_df.head(limit)[
            [
                "system_code",
                "time_slot",
                "utilization",
                "queue_length",
                "parking_stress_index",
                "neighbor_utilization_mean",
                "anomaly_score",
                "anomaly_reason",
            ]
        ].copy()
        return self._records(frame)

    def monitoring(self) -> dict:
        return {
            "drift_report": self.drift_report,
            "rolling_backtest": self._records(self.backtest_df),
        }

    def metrics(self) -> dict:
        return {
            "model_metrics": self._records(self.metrics_df),
            "multi_horizon_metrics": self._records(self.multi_horizon_df),
        }

    def registry_payload(self) -> dict:
        return self.registry

    def live_state(self) -> dict:
        return {
            "forecasts": self.latest_forecasts(limit=50),
            "retrain_jobs": self.store.retrain_jobs()[:20],
        }

    def recent_activity(self, limit: int = 20) -> list[dict[str, Any]]:
        audit_records = [
            {
                "source": "audit",
                "created_at": item["created_at"],
                "title": item["event_type"],
                "summary": json.dumps(item["payload"]),
            }
            for item in self.store.audit_events(limit=limit)
        ]
        request_records = [
            {
                "source": "api",
                "created_at": item["created_at"],
                "title": f"{item['method']} {item['path']}",
                "summary": f"status={item['status_code']} duration_ms={item['duration_ms']:.2f}",
            }
            for item in self.store.api_request_logs(limit=limit)
        ]
        combined = audit_records + request_records
        combined.sort(key=lambda item: item["created_at"], reverse=True)
        return combined[:limit]

    def _build_ops_alerts(
        self,
        live_df: pd.DataFrame,
        drift_df: pd.DataFrame,
        latest_job: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []

        if not live_df.empty:
            critical_df = live_df[live_df["risk_band"] == "Critical"].sort_values(
                "predicted_utilization_1h",
                ascending=False,
            )
            if not critical_df.empty:
                top_critical = critical_df.iloc[0]
                alerts.append(
                    {
                        "severity": "critical" if len(critical_df) >= 3 else "warning",
                        "category": "capacity",
                        "title": "Critical parking pressure detected",
                        "message": f"{len(critical_df)} systems are critical. Highest risk is {top_critical['system_code']} with predicted utilization {float(top_critical['predicted_utilization_1h']):.2%}.",
                        "recommended_action": "Reroute drivers to the top-ranked live recommendations and monitor nearby spillover systems.",
                    }
                )

            constrained_df = live_df[
                (live_df["predicted_available_spaces_1h"] <= 50)
                | (live_df.get("predicted_free_ratio", 1.0) <= 0.05)
            ].sort_values("predicted_available_spaces_1h")
            if not constrained_df.empty:
                tight_systems = ", ".join(constrained_df["system_code"].head(3).tolist())
                alerts.append(
                    {
                        "severity": "warning",
                        "category": "availability",
                        "title": "Low-availability systems emerging",
                        "message": f"{len(constrained_df)} systems are forecast to have 50 or fewer free spaces. Most constrained: {tight_systems}.",
                        "recommended_action": "Prioritize alternative parking guidance before queues spill into adjacent systems.",
                    }
                )

        if not drift_df.empty:
            top_drift = drift_df.sort_values("psi", ascending=False).iloc[0]
            max_psi = float(top_drift["psi"])
            drift_severity = self._drift_severity(max_psi)
            alerts.append(
                {
                    "severity": "info" if drift_severity == "stable" else drift_severity,
                    "category": "drift",
                    "title": "Data drift monitor",
                    "message": f"Highest PSI is {max_psi:.3f} on {top_drift['feature']}; current drift status is {drift_severity}.",
                    "recommended_action": "Continue monitoring feature movement and retrain if PSI enters the warning band.",
                }
            )

        if latest_job is None:
            alerts.append(
                {
                    "severity": "info",
                    "category": "retraining",
                    "title": "No retraining jobs on record",
                    "message": "The live ops database does not yet contain a retraining history.",
                    "recommended_action": "Use the retraining endpoint when drift or live performance starts to degrade.",
                }
            )
        else:
            status = latest_job["status"]
            severity = "critical" if status == "failed" else "info"
            timestamp = latest_job.get("finished_at") or latest_job.get("started_at") or latest_job.get("created_at")
            alerts.append(
                {
                    "severity": severity,
                    "category": "retraining",
                    "title": f"Latest retraining job is {status}",
                    "message": f"Most recent retraining job status: {status} at {timestamp}.",
                    "recommended_action": "Investigate failures immediately, or keep the current champion model in service if the last run completed cleanly.",
                }
            )

        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda item: (severity_order.get(item["severity"], 3), item["category"]))
        return alerts

    def ops_alerts(self) -> list[dict[str, Any]]:
        return self._build_ops_alerts(
            live_df=self._current_live_frame(),
            drift_df=self._drift_features_frame(),
            latest_job=self._latest_retrain_job(),
        )

    def _build_talking_points(
        self,
        live_df: pd.DataFrame,
        drift_df: pd.DataFrame,
        latest_job: dict[str, Any] | None,
        champion_metric: dict[str, Any] | None,
        mean_backtest_band_accuracy: float,
    ) -> list[str]:
        points: list[str] = []
        critical_count = int((live_df["risk_band"] == "Critical").sum()) if not live_df.empty else 0
        best_option = live_df.sort_values("balanced_score", ascending=False).iloc[0]["system_code"] if not live_df.empty else "N/A"
        points.append(
            f"{critical_count} systems are currently critical, and {best_option} is the strongest live recommendation for rerouting traffic."
        )

        if champion_metric is not None:
            points.append(
                f"{self.registry['champion_model']} remains the champion model with test RMSE {float(champion_metric['rmse']):.4f} and rolling band accuracy {mean_backtest_band_accuracy:.3f}."
            )

        if not drift_df.empty:
            top_drift = drift_df.sort_values("psi", ascending=False).iloc[0]
            drift_status = self._drift_severity(float(top_drift["psi"]))
            points.append(
                f"Drift is {drift_status}; the highest PSI is {float(top_drift['psi']):.3f} on {top_drift['feature']}."
            )

        live_observation_count = int(len(self.store.live_observations()))
        points.append(
            f"The live board has processed {live_observation_count} live observations, so the current state reflects interactive runtime behavior rather than static CSV output."
        )

        if latest_job is not None:
            last_status = latest_job["status"]
            last_timestamp = latest_job.get("finished_at") or latest_job.get("created_at")
            points.append(f"The latest retraining job is {last_status} and was last updated at {last_timestamp}.")

        return points

    def ops_summary(self) -> dict:
        live_df = self._current_live_frame()
        drift_df = self._drift_features_frame()
        latest_job = self._latest_retrain_job()
        alerts = self._build_ops_alerts(live_df=live_df, drift_df=drift_df, latest_job=latest_job)

        champion_metric = next(
            (
                record
                for record in self.metrics_df.to_dict(orient="records")
                if record.get("model") == self.registry["champion_model"]
            ),
            None,
        )
        rolling_xgb = self.backtest_df[
            self.backtest_df["model"].astype(str).str.contains("XGBoost", case=False, na=False)
        ].copy()
        mean_backtest_band_accuracy = float(rolling_xgb["band_accuracy"].mean()) if not rolling_xgb.empty else 0.0
        max_psi = float(drift_df["psi"].max()) if not drift_df.empty else 0.0
        drift_severity = self._drift_severity(max_psi)

        low_availability_df = live_df[
            (live_df["predicted_available_spaces_1h"] <= 50)
            | (live_df.get("predicted_free_ratio", 1.0) <= 0.05)
        ].copy()
        critical_watchlist = self._records(
            live_df.sort_values("predicted_utilization_1h", ascending=False).head(5)[
                [
                    "system_code",
                    "predicted_available_spaces_1h",
                    "predicted_utilization_1h",
                    "prediction_interval_upper",
                    "risk_band",
                    "recommendation_rank",
                ]
            ]
        )
        recommended_alternatives = self.recommendations(limit=5)

        alert_counts = {
            level: sum(1 for alert in alerts if alert["severity"] == level)
            for level in ["critical", "warning", "info"]
        }
        talking_points = self._build_talking_points(
            live_df=live_df,
            drift_df=drift_df,
            latest_job=latest_job,
            champion_metric=champion_metric,
            mean_backtest_band_accuracy=mean_backtest_band_accuracy,
        )

        live_snapshot = {
            "systems": int(live_df["system_code"].nunique()) if not live_df.empty else 0,
            "live_observations": int(len(self.store.live_observations())),
            "mean_utilization": float(live_df["predicted_utilization_1h"].mean()) if not live_df.empty else 0.0,
            "critical_systems": int((live_df["risk_band"] == "Critical").sum()) if not live_df.empty else 0,
            "high_risk_systems": int(live_df["risk_band"].isin(["Critical", "High"]).sum()) if not live_df.empty else 0,
            "low_availability_systems": int(len(low_availability_df)),
            "best_option": recommended_alternatives[0] if recommended_alternatives else None,
            "most_congested": critical_watchlist[0] if critical_watchlist else None,
        }

        drift_overview = {
            "severity": drift_severity,
            "max_psi": max_psi,
            "features_monitored": int(len(drift_df)),
            "top_features": self._records(drift_df.sort_values("psi", ascending=False).head(5)) if not drift_df.empty else [],
        }

        model_overview = {
            "champion_model": self.registry["champion_model"],
            "feature_count": int(self.registry["feature_count"]),
            "test_rmse": float(champion_metric["rmse"]) if champion_metric is not None else None,
            "test_band_accuracy": float(champion_metric["band_accuracy"]) if champion_metric is not None else None,
            "rolling_band_accuracy": mean_backtest_band_accuracy,
        }

        return {
            "as_of_utc": str(pd.Timestamp.utcnow()),
            "live_snapshot": live_snapshot,
            "drift_overview": drift_overview,
            "model_overview": model_overview,
            "retrain_status": latest_job or {"status": "not_run"},
            "alerts": alerts,
            "alert_counts": alert_counts,
            "critical_watchlist": critical_watchlist,
            "recommended_alternatives": recommended_alternatives,
            "retrain_jobs": self.store.retrain_jobs()[:10],
            "recent_activity": self.recent_activity(limit=15),
            "talking_points": talking_points,
        }

    def demo_scenarios(self) -> list[dict]:
        return self.demo_engine.list_scenarios()

    def demo_playbooks(self) -> list[dict]:
        return self.demo_engine.list_playbooks()

    def reset_live_state(self, clear_jobs: bool = False) -> dict:
        self.store.reset_live_state(self.recommendations_df, clear_jobs=clear_jobs)
        payload = {
            "status": "reset",
            "systems_seeded": int(self.recommendations_df["system_code"].nunique()),
            "live_observations": 0,
            "clear_jobs": clear_jobs,
        }
        self.store.append_audit_event("demo_reset", payload, actor="service")
        return payload

    def _max_interval_width(self) -> float:
        series = self.latest_df["uncertainty_width"].fillna(0)
        if series.empty:
            return 0.08
        return float(series.max())

    def _combined_history(self, system_code: str) -> pd.DataFrame:
        base = self.feature_df[self.feature_df["system_code"] == system_code].copy()
        live = self.store.live_observations(system_code)
        if not live.empty:
            for column in ["timestamp", "time_slot", "target_time_slot"]:
                if column in live.columns:
                    live[column] = pd.to_datetime(live[column], errors="coerce")
            common_columns = [column for column in base.columns if column in live.columns]
            live = live[common_columns].copy().dropna(axis=1, how="all")
            if live.empty:
                history = base
            else:
                aligned_columns = [column for column in base.columns if column in live.columns]
                history = pd.concat([base[aligned_columns], live[aligned_columns]], ignore_index=True, sort=False)
                history = history.reindex(columns=base.columns)
        else:
            history = base
        return history.sort_values("time_slot").reset_index(drop=True)

    def _neighbor_context(self, system_code: str, current_time: pd.Timestamp) -> dict[str, float]:
        rows = self.neighbor_graph[self.neighbor_graph["system_code"] == system_code]
        if rows.empty:
            return {
                "neighbor_utilization_mean": 0.0,
                "neighbor_queue_mean": 0.0,
                "neighbor_stress_mean": 0.0,
                "mean_neighbor_distance": 0.0,
            }

        util = 0.0
        queue = 0.0
        stress = 0.0
        distance = 0.0
        for _, item in rows.iterrows():
            neighbor_history = self._combined_history(item["neighbor_code"])
            neighbor_history = neighbor_history[neighbor_history["time_slot"] <= current_time]
            if neighbor_history.empty:
                continue
            latest = neighbor_history.sort_values("time_slot").tail(1).iloc[0]
            util += float(item["weight"]) * float(latest.get("utilization", 0.0))
            queue += float(item["weight"]) * float(latest.get("queue_length", 0.0))
            stress += float(item["weight"]) * float(latest.get("parking_stress_index", 0.0))
            distance += float(item["distance"])

        return {
            "neighbor_utilization_mean": util,
            "neighbor_queue_mean": queue,
            "neighbor_stress_mean": stress,
            "mean_neighbor_distance": distance / max(len(rows), 1),
        }

    def _build_live_feature_row(self, request: IngestObservationRequest) -> pd.DataFrame:
        history = self._combined_history(request.system_code)
        if history.empty:
            raise ValueError(f"Unknown system_code: {request.system_code}")

        template = history.tail(1).copy()
        current_time = (
            pd.to_datetime(request.timestamp, errors="coerce")
            if request.timestamp is not None
            else template["time_slot"].iloc[0] + pd.Timedelta(minutes=30)
        )
        if pd.isna(current_time):
            raise ValueError("Invalid timestamp provided.")
        current_time = pd.Timestamp(current_time).round("30min")

        row = template.copy()
        row["timestamp"] = current_time
        row["time_slot"] = current_time
        row["system_code"] = request.system_code
        row["capacity"] = request.capacity if request.capacity is not None else float(template["capacity"].iloc[0])
        row["latitude"] = request.latitude if request.latitude is not None else float(template["latitude"].iloc[0])
        row["longitude"] = request.longitude if request.longitude is not None else float(template["longitude"].iloc[0])
        row["occupancy_clean"] = min(float(request.occupancy), float(row["capacity"].iloc[0]))
        row["occupancy_raw"] = float(request.occupancy)
        row["occupancy"] = float(request.occupancy)
        row["available_spaces"] = float(row["capacity"].iloc[0]) - float(row["occupancy_clean"].iloc[0])
        row["utilization"] = float(row["occupancy_clean"].iloc[0]) / max(float(row["capacity"].iloc[0]), 1.0)
        row["queue_length"] = request.queue_length
        row["traffic_condition_nearby"] = request.traffic_condition_nearby
        row["traffic_level"] = {"low": 0, "average": 1, "high": 2}[request.traffic_condition_nearby]
        row["is_special_day"] = request.is_special_day
        row["vehicle_type"] = request.vehicle_type if request.vehicle_type is not None else str(template["vehicle_type"].iloc[0])
        row["vehicle_priority"] = {"cycle": 0, "bike": 1, "car": 2, "truck": 3}.get(str(row["vehicle_type"].iloc[0]), 1)
        row["capacity_pressure"] = float(row["queue_length"].iloc[0]) / max(float(row["capacity"].iloc[0]), 1.0)
        row["parking_stress_index"] = (
            0.7 * float(row["utilization"].iloc[0])
            + 0.2 * float(row["traffic_level"].iloc[0]) / 2
            + 0.1 * float(row["capacity_pressure"].iloc[0])
        )
        row["date"] = str(current_time.date())
        row["month"] = current_time.month
        row["day"] = current_time.day
        row["day_of_week"] = current_time.dayofweek
        row["day_name"] = current_time.day_name()
        row["is_weekend"] = int(current_time.dayofweek in [5, 6])
        row["hour"] = current_time.hour
        row["minute"] = current_time.minute
        row["slot_index"] = current_time.hour * 2 + current_time.minute // 30
        row["hour_sin"] = math.sin(2 * math.pi * current_time.hour / 24)
        row["hour_cos"] = math.cos(2 * math.pi * current_time.hour / 24)
        row["dow_sin"] = math.sin(2 * math.pi * current_time.dayofweek / 7)
        row["dow_cos"] = math.cos(2 * math.pi * current_time.dayofweek / 7)

        previous = history[history["time_slot"] < current_time].sort_values("time_slot")
        previous_tail = previous.tail(6)
        previous_1 = previous.tail(1)
        previous_2 = previous.tail(2)
        previous_4 = previous.tail(4)
        row["lag_occupancy_30m"] = float(previous_1["occupancy_clean"].iloc[-1]) if len(previous_1) else float(template["lag_occupancy_30m"].iloc[0])
        row["lag_occupancy_60m"] = float(previous_2["occupancy_clean"].iloc[0]) if len(previous_2) >= 2 else float(template["lag_occupancy_60m"].iloc[0])
        row["lag_utilization_30m"] = float(previous_1["utilization"].iloc[-1]) if len(previous_1) else float(template["lag_utilization_30m"].iloc[0])
        row["lag_utilization_60m"] = float(previous_2["utilization"].iloc[0]) if len(previous_2) >= 2 else float(template["lag_utilization_60m"].iloc[0])
        row["lag_utilization_120m"] = float(previous_4["utilization"].iloc[0]) if len(previous_4) >= 4 else float(template["lag_utilization_120m"].iloc[0])
        row["lag_queue_30m"] = float(previous_1["queue_length"].iloc[-1]) if len(previous_1) else float(template["lag_queue_30m"].iloc[0])
        same_slot = previous[previous["slot_index"] == row["slot_index"].iloc[0]].tail(1)
        row["lag_utilization_prev_day"] = (
            float(same_slot["utilization"].iloc[-1]) if len(same_slot) else float(template["lag_utilization_prev_day"].iloc[0])
        )
        row["lag_queue_prev_day"] = (
            float(same_slot["queue_length"].iloc[-1]) if len(same_slot) else float(template["lag_queue_prev_day"].iloc[0])
        )
        row["lag_stress_prev_day"] = (
            float(same_slot["parking_stress_index"].iloc[-1])
            if len(same_slot)
            else float(template["lag_stress_prev_day"].iloc[0])
        )
        row["rolling_utilization_3"] = (
            float(previous_tail.tail(3)["utilization"].mean()) if len(previous_tail) else float(template["rolling_utilization_3"].iloc[0])
        )
        row["rolling_utilization_6"] = (
            float(previous_tail["utilization"].mean()) if len(previous_tail) else float(template["rolling_utilization_6"].iloc[0])
        )
        row["rolling_queue_3"] = (
            float(previous_tail.tail(3)["queue_length"].mean()) if len(previous_tail) else float(template["rolling_queue_3"].iloc[0])
        )
        row["rolling_queue_6"] = (
            float(previous_tail["queue_length"].mean()) if len(previous_tail) else float(template["rolling_queue_6"].iloc[0])
        )
        row["rolling_stress_3"] = (
            float(previous_tail.tail(3)["parking_stress_index"].mean())
            if len(previous_tail)
            else float(template["rolling_stress_3"].iloc[0])
        )
        row["utilization_delta_30m"] = float(row["utilization"].iloc[0]) - float(row["lag_utilization_30m"].iloc[0])
        row["utilization_delta_60m"] = float(row["utilization"].iloc[0]) - float(row["lag_utilization_60m"].iloc[0])

        last_time = previous["time_slot"].iloc[-1] if len(previous) else None
        is_contiguous = bool(last_time is not None and (current_time - last_time) == pd.Timedelta(minutes=30))
        row["is_new_session"] = not is_contiguous
        row["session_id"] = int(template["session_id"].iloc[0]) if is_contiguous else int(template["session_id"].iloc[0]) + 1
        row["minutes_from_previous"] = float((current_time - last_time).total_seconds() / 60) if last_time is not None else 0.0
        row["session_step"] = int(template["session_step"].iloc[0]) + 1 if is_contiguous else 0
        row["session_size"] = max(int(row["session_step"].iloc[0]) + 1, 1)
        row["session_progress"] = float(row["session_step"].iloc[0]) / float(row["session_size"].iloc[0])

        neighbor_context = self._neighbor_context(request.system_code, current_time)
        for key, value in neighbor_context.items():
            row[key] = value
        row["spatial_utilization_gap"] = float(row["utilization"].iloc[0]) - float(row["neighbor_utilization_mean"].iloc[0])
        row["target_time_slot"] = current_time + pd.Timedelta(hours=1)
        return row

    def _score_live_forecast(self, row: pd.DataFrame) -> dict[str, Any]:
        predicted_utilization = float(row["predicted_utilization_1h"].iloc[0])
        predicted_available = float(row["predicted_available_spaces_1h"].iloc[0])
        capacity = max(float(row["capacity"].iloc[0]), 1.0)
        interval_lower = float(row["prediction_interval_lower"].iloc[0])
        interval_upper = float(row["prediction_interval_upper"].iloc[0])
        uncertainty_width = float(row["uncertainty_width"].iloc[0])
        predicted_free_ratio = predicted_available / capacity
        capacity_pressure = float(row["capacity_pressure"].iloc[0])
        scores = {
            "predicted_free_ratio": predicted_free_ratio,
            "balanced_score": 0.45 * predicted_free_ratio + 0.25 * (1 - predicted_utilization) + 0.20 * (1 - capacity_pressure) + 0.10 * (1 - uncertainty_width),
            "low_risk_score": 1 - interval_upper,
            "low_congestion_score": 0.6 * (1 - capacity_pressure) + 0.4 * (1 - predicted_utilization),
            "max_availability_score": predicted_free_ratio,
            "risk_band": "Low" if interval_upper <= 0.55 else "Moderate" if interval_upper <= 0.75 else "High" if interval_upper <= 0.9 else "Critical",
        }
        return scores

    def ingest_observation(self, request: IngestObservationRequest) -> dict:
        row = self._build_live_feature_row(request)
        features = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        xgb_prediction = float(self.xgb_model.predict(features)[0])
        rf_prediction = float(self.rf_model.predict(features)[0])
        rf_weight = self.registry["ensemble_weights"]["random_forest"]
        xgb_weight = self.registry["ensemble_weights"]["xgboost"]
        ensemble_prediction = rf_weight * rf_prediction + xgb_weight * xgb_prediction
        interval_width = self._max_interval_width()
        row["predicted_utilization_1h"] = ensemble_prediction
        row["rf_predicted_utilization_1h"] = rf_prediction
        row["ensemble_predicted_utilization_1h"] = ensemble_prediction
        row["prediction_interval_lower"] = max(0.0, ensemble_prediction - interval_width / 2)
        row["prediction_interval_upper"] = min(1.0, ensemble_prediction + interval_width / 2)
        row["uncertainty_width"] = float(row["prediction_interval_upper"].iloc[0]) - float(row["prediction_interval_lower"].iloc[0])
        row["predicted_available_spaces_1h"] = round(float(row["capacity"].iloc[0]) * (1 - min(max(ensemble_prediction, 0), 1)))
        scores = self._score_live_forecast(row)
        for key, value in scores.items():
            row[key] = value

        existing = self.store.live_forecasts()
        if existing.empty:
            current_rank = 1
        else:
            simulated = existing[existing["system_code"] != request.system_code].copy()
            simulated = pd.concat([simulated, row], ignore_index=True, sort=False)
            simulated = simulated.sort_values("balanced_score", ascending=False).reset_index(drop=True)
            simulated["recommendation_rank"] = simulated.index + 1
            current_rank = int(simulated[simulated["system_code"] == request.system_code]["recommendation_rank"].iloc[0])
        row["recommendation_rank"] = current_rank

        payload = row.iloc[0].to_dict()
        observation_id = self.store.append_live_observation(payload)
        self.store.upsert_live_forecast(request.system_code, payload)
        result = {
            "observation_id": observation_id,
            "system_code": request.system_code,
            "time_slot": str(row["time_slot"].iloc[0]),
            "predicted_utilization_1h": round(float(ensemble_prediction), 6),
            "prediction_interval_lower": round(float(row["prediction_interval_lower"].iloc[0]), 6),
            "prediction_interval_upper": round(float(row["prediction_interval_upper"].iloc[0]), 6),
            "predicted_free_spaces_1h": int(row["predicted_available_spaces_1h"].iloc[0]),
            "recommendation_rank": current_rank,
        }
        self.store.append_audit_event("live_ingest", result, actor="service")
        return result

    def ingest_payload(self, payload: dict[str, Any]) -> dict:
        return self.ingest_observation(IngestObservationRequest(**payload))

    def what_if(self, request: WhatIfRequest) -> dict:
        current_rows = self.feature_df[self.feature_df["system_code"] == request.system_code].sort_values("time_slot")
        if current_rows.empty:
            raise ValueError(f"Unknown system_code: {request.system_code}")

        row = current_rows.tail(1).copy()
        if request.queue_length is not None:
            row["queue_length"] = request.queue_length
        if request.traffic_condition_nearby is not None:
            row["traffic_condition_nearby"] = request.traffic_condition_nearby
            row["traffic_level"] = {"low": 0, "average": 1, "high": 2}[request.traffic_condition_nearby]
        if request.is_special_day is not None:
            row["is_special_day"] = request.is_special_day
        if request.vehicle_type is not None:
            row["vehicle_type"] = request.vehicle_type

        row["capacity_pressure"] = row["queue_length"] / row["capacity"]
        row["parking_stress_index"] = (
            0.7 * row["utilization"] + 0.2 * row["traffic_level"] / 2 + 0.1 * row["capacity_pressure"]
        ).clip(0, 1)
        row["rolling_queue_3"] = row["queue_length"]
        row["rolling_queue_6"] = row["queue_length"]
        row["rolling_stress_3"] = row["parking_stress_index"]

        features = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        xgb_prediction = float(self.xgb_model.predict(features)[0])
        rf_prediction = float(self.rf_model.predict(features)[0])
        rf_weight = self.registry["ensemble_weights"]["random_forest"]
        xgb_weight = self.registry["ensemble_weights"]["xgboost"]
        ensemble_prediction = rf_weight * rf_prediction + xgb_weight * xgb_prediction
        interval_width = max(
            item for item in [
                record.get("uncertainty_width", 0)
                for record in self.latest_df[["uncertainty_width"]].fillna(0).to_dict(orient="records")
            ]
        )
        lower = max(0.0, ensemble_prediction - interval_width / 2)
        upper = min(1.0, ensemble_prediction + interval_width / 2)
        free_spaces = int(round(float(row["capacity"].iloc[0] * (1 - min(max(ensemble_prediction, 0), 1)))))
        result = {
            "system_code": request.system_code,
            "predicted_utilization_1h": round(float(ensemble_prediction), 6),
            "prediction_interval_lower": round(float(lower), 6),
            "prediction_interval_upper": round(float(upper), 6),
            "predicted_free_spaces_1h": free_spaces,
            "xgboost_prediction": round(float(xgb_prediction), 6),
            "random_forest_prediction": round(float(rf_prediction), 6),
        }
        self.store.append_audit_event("what_if_analysis", result, actor="service")
        return result

    def run_demo_scenario(self, scenario_name: str, steps: int = 3, reset_first: bool = True) -> dict:
        if reset_first:
            self.reset_live_state(clear_jobs=False)
        result = self.demo_engine.run(self, scenario_name=scenario_name, steps=steps)
        self.store.append_audit_event(
            "demo_scenario_run",
            {
                "scenario_name": scenario_name,
                "steps": steps,
                "events_executed": result["events_executed"],
            },
            actor="service",
        )
        return result

    def run_demo_playbook(self, playbook_name: str, reset_first: bool = True) -> dict:
        result = self.demo_engine.run_playbook(self, playbook_name=playbook_name, reset_first=reset_first)
        self.store.append_audit_event(
            "demo_playbook_run",
            {
                "playbook_name": playbook_name,
                "total_events_executed": result["total_events_executed"],
                "stages": len(result["stages"]),
            },
            actor="service",
        )
        return result


@lru_cache(maxsize=1)
def get_service() -> SmartParkingService:
    return SmartParkingService()


def reset_service_cache() -> None:
    get_service.cache_clear()
