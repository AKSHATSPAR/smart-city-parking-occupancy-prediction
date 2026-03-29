from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import LIVE_OPS_DB_PATH


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveOpsStore:
    def __init__(self, db_path: Path = LIVE_OPS_DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_observations (
                    id TEXT PRIMARY KEY,
                    system_code TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_forecasts (
                    system_code TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrain_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    message TEXT
                );

                CREATE TABLE IF NOT EXISTS api_request_logs (
                    request_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    client_host TEXT,
                    duration_ms REAL NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def seed_live_forecasts(self, recommendations_df: pd.DataFrame) -> None:
        with self.lock, self._connect() as connection:
            existing_count = connection.execute("SELECT COUNT(*) FROM live_forecasts").fetchone()[0]
            if existing_count:
                return
            now = utc_now_iso()
            for _, row in recommendations_df.iterrows():
                connection.execute(
                    """
                    INSERT OR REPLACE INTO live_forecasts(system_code, payload_json, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (str(row["system_code"]), json.dumps(_safe_record(row.to_dict())), now),
                )
            connection.commit()

    def reset_live_state(self, recommendations_df: pd.DataFrame, clear_jobs: bool = False) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("DELETE FROM live_observations")
            connection.execute("DELETE FROM live_forecasts")
            if clear_jobs:
                connection.execute("DELETE FROM retrain_jobs")
            now = utc_now_iso()
            for _, row in recommendations_df.iterrows():
                connection.execute(
                    """
                    INSERT OR REPLACE INTO live_forecasts(system_code, payload_json, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (str(row["system_code"]), json.dumps(_safe_record(row.to_dict())), now),
                )
            connection.commit()

    def append_live_observation(self, payload: dict[str, Any]) -> str:
        observation_id = str(uuid.uuid4())
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO live_observations(id, system_code, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (observation_id, payload["system_code"], json.dumps(_safe_record(payload)), utc_now_iso()),
            )
            connection.commit()
        return observation_id

    def live_observations(self, system_code: str | None = None) -> pd.DataFrame:
        with self._connect() as connection:
            if system_code is None:
                query = "SELECT payload_json FROM live_observations ORDER BY created_at ASC"
                rows = connection.execute(query).fetchall()
            else:
                query = "SELECT payload_json FROM live_observations WHERE system_code = ? ORDER BY created_at ASC"
                rows = connection.execute(query, (system_code,)).fetchall()
        records = [json.loads(item[0]) for item in rows]
        return pd.DataFrame(records)

    def upsert_live_forecast(self, system_code: str, payload: dict[str, Any]) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO live_forecasts(system_code, payload_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (system_code, json.dumps(_safe_record(payload)), utc_now_iso()),
            )
            connection.commit()

    def live_forecasts(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload_json FROM live_forecasts").fetchall()
        records = [json.loads(item[0]) for item in rows]
        return pd.DataFrame(records)

    def create_retrain_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retrain_jobs(job_id, status, created_at, message)
                VALUES (?, 'queued', ?, ?)
                """,
                (job_id, utc_now_iso(), "Retraining job queued"),
            )
            connection.commit()
        return job_id

    def append_api_request_log(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        client_host: str | None,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO api_request_logs(
                    request_id, method, path, status_code, client_host, duration_ms, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    method,
                    path,
                    int(status_code),
                    client_host,
                    float(duration_ms),
                    json.dumps(_safe_record(metadata or {})),
                    utc_now_iso(),
                ),
            )
            connection.commit()

    def api_request_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT request_id, method, path, status_code, client_host, duration_ms, metadata_json, created_at
                FROM api_request_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = []
        for row in rows:
            metadata = json.loads(row[6]) if row[6] else {}
            records.append(
                {
                    "request_id": row[0],
                    "method": row[1],
                    "path": row[2],
                    "status_code": row[3],
                    "client_host": row[4],
                    "duration_ms": row[5],
                    "metadata": metadata,
                    "created_at": row[7],
                }
            )
        return records

    def append_audit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        actor: str = "system",
    ) -> str:
        event_id = str(uuid.uuid4())
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(event_id, event_type, actor, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, event_type, actor, json.dumps(_safe_record(payload)), utc_now_iso()),
            )
            connection.commit()
        return event_id

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, event_type, actor, payload_json, created_at
                FROM audit_events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = []
        for row in rows:
            records.append(
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "actor": row[2],
                    "payload": json.loads(row[3]),
                    "created_at": row[4],
                }
            )
        return records

    def update_retrain_job(self, job_id: str, status: str, message: str) -> None:
        timestamps = {
            "started_at": utc_now_iso() if status == "running" else None,
            "finished_at": utc_now_iso() if status in {"completed", "failed"} else None,
        }
        with self.lock, self._connect() as connection:
            if status == "running":
                connection.execute(
                    """
                    UPDATE retrain_jobs
                    SET status = ?, started_at = ?, message = ?
                    WHERE job_id = ?
                    """,
                    (status, timestamps["started_at"], message, job_id),
                )
            elif status in {"completed", "failed"}:
                connection.execute(
                    """
                    UPDATE retrain_jobs
                    SET status = ?, finished_at = ?, message = ?
                    WHERE job_id = ?
                    """,
                    (status, timestamps["finished_at"], message, job_id),
                )
            else:
                connection.execute(
                    "UPDATE retrain_jobs SET status = ?, message = ? WHERE job_id = ?",
                    (status, message, job_id),
                )
            connection.commit()

    def retrain_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, status, created_at, started_at, finished_at, message
                FROM retrain_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        columns = ["job_id", "status", "created_at", "started_at", "finished_at", "message"]
        return dict(zip(columns, row))

    def retrain_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, status, created_at, started_at, finished_at, message
                FROM retrain_jobs
                ORDER BY created_at DESC
                """
            ).fetchall()
        columns = ["job_id", "status", "created_at", "started_at", "finished_at", "message"]
        return [dict(zip(columns, row)) for row in rows]


def _safe_record(payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, pd.Timestamp):
            output[key] = value.isoformat()
        elif isinstance(value, float) and pd.isna(value):
            output[key] = None
        else:
            output[key] = value
    return output
