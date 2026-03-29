from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_parking.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["systems"] >= 1
    assert "X-Request-ID" in response.headers


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ready", "degraded"}
    assert "checks" in body


def test_recommendations_endpoint():
    response = client.get("/recommendations", params={"objective": "Lowest Risk", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) >= 1
    assert "system_code" in body["items"][0]


def test_what_if_endpoint():
    response = client.post(
        "/forecast/what-if",
        json={
            "system_code": "BHMBCCMKT01",
            "queue_length": 4,
            "traffic_condition_nearby": "average",
            "is_special_day": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_utilization_1h"] <= 1
    assert "predicted_free_spaces_1h" in body


def test_live_ingest_and_state_endpoint():
    reset_response = client.post("/demo/reset", json={"clear_jobs": False})
    assert reset_response.status_code == 200

    ingest_response = client.post(
        "/live/ingest",
        json={
            "system_code": "BHMBCCMKT01",
            "occupancy": 290,
            "queue_length": 4,
            "traffic_condition_nearby": "average",
            "is_special_day": 0,
            "timestamp": "2016-12-19 17:00:00",
        },
    )
    assert ingest_response.status_code == 200
    ingest_body = ingest_response.json()
    assert ingest_body["system_code"] == "BHMBCCMKT01"
    assert 0 <= ingest_body["predicted_utilization_1h"] <= 1

    state_response = client.get("/live/state")
    assert state_response.status_code == 200
    state_body = state_response.json()
    assert "forecasts" in state_body
    assert len(state_body["forecasts"]) >= 1


def test_demo_scenarios_endpoint():
    response = client.get("/demo/scenarios")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) >= 1
    assert any(item["name"] == "rush_hour_surge" for item in body["items"])


def test_demo_playbooks_endpoint():
    response = client.get("/demo/playbooks")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) >= 1
    assert any(item["name"] == "executive_showcase" for item in body["items"])


def test_demo_run_endpoint():
    reset_response = client.post("/demo/reset", json={"clear_jobs": False})
    assert reset_response.status_code == 200

    run_response = client.post(
        "/demo/run",
        json={
            "scenario_name": "spillover_disruption",
            "steps": 2,
            "reset_first": False,
        },
    )
    assert run_response.status_code == 200
    body = run_response.json()
    assert body["scenario_name"] == "spillover_disruption"
    assert body["steps_executed"] == 2
    assert body["events_executed"] >= 2
    assert len(body["top_recommendations"]) >= 1


def test_demo_playbook_run_endpoint():
    reset_response = client.post("/demo/reset", json={"clear_jobs": False})
    assert reset_response.status_code == 200

    response = client.post(
        "/demo/playbook",
        json={
            "playbook_name": "executive_showcase",
            "reset_first": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["playbook_name"] == "executive_showcase"
    assert len(body["stages"]) == 3
    assert body["total_events_executed"] >= 1


def test_ops_summary_and_alerts_endpoints():
    summary_response = client.get("/ops/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert "live_snapshot" in summary
    assert "alerts" in summary
    assert "talking_points" in summary
    assert len(summary["talking_points"]) >= 1

    alerts_response = client.get("/ops/alerts")
    assert alerts_response.status_code == 200
    alerts = alerts_response.json()
    assert len(alerts["items"]) >= 1
    assert "severity" in alerts["items"][0]
    assert "X-RateLimit-Limit" in summary_response.headers


def test_ops_activity_endpoint():
    client.get("/ops/summary")
    activity_response = client.get("/ops/activity", params={"limit": 5})
    assert activity_response.status_code == 200
    body = activity_response.json()
    assert len(body["items"]) >= 1
    assert "source" in body["items"][0]
