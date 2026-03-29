from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_parking.config import (
    ANOMALIES_PATH,
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    LATEST_FORECAST_PATH,
    LOCATION_PROFILE_PATH,
    MODEL_METRICS_PATH,
    MULTI_HORIZON_METRICS_PATH,
    RECOMMENDATIONS_PATH,
    SQLITE_DB_PATH,
    TEST_PREDICTIONS_PATH,
    XGB_MODEL_PATH,
)
from smart_parking.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from smart_parking.service import get_service


st.set_page_config(
    page_title="Advanced Smart Parking Command Center",
    page_icon="P",
    layout="wide",
)

st.title("Advanced Smart Parking Command Center")
st.caption(
    "Forecasting, recommendation ranking, spatial intelligence, anomaly detection, and decision support for urban parking."
)


@st.cache_data
def load_system_catalog() -> list[str]:
    frame = pd.read_csv(CLEAN_DATA_PATH, usecols=["system_code"])
    return sorted(frame["system_code"].dropna().unique().tolist())


@st.cache_data
def load_overview_bundle() -> dict[str, pd.DataFrame]:
    return {
        "clean": pd.read_csv(CLEAN_DATA_PATH, parse_dates=["timestamp", "time_slot"]),
        "metrics": pd.read_csv(MODEL_METRICS_PATH),
        "multi_horizon": pd.read_csv(MULTI_HORIZON_METRICS_PATH),
        "latest": pd.read_csv(LATEST_FORECAST_PATH, parse_dates=["time_slot", "target_time_slot"]),
        "recommendations": pd.read_csv(RECOMMENDATIONS_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"]),
    }


@st.cache_data
def load_forecast_bundle() -> dict[str, pd.DataFrame]:
    return {
        "predictions": pd.read_csv(TEST_PREDICTIONS_PATH, parse_dates=["time_slot", "target_time_slot"]),
        "latest": pd.read_csv(LATEST_FORECAST_PATH, parse_dates=["time_slot", "target_time_slot"]),
        "features": pd.read_csv(FEATURE_DATA_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"]),
    }


@st.cache_data
def load_recommendation_bundle() -> pd.DataFrame:
    return pd.read_csv(RECOMMENDATIONS_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"])


@st.cache_data
def load_risk_bundle() -> dict[str, pd.DataFrame]:
    return {
        "anomalies": pd.read_csv(ANOMALIES_PATH, parse_dates=["timestamp", "time_slot", "target_time_slot"]),
        "profiles": pd.read_csv(LOCATION_PROFILE_PATH),
        "clean": pd.read_csv(CLEAN_DATA_PATH, parse_dates=["timestamp", "time_slot"]),
    }


@st.cache_resource
def load_xgb_pipeline():
    return joblib.load(XGB_MODEL_PATH)


@st.cache_resource
def load_service():
    return get_service()


def rank_recommendations(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    sort_map = {
        "Balanced": "balanced_score",
        "Maximum Availability": "max_availability_score",
        "Lowest Risk": "low_risk_score",
        "Lowest Congestion": "low_congestion_score",
    }
    score_column = sort_map[objective]
    ranked = frame.sort_values(score_column, ascending=False).copy()
    ranked["objective_rank"] = ranked[score_column].rank(ascending=False, method="dense").astype(int)
    return ranked


def live_reference_frame(service) -> pd.DataFrame:
    live_df = service.store.live_forecasts()
    if live_df.empty:
        live_df = service.recommendations_df.copy()
    return live_df.copy()


def live_observations_frame(service, limit: int = 25) -> pd.DataFrame:
    live_df = service.store.live_observations().copy()
    if live_df.empty:
        return live_df
    for column in ["timestamp", "time_slot", "target_time_slot"]:
        if column in live_df.columns:
            live_df[column] = pd.to_datetime(live_df[column], errors="coerce")
    sort_column = "time_slot" if "time_slot" in live_df.columns else "timestamp"
    if sort_column in live_df.columns:
        live_df = live_df.sort_values(sort_column, ascending=False)
    return live_df.head(limit).reset_index(drop=True)


SECTION_OPTIONS = [
    "Overview",
    "Live Demo",
    "Ops Wall",
    "Forecast Lab",
    "Recommendations",
    "Risk & Anomalies",
    "Spatial Intelligence",
    "SQL Snapshot",
]


def render_section_selector() -> str:
    default_section = st.session_state.get("cloud_selected_section", "Overview")
    default_section = default_section if default_section in SECTION_OPTIONS else "Overview"

    st.markdown("#### Workspace")
    if hasattr(st, "pills"):
        section = st.pills(
            "Workspace",
            SECTION_OPTIONS,
            selection_mode="single",
            default=default_section,
            label_visibility="collapsed",
        )
    else:
        section = st.radio(
            "Workspace",
            SECTION_OPTIONS,
            index=SECTION_OPTIONS.index(default_section),
            horizontal=True,
            label_visibility="collapsed",
        )

    section = section or default_section
    st.session_state["cloud_selected_section"] = section
    st.divider()
    return section


section = render_section_selector()
st.sidebar.header("Control Panel")
system_catalog = load_system_catalog()
selected_system = st.sidebar.selectbox("Parking location", system_catalog)
render_map_charts = st.sidebar.toggle("Render map charts", value=True)
recommendation_objective = st.sidebar.selectbox(
    "Recommendation objective",
    ["Balanced", "Maximum Availability", "Lowest Risk", "Lowest Congestion"],
)


if section == "Overview":
    data = load_overview_bundle()
    clean_df = data["clean"]
    metrics_df = data["metrics"]
    multi_horizon_df = data["multi_horizon"]
    latest_df = data["latest"]
    recommendations_df = data["recommendations"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parking Systems", int(clean_df["system_code"].nunique()))
    col2.metric("Observations", int(clean_df.shape[0]))
    col3.metric("Average Utilization", f"{clean_df['utilization'].mean():.2%}")
    col4.metric("Top Recommendation", recommendations_df.sort_values("recommendation_rank").iloc[0]["system_code"])

    st.subheader("Model Performance")
    st.dataframe(
        metrics_df.style.format({"mae": "{:.4f}", "rmse": "{:.4f}", "r2": "{:.4f}", "band_accuracy": "{:.4f}"}),
        width="stretch",
    )

    horizon_chart = px.bar(
        multi_horizon_df.sort_values("rmse"),
        x="forecast_horizon",
        y="rmse",
        color="forecast_horizon",
        title="Forecasting error across multiple horizons",
    )
    st.plotly_chart(horizon_chart, width="stretch")

    latest_view = latest_df.copy()
    for column in ["utilization", "predicted_utilization_1h", "prediction_interval_lower", "prediction_interval_upper"]:
        latest_view[column] = latest_view[column].map(lambda value: f"{value:.2%}")
    st.subheader("Latest forecast snapshot")
    st.dataframe(latest_view, width="stretch")


elif section == "Live Demo":
    try:
        service = load_service()
    except Exception as exc:
        st.error("The cloud runtime failed while loading live demo services.")
        st.exception(exc)
        st.stop()

    if "cloud_last_demo_action" not in st.session_state:
        st.session_state["cloud_last_demo_action"] = None
    if "cloud_last_demo_result" not in st.session_state:
        st.session_state["cloud_last_demo_result"] = None

    live_df = live_reference_frame(service)
    live_recommendations_df = pd.DataFrame(service.recommendations(objective="Balanced", limit=14))
    live_observations_df = live_observations_frame(service, limit=20)

    risk_counts = (
        live_df["risk_band"].fillna("Unknown").value_counts().rename_axis("risk_band").reset_index(name="count")
        if not live_df.empty
        else pd.DataFrame(columns=["risk_band", "count"])
    )
    top_live_choice = live_recommendations_df.iloc[0]["system_code"] if not live_recommendations_df.empty else "N/A"
    critical_count = int((live_df["risk_band"] == "Critical").sum()) if not live_df.empty else 0
    live_mean_utilization = float(live_df["predicted_utilization_1h"].mean()) if not live_df.empty else 0.0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Live observations", int(len(live_observations_df)))
    metric_col2.metric("Best live option", top_live_choice)
    metric_col3.metric("Critical systems", critical_count)
    metric_col4.metric("Mean live utilization", f"{live_mean_utilization:.2%}")

    st.caption("Trigger scenario stress, inject custom incidents, and show live recommendation changes during your demo.")

    action_col1, action_col2, action_col3 = st.columns([1.1, 1.1, 0.8])
    with action_col1:
        scenario_catalog = service.demo_scenarios()
        scenario_options = {f"{item['title']} ({item['name']})": item for item in scenario_catalog}
        with st.form("cloud_run_demo_scenario_form"):
            selected_scenario_label = st.selectbox("Scenario", list(scenario_options.keys()))
            selected_scenario = scenario_options[selected_scenario_label]
            st.caption(selected_scenario["description"])
            scenario_steps = st.slider("Scenario steps", min_value=1, max_value=6, value=int(selected_scenario["recommended_steps"]))
            reset_before_run = st.checkbox("Reset before run", value=True)
            run_scenario = st.form_submit_button("Run Scenario", type="primary", width="stretch")
        if run_scenario:
            result = service.run_demo_scenario(
                scenario_name=selected_scenario["name"],
                steps=scenario_steps,
                reset_first=reset_before_run,
            )
            st.session_state["cloud_last_demo_action"] = f"Scenario executed: {selected_scenario['title']}"
            st.session_state["cloud_last_demo_result"] = result
            st.rerun()

        playbook_catalog = service.demo_playbooks()
        playbook_options = {f"{item['title']} ({item['name']})": item for item in playbook_catalog}
        with st.form("cloud_run_demo_playbook_form"):
            selected_playbook_label = st.selectbox("Guided demo", list(playbook_options.keys()))
            selected_playbook = playbook_options[selected_playbook_label]
            st.caption(selected_playbook["description"])
            playbook_stage_lines = [
                f"{idx + 1}. {stage['scenario_name']} x{stage['steps']}: {stage['narrative']}"
                for idx, stage in enumerate(selected_playbook["stages"])
            ]
            st.code("\n".join(playbook_stage_lines), language="text")
            reset_before_playbook = st.checkbox("Reset before guided demo", value=True)
            run_playbook = st.form_submit_button("Run Guided Demo", type="primary", width="stretch")
        if run_playbook:
            result = service.run_demo_playbook(
                playbook_name=selected_playbook["name"],
                reset_first=reset_before_playbook,
            )
            st.session_state["cloud_last_demo_action"] = f"Guided demo executed: {selected_playbook['title']}"
            st.session_state["cloud_last_demo_result"] = result
            st.rerun()

    with action_col2:
        live_systems = sorted(live_df["system_code"].dropna().unique().tolist()) if not live_df.empty else []
        with st.form("cloud_manual_live_injection_form"):
            selected_live_system = st.selectbox("Manual incident system", live_systems)
            selected_live_row = live_df[live_df["system_code"] == selected_live_system].iloc[0]
            default_utilization_pct = int(round(float(selected_live_row.get("utilization", 0.5)) * 100))
            default_queue = int(round(float(selected_live_row.get("queue_length", 0))))
            traffic_options = ["low", "average", "high"]
            current_traffic = str(selected_live_row.get("traffic_condition_nearby", "average"))
            traffic_index = traffic_options.index(current_traffic) if current_traffic in traffic_options else 1
            manual_utilization_pct = st.slider("Target occupancy (%)", min_value=0, max_value=100, value=min(max(default_utilization_pct, 0), 100))
            manual_queue = st.slider("Queue length", min_value=0, max_value=30, value=max(default_queue, 0))
            manual_traffic = st.selectbox("Traffic", traffic_options, index=traffic_index)
            manual_special_day = st.checkbox("Special day", value=bool(int(selected_live_row.get("is_special_day", 0))))
            inject_event = st.form_submit_button("Inject Custom Event", width="stretch")
        if inject_event:
            capacity = float(selected_live_row["capacity"])
            payload = {
                "system_code": selected_live_system,
                "occupancy": round(capacity * manual_utilization_pct / 100, 2),
                "queue_length": int(manual_queue),
                "traffic_condition_nearby": manual_traffic,
                "is_special_day": int(manual_special_day),
                "vehicle_type": str(selected_live_row.get("vehicle_type", "car")),
                "capacity": capacity,
                "latitude": float(selected_live_row["latitude"]),
                "longitude": float(selected_live_row["longitude"]),
            }
            result = service.ingest_payload(payload)
            st.session_state["cloud_last_demo_action"] = f"Custom event injected: {selected_live_system}"
            st.session_state["cloud_last_demo_result"] = {"manual_injection": result}
            st.rerun()

    with action_col3:
        if st.button("Refresh Live Board", width="stretch"):
            st.rerun()
        if st.button("Reset Live State", width="stretch"):
            reset_result = service.reset_live_state(clear_jobs=False)
            st.session_state["cloud_last_demo_action"] = "Live state reset"
            st.session_state["cloud_last_demo_result"] = reset_result
            st.rerun()

    if st.session_state["cloud_last_demo_action"]:
        st.info(st.session_state["cloud_last_demo_action"])

    last_demo_result = st.session_state.get("cloud_last_demo_result")
    if isinstance(last_demo_result, dict) and "stages" in last_demo_result:
        stage_summary_df = pd.DataFrame(
            [
                {
                    "stage": item["stage_number"],
                    "scenario": item["scenario_title"],
                    "steps": item["steps"],
                    "events_executed": item["events_executed"],
                    "delta_mean_utilization": item["delta_mean_utilization"],
                    "delta_critical_systems": item["delta_critical_systems"],
                    "top_recommendation_after": item["top_recommendation_after"]["system_code"]
                    if item["top_recommendation_after"]
                    else None,
                }
                for item in last_demo_result["stages"]
            ]
        )
        st.dataframe(stage_summary_df, width="stretch")
        with st.expander("Guided demo narration", expanded=True):
            for item in last_demo_result["stages"]:
                st.markdown(
                    f"**Stage {item['stage_number']}: {item['scenario_title']}**  \n"
                    f"{item['narrative']}  \n"
                    f"Events executed: `{item['events_executed']}` | "
                    f"Mean utilization delta: `{item['delta_mean_utilization']:+.4f}` | "
                    f"Critical systems delta: `{item['delta_critical_systems']:+d}`"
                )
    elif isinstance(last_demo_result, dict) and "step_summaries" in last_demo_result:
        step_summary_df = pd.DataFrame(
            [
                {
                    "step": item["step"],
                    "events_applied": len(item["events_applied"]),
                    "portfolio_mean_utilization": item["portfolio_mean_utilization"],
                    "critical_systems": item["critical_systems"],
                    "top_recommendation": item["top_recommendation"]["system_code"],
                    "most_congested_system": item["most_congested_system"]["system_code"],
                }
                for item in last_demo_result["step_summaries"]
            ]
        )
        st.dataframe(step_summary_df, width="stretch")
    elif last_demo_result:
        with st.expander("Latest live action payload", expanded=False):
            st.json(last_demo_result)

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        if not risk_counts.empty:
            risk_chart = px.pie(
                risk_counts,
                names="risk_band",
                values="count",
                hole=0.55,
                title="Live risk distribution",
            )
            st.plotly_chart(risk_chart, width="stretch")
        else:
            st.info("No live risk data available yet.")

    with viz_col2:
        if render_map_charts and not live_recommendations_df.empty:
            live_map = px.scatter_map(
                live_recommendations_df,
                lat="latitude",
                lon="longitude",
                color="risk_band",
                size="predicted_available_spaces_1h",
                hover_name="system_code",
                hover_data={
                    "recommendation_rank": True,
                    "predicted_utilization_1h": ":.2f",
                    "predicted_available_spaces_1h": True,
                },
                zoom=11,
                height=430,
                title="Live recommendation board",
            )
            st.plotly_chart(live_map, width="stretch")
        elif not live_recommendations_df.empty:
            fallback_geo = px.scatter(
                live_recommendations_df,
                x="longitude",
                y="latitude",
                color="risk_band",
                size="predicted_available_spaces_1h",
                hover_name="system_code",
                title="Live recommendation board",
            )
            st.plotly_chart(fallback_geo, width="stretch")
        else:
            st.info("Live recommendations will appear here after the board loads.")

    st.subheader("Current live recommendation board")
    if not live_recommendations_df.empty:
        recommendation_columns = [
            "recommendation_rank",
            "system_code",
            "predicted_available_spaces_1h",
            "predicted_utilization_1h",
            "prediction_interval_upper",
            "risk_band",
            "balanced_score",
        ]
        st.dataframe(live_recommendations_df[recommendation_columns], width="stretch")
    else:
        st.info("No live recommendations are available.")

    if not live_observations_df.empty:
        st.subheader("Recent live observations")
        observation_columns = [
            "system_code",
            "time_slot",
            "occupancy_clean",
            "queue_length",
            "traffic_condition_nearby",
            "predicted_utilization_1h",
            "risk_band",
            "recommendation_rank",
        ]
        available_columns = [column for column in observation_columns if column in live_observations_df.columns]
        st.dataframe(live_observations_df[available_columns], width="stretch")
    else:
        st.info("No live observations yet. Run a scenario or inject an incident to populate this table.")


elif section == "Ops Wall":
    try:
        service = load_service()
    except Exception as exc:
        st.error("The cloud runtime failed while loading ops services.")
        st.exception(exc)
        st.stop()

    st.caption("Live operational health, alerting, drift tracking, model quality, and retraining status.")
    if st.button("Refresh Ops Wall", key="cloud_refresh_ops_wall", width="stretch"):
        st.rerun()

    ops_summary = service.ops_summary()
    live_snapshot = ops_summary["live_snapshot"]
    drift_overview = ops_summary["drift_overview"]
    model_overview = ops_summary["model_overview"]
    retrain_status = ops_summary["retrain_status"]

    ops_metric_col1, ops_metric_col2, ops_metric_col3, ops_metric_col4, ops_metric_col5 = st.columns(5)
    ops_metric_col1.metric("Champion Model", model_overview["champion_model"])
    ops_metric_col2.metric("Critical Systems", int(live_snapshot["critical_systems"]))
    ops_metric_col3.metric("Low Availability", int(live_snapshot["low_availability_systems"]))
    ops_metric_col4.metric("Drift Severity", drift_overview["severity"].title())
    ops_metric_col5.metric("Latest Retrain", str(retrain_status.get("status", "not_run")).title())

    ops_health_col1, ops_health_col2 = st.columns(2)
    with ops_health_col1:
        severity_counts_df = pd.DataFrame(
            [{"severity": key, "count": value} for key, value in ops_summary["alert_counts"].items()]
        )
        severity_chart = px.bar(
            severity_counts_df,
            x="severity",
            y="count",
            color="severity",
            color_discrete_map={"critical": "#c92a2a", "warning": "#f08c00", "info": "#1971c2"},
            title="Operational alert mix",
        )
        st.plotly_chart(severity_chart, width="stretch")

    with ops_health_col2:
        drift_features_df = pd.DataFrame(drift_overview["top_features"])
        if not drift_features_df.empty:
            drift_chart = px.bar(
                drift_features_df.sort_values("psi", ascending=True),
                x="psi",
                y="feature",
                color="drift_level",
                orientation="h",
                title="Top monitored drift features",
                color_discrete_map={"stable": "#2b8a3e", "warning": "#f08c00", "critical": "#c92a2a"},
            )
            st.plotly_chart(drift_chart, width="stretch")
        else:
            st.info("No drift data available.")

    st.subheader("Priority alerts")
    for alert in ops_summary["alerts"]:
        message = f"{alert['title']}: {alert['message']} Action: {alert['recommended_action']}"
        if alert["severity"] == "critical":
            st.error(message)
        elif alert["severity"] == "warning":
            st.warning(message)
        else:
            st.info(message)

    st.subheader("Demo talking points")
    st.markdown("\n".join(f"- {point}" for point in ops_summary["talking_points"]))

    ops_data_col1, ops_data_col2 = st.columns(2)
    with ops_data_col1:
        st.markdown("**Critical Watchlist**")
        st.dataframe(pd.DataFrame(ops_summary["critical_watchlist"]), width="stretch")
        st.markdown("**Recommended Alternatives**")
        st.dataframe(pd.DataFrame(ops_summary["recommended_alternatives"]), width="stretch")

    with ops_data_col2:
        st.markdown("**Model Health Snapshot**")
        model_health_df = pd.DataFrame(
            [
                {"metric": "Test RMSE", "value": model_overview["test_rmse"]},
                {"metric": "Test Band Accuracy", "value": model_overview["test_band_accuracy"]},
                {"metric": "Rolling Band Accuracy", "value": model_overview["rolling_band_accuracy"]},
                {"metric": "Features", "value": model_overview["feature_count"]},
            ]
        )
        st.dataframe(model_health_df, width="stretch")

        st.markdown("**Retraining History**")
        retrain_jobs_df = pd.DataFrame(ops_summary["retrain_jobs"])
        if not retrain_jobs_df.empty:
            st.dataframe(retrain_jobs_df, width="stretch")
        else:
            st.info("No retraining jobs recorded yet.")

    st.subheader("Recent Operational Activity")
    recent_activity_df = pd.DataFrame(ops_summary["recent_activity"])
    if not recent_activity_df.empty:
        st.dataframe(recent_activity_df, width="stretch")
    else:
        st.info("No API or audit activity has been recorded yet.")


elif section == "Forecast Lab":
    data = load_forecast_bundle()
    predictions_df = data["predictions"]
    latest_df = data["latest"]
    feature_df = data["features"]
    xgb_pipeline = load_xgb_pipeline()

    available_models = sorted(predictions_df["model"].unique().tolist())
    selected_model = st.selectbox("Forecast model", available_models)

    system_predictions = predictions_df[
        (predictions_df["system_code"] == selected_system) & (predictions_df["model"] == selected_model)
    ].sort_values("target_time_slot")

    forecast_figure = go.Figure()
    forecast_figure.add_trace(
        go.Scatter(
            x=system_predictions["target_time_slot"],
            y=system_predictions["target_utilization_1h"],
            mode="lines",
            name="Actual utilization",
            line=dict(color="#0b7285", width=2),
        )
    )
    if "prediction_interval_lower" in system_predictions.columns and system_predictions["prediction_interval_lower"].notna().any():
        lower = system_predictions["prediction_interval_lower"].fillna(system_predictions["predicted_utilization_1h"])
        upper = system_predictions["prediction_interval_upper"].fillna(system_predictions["predicted_utilization_1h"])
        forecast_figure.add_trace(go.Scatter(x=system_predictions["target_time_slot"], y=lower, line=dict(width=0), hoverinfo="skip", showlegend=False))
        forecast_figure.add_trace(
            go.Scatter(
                x=system_predictions["target_time_slot"],
                y=upper,
                fill="tonexty",
                fillcolor="rgba(255,140,0,0.15)",
                line=dict(width=0),
                hoverinfo="skip",
                name="Prediction interval",
            )
        )
    forecast_figure.add_trace(
        go.Scatter(
            x=system_predictions["target_time_slot"],
            y=system_predictions["predicted_utilization_1h"],
            mode="lines",
            name=f"{selected_model} prediction",
            line=dict(color="#ff6b6b", width=2),
        )
    )
    forecast_figure.update_layout(title=f"{selected_model} forecast for {selected_system}", yaxis_title="Utilization")
    st.plotly_chart(forecast_figure, width="stretch")

    st.subheader("What-if next-hour simulation")
    simulation_rows = feature_df[feature_df["system_code"] == selected_system].sort_values("time_slot")
    selected_index = st.selectbox(
        "Choose historical context",
        simulation_rows.index.tolist(),
        format_func=lambda idx: simulation_rows.loc[idx, "time_slot"].strftime("%Y-%m-%d %H:%M"),
    )
    selected_row = simulation_rows.loc[[selected_index]].copy()

    adjusted_queue = st.slider("Queue length", min_value=0, max_value=15, value=int(round(float(selected_row["queue_length"].iloc[0]))))
    adjusted_traffic = st.selectbox(
        "Nearby traffic condition",
        options=["low", "average", "high"],
        index=["low", "average", "high"].index(str(selected_row["traffic_condition_nearby"].iloc[0])),
    )
    adjusted_special_day = st.toggle("Special day", value=bool(int(selected_row["is_special_day"].iloc[0])))

    selected_row["queue_length"] = adjusted_queue
    selected_row["traffic_condition_nearby"] = adjusted_traffic
    selected_row["is_special_day"] = int(adjusted_special_day)
    selected_row["traffic_level"] = {"low": 0, "average": 1, "high": 2}[adjusted_traffic]
    selected_row["capacity_pressure"] = selected_row["queue_length"] / selected_row["capacity"]
    selected_row["parking_stress_index"] = (
        0.7 * selected_row["utilization"] + 0.2 * selected_row["traffic_level"] / 2 + 0.1 * selected_row["capacity_pressure"]
    ).clip(0, 1)

    simulation_prediction = xgb_pipeline.predict(selected_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]
    simulation_spaces = int(round(float(selected_row["capacity"].iloc[0] * (1 - min(max(simulation_prediction, 0), 1)))))
    sim_col1, sim_col2 = st.columns(2)
    sim_col1.metric("Predicted utilization in 1 hour", f"{simulation_prediction:.2%}")
    sim_col2.metric("Predicted free spaces in 1 hour", simulation_spaces)

    st.subheader("Latest forecast snapshot")
    latest_view = latest_df.copy()
    for column in ["utilization", "predicted_utilization_1h", "prediction_interval_lower", "prediction_interval_upper"]:
        latest_view[column] = latest_view[column].map(lambda value: f"{value:.2%}")
    st.dataframe(latest_view, width="stretch")


elif section == "Recommendations":
    recommendations_df = load_recommendation_bundle()
    ranked_recommendations = rank_recommendations(recommendations_df, recommendation_objective)
    top_choice = ranked_recommendations.iloc[0]

    rec_col1, rec_col2, rec_col3 = st.columns(3)
    rec_col1.metric("Best option", top_choice["system_code"])
    rec_col2.metric("Predicted free spaces", int(top_choice["predicted_available_spaces_1h"]))
    rec_col3.metric("Risk band", top_choice["risk_band"])

    display_columns = [
        "objective_rank",
        "system_code",
        "predicted_available_spaces_1h",
        "predicted_utilization_1h",
        "prediction_interval_upper",
        "risk_band",
        "balanced_score",
        "max_availability_score",
        "low_risk_score",
        "low_congestion_score",
    ]
    st.dataframe(ranked_recommendations[display_columns], width="stretch")

    if render_map_charts:
        recommendation_map = px.scatter_map(
            ranked_recommendations,
            lat="latitude",
            lon="longitude",
            color="risk_band",
            size="predicted_available_spaces_1h",
            hover_name="system_code",
            hover_data={"objective_rank": True, "predicted_utilization_1h": ":.2f"},
            zoom=11,
            height=520,
        )
        st.plotly_chart(recommendation_map, width="stretch")
    else:
        location_chart = px.scatter(
            ranked_recommendations,
            x="longitude",
            y="latitude",
            color="risk_band",
            size="predicted_available_spaces_1h",
            hover_name="system_code",
            title="Parking options by location",
        )
        st.plotly_chart(location_chart, width="stretch")


elif section == "Risk & Anomalies":
    data = load_risk_bundle()
    anomalies_df = data["anomalies"]

    anomaly_count = int(anomalies_df["anomaly_flag"].sum()) if "anomaly_flag" in anomalies_df.columns else len(anomalies_df)
    st.metric("Detected anomalies", anomaly_count)

    anomaly_chart = px.scatter(
        anomalies_df.head(250),
        x="queue_length",
        y="utilization",
        color="anomaly_reason",
        size="anomaly_score",
        hover_name="system_code",
        title="Top detected parking anomalies",
    )
    st.plotly_chart(anomaly_chart, width="stretch")
    st.dataframe(
        anomalies_df[["system_code", "time_slot", "utilization", "queue_length", "anomaly_score", "anomaly_reason"]].head(50),
        width="stretch",
    )


elif section == "Spatial Intelligence":
    data = load_risk_bundle()
    profiles_df = data["profiles"]
    clean_df = data["clean"]

    cluster_chart = px.scatter(
        profiles_df,
        x="pca_1",
        y="pca_2",
        color="cluster",
        size="mean_utilization",
        hover_name="system_code",
        title="Clustered parking profiles",
    )
    st.plotly_chart(cluster_chart, width="stretch")

    if render_map_charts:
        location_scatter = px.scatter_map(
            profiles_df,
            lat="latitude",
            lon="longitude",
            color="cluster",
            size="mean_utilization",
            hover_name="system_code",
            zoom=11,
            height=520,
            title="Spatial parking layout",
        )
        st.plotly_chart(location_scatter, width="stretch")
    else:
        location_scatter = px.scatter(
            profiles_df,
            x="longitude",
            y="latitude",
            color="cluster",
            size="mean_utilization",
            hover_name="system_code",
            title="Spatial parking layout",
        )
        st.plotly_chart(location_scatter, width="stretch")

    current_system_df = clean_df[clean_df["system_code"] == selected_system].sort_values("time_slot")
    trend_chart = px.line(
        current_system_df,
        x="time_slot",
        y=["utilization", "parking_stress_index"],
        title=f"Temporal demand profile for {selected_system}",
    )
    st.plotly_chart(trend_chart, width="stretch")


elif section == "SQL Snapshot":
    st.subheader("SQLite Project Database")
    st.code(f"SQLite database path: {SQLITE_DB_PATH}")
    st.markdown(
        """
        Example advanced queries:

        ```sql
        SELECT * FROM vw_daily_system_summary LIMIT 10;
        SELECT * FROM vw_peak_periods LIMIT 10;
        SELECT * FROM vw_recommended_parking LIMIT 10;
        SELECT * FROM vw_multi_horizon_summary;
        ```
        """
    )
