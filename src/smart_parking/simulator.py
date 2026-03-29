from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    title: str
    description: str
    recommended_steps: int


@dataclass(frozen=True)
class PlaybookStage:
    scenario_name: str
    steps: int
    narrative: str


@dataclass(frozen=True)
class PlaybookDefinition:
    name: str
    title: str
    description: str
    stages: tuple[PlaybookStage, ...]


class DemoScenarioEngine:
    def __init__(self, recommendations_df: pd.DataFrame, neighbor_graph: pd.DataFrame) -> None:
        self.recommendations_df = recommendations_df.copy()
        self.neighbor_graph = neighbor_graph.copy()
        self.definitions = {
            "rush_hour_surge": ScenarioDefinition(
                name="rush_hour_surge",
                title="Weekday Rush-Hour Surge",
                description="Pushes already-busy parking systems toward peak load with rising queues and heavy traffic.",
                recommended_steps=3,
            ),
            "event_overflow": ScenarioDefinition(
                name="event_overflow",
                title="Special Event Overflow",
                description="Simulates a city event that rapidly fills large-capacity parking systems and increases operational risk.",
                recommended_steps=3,
            ),
            "spillover_disruption": ScenarioDefinition(
                name="spillover_disruption",
                title="Localized Spillover Disruption",
                description="Overloads one critical parking system and propagates pressure to its nearest neighboring systems.",
                recommended_steps=2,
            ),
            "holiday_relief": ScenarioDefinition(
                name="holiday_relief",
                title="Holiday Demand Relief",
                description="Reduces parking occupancy, queue lengths, and nearby traffic to show system recovery behavior.",
                recommended_steps=3,
            ),
        }
        self.playbooks = {
            "executive_showcase": PlaybookDefinition(
                name="executive_showcase",
                title="Executive Showcase",
                description="A polished demonstration flow that starts from a stable city state, applies rush-hour stress, triggers a localized disruption, and then shows recovery.",
                stages=(
                    PlaybookStage(
                        scenario_name="rush_hour_surge",
                        steps=2,
                        narrative="Demand rises as the city enters a rush-hour period and busy parking systems start filling rapidly.",
                    ),
                    PlaybookStage(
                        scenario_name="spillover_disruption",
                        steps=2,
                        narrative="One critical parking system saturates and nearby systems absorb spillover traffic.",
                    ),
                    PlaybookStage(
                        scenario_name="holiday_relief",
                        steps=2,
                        narrative="Traffic pressure eases and the recommendation engine redirects drivers toward lower-risk locations.",
                    ),
                ),
            ),
            "event_response_story": PlaybookDefinition(
                name="event_response_story",
                title="Event Response Story",
                description="Demonstrates how the platform reacts to special-event overflow and then recovers after operational intervention.",
                stages=(
                    PlaybookStage(
                        scenario_name="event_overflow",
                        steps=2,
                        narrative="A city event increases demand at large-capacity lots and pushes more systems into high-risk territory.",
                    ),
                    PlaybookStage(
                        scenario_name="spillover_disruption",
                        steps=1,
                        narrative="Overflow pressure spreads into neighboring systems, making rerouting quality more important.",
                    ),
                    PlaybookStage(
                        scenario_name="holiday_relief",
                        steps=3,
                        narrative="Demand relaxes and the live board shows clear improvement in congestion and recommendation quality.",
                    ),
                ),
            ),
        }

    def list_scenarios(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "title": definition.title,
                "description": definition.description,
                "recommended_steps": definition.recommended_steps,
            }
            for definition in self.definitions.values()
        ]

    def run(self, service: Any, scenario_name: str, steps: int) -> dict[str, Any]:
        definition = self.definitions.get(scenario_name)
        if definition is None:
            allowed = ", ".join(sorted(self.definitions))
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {allowed}")

        step_summaries: list[dict[str, Any]] = []
        total_events = 0
        for step in range(1, steps + 1):
            payloads = self._build_payloads(service, scenario_name, step)
            applied = [service.ingest_payload(payload) for payload in payloads]
            total_events += len(applied)
            live_df = self._current_live_frame(service)
            step_summaries.append(
                {
                    "step": step,
                    "events_applied": applied,
                    "portfolio_mean_utilization": round(float(live_df["predicted_utilization_1h"].mean()), 6),
                    "critical_systems": int((live_df["risk_band"] == "Critical").sum()),
                    "top_recommendation": self._records(
                        live_df.sort_values("balanced_score", ascending=False).head(1),
                        [
                            "system_code",
                            "predicted_available_spaces_1h",
                            "predicted_utilization_1h",
                            "risk_band",
                            "recommendation_rank",
                        ],
                    )[0],
                    "most_congested_system": self._records(
                        live_df.sort_values("predicted_utilization_1h", ascending=False).head(1),
                        [
                            "system_code",
                            "predicted_available_spaces_1h",
                            "predicted_utilization_1h",
                            "risk_band",
                            "recommendation_rank",
                        ],
                    )[0],
                }
            )

        live_df = self._current_live_frame(service)
        risk_counts = {
            str(label): int(count)
            for label, count in live_df["risk_band"].fillna("Unknown").value_counts().sort_index().items()
        }
        return {
            "scenario_name": definition.name,
            "scenario_title": definition.title,
            "description": definition.description,
            "steps_executed": steps,
            "events_executed": total_events,
            "risk_counts": risk_counts,
            "step_summaries": step_summaries,
            "top_recommendations": service.recommendations(limit=5),
            "critical_watchlist": self._records(
                live_df.sort_values("predicted_utilization_1h", ascending=False).head(5),
                [
                    "system_code",
                    "predicted_available_spaces_1h",
                    "predicted_utilization_1h",
                    "prediction_interval_upper",
                    "risk_band",
                    "recommendation_rank",
                ],
            ),
        }

    def list_playbooks(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "title": definition.title,
                "description": definition.description,
                "stage_count": len(definition.stages),
                "stages": [
                    {
                        "scenario_name": stage.scenario_name,
                        "steps": stage.steps,
                        "narrative": stage.narrative,
                    }
                    for stage in definition.stages
                ],
            }
            for definition in self.playbooks.values()
        ]

    def run_playbook(self, service: Any, playbook_name: str, reset_first: bool = True) -> dict[str, Any]:
        definition = self.playbooks.get(playbook_name)
        if definition is None:
            allowed = ", ".join(sorted(self.playbooks))
            raise ValueError(f"Unknown playbook '{playbook_name}'. Available playbooks: {allowed}")

        if reset_first:
            service.reset_live_state(clear_jobs=False)

        baseline_df = self._current_live_frame(service)
        baseline_snapshot = self._snapshot_summary(baseline_df)
        stages_output: list[dict[str, Any]] = []
        total_events = 0
        current_df = baseline_df

        for index, stage in enumerate(definition.stages, start=1):
            before = self._snapshot_summary(current_df)
            scenario_result = self.run(service, scenario_name=stage.scenario_name, steps=stage.steps)
            current_df = self._current_live_frame(service)
            after = self._snapshot_summary(current_df)
            total_events += int(scenario_result["events_executed"])
            stages_output.append(
                {
                    "stage_number": index,
                    "scenario_name": stage.scenario_name,
                    "scenario_title": scenario_result["scenario_title"],
                    "steps": stage.steps,
                    "narrative": stage.narrative,
                    "events_executed": scenario_result["events_executed"],
                    "before": before,
                    "after": after,
                    "delta_mean_utilization": round(after["mean_utilization"] - before["mean_utilization"], 6),
                    "delta_critical_systems": int(after["critical_systems"] - before["critical_systems"]),
                    "top_recommendation_after": scenario_result["top_recommendations"][0] if scenario_result["top_recommendations"] else None,
                    "critical_watchlist_after": scenario_result["critical_watchlist"],
                    "scenario_result": scenario_result,
                }
            )

        final_df = self._current_live_frame(service)
        final_snapshot = self._snapshot_summary(final_df)
        return {
            "playbook_name": definition.name,
            "playbook_title": definition.title,
            "description": definition.description,
            "baseline_snapshot": baseline_snapshot,
            "final_snapshot": final_snapshot,
            "stages": stages_output,
            "total_events_executed": total_events,
            "top_recommendations": service.recommendations(limit=5),
            "critical_watchlist": self._records(
                final_df.sort_values("predicted_utilization_1h", ascending=False).head(5),
                [
                    "system_code",
                    "predicted_available_spaces_1h",
                    "predicted_utilization_1h",
                    "prediction_interval_upper",
                    "risk_band",
                    "recommendation_rank",
                ],
            ),
        }

    def _build_payloads(self, service: Any, scenario_name: str, step: int) -> list[dict[str, Any]]:
        current = self._current_live_frame(service)
        builders = {
            "rush_hour_surge": self._rush_hour_payloads,
            "event_overflow": self._event_overflow_payloads,
            "spillover_disruption": self._spillover_payloads,
            "holiday_relief": self._holiday_relief_payloads,
        }
        return builders[scenario_name](current, step)

    def _current_live_frame(self, service: Any) -> pd.DataFrame:
        live_df = service.store.live_forecasts()
        if live_df.empty:
            live_df = self.recommendations_df.copy()
        live_df = live_df.copy()
        if "recommendation_rank" not in live_df.columns:
            live_df = live_df.sort_values("balanced_score", ascending=False).reset_index(drop=True)
            live_df["recommendation_rank"] = live_df.index + 1
        return live_df

    def _snapshot_summary(self, live_df: pd.DataFrame) -> dict[str, Any]:
        if live_df.empty:
            return {
                "mean_utilization": 0.0,
                "critical_systems": 0,
                "high_risk_systems": 0,
                "top_recommendation": None,
                "most_congested_system": None,
            }

        ranked = live_df.sort_values("balanced_score", ascending=False).reset_index(drop=True).copy()
        ranked["recommendation_rank"] = ranked.index + 1
        congested = live_df.sort_values("predicted_utilization_1h", ascending=False).head(1)
        return {
            "mean_utilization": round(float(live_df["predicted_utilization_1h"].mean()), 6),
            "critical_systems": int((live_df["risk_band"] == "Critical").sum()),
            "high_risk_systems": int(live_df["risk_band"].isin(["Critical", "High"]).sum()),
            "top_recommendation": self._records(
                ranked.head(1),
                [
                    "system_code",
                    "predicted_available_spaces_1h",
                    "predicted_utilization_1h",
                    "risk_band",
                    "recommendation_rank",
                ],
            )[0],
            "most_congested_system": self._records(
                congested,
                [
                    "system_code",
                    "predicted_available_spaces_1h",
                    "predicted_utilization_1h",
                    "risk_band",
                    "recommendation_rank",
                ],
            )[0],
        }

    def _rush_hour_payloads(self, current: pd.DataFrame, step: int) -> list[dict[str, Any]]:
        targets = current.sort_values("predicted_utilization_1h", ascending=False).head(6)
        payloads = []
        for index, (_, row) in enumerate(targets.iterrows()):
            payloads.append(
                self._payload_from_row(
                    row,
                    occupancy_delta_ratio=0.05 + 0.015 * step + 0.005 * index,
                    queue_delta=2 + step + index // 2,
                    traffic="high" if step >= 2 else "average",
                    is_special_day=0,
                )
            )
        return payloads

    def _event_overflow_payloads(self, current: pd.DataFrame, step: int) -> list[dict[str, Any]]:
        big_capacity = current.sort_values("capacity", ascending=False).head(4)
        high_pressure = current.sort_values("predicted_utilization_1h", ascending=False).head(3)
        targets = pd.concat([big_capacity, high_pressure], ignore_index=True).drop_duplicates("system_code")
        payloads = []
        for index, (_, row) in enumerate(targets.iterrows()):
            payloads.append(
                self._payload_from_row(
                    row,
                    occupancy_delta_ratio=0.08 + 0.02 * step + 0.01 * (index % 2),
                    queue_delta=4 + step + index,
                    traffic="high",
                    is_special_day=1,
                )
            )
        return payloads

    def _spillover_payloads(self, current: pd.DataFrame, step: int) -> list[dict[str, Any]]:
        origin = current.sort_values("predicted_utilization_1h", ascending=False).head(1)
        if origin.empty:
            return []

        origin_row = origin.iloc[0]
        neighbor_rows = self.neighbor_graph[self.neighbor_graph["system_code"] == origin_row["system_code"]].copy()
        neighbor_rows = neighbor_rows.sort_values("weight", ascending=False).drop_duplicates("neighbor_code").head(3)
        payloads = [
            self._payload_from_row(
                origin_row,
                target_utilization=min(0.99, max(float(origin_row["utilization"]) + 0.12 + 0.02 * step, 0.93)),
                queue_delta=5 + step,
                traffic="high",
                is_special_day=0,
            )
        ]
        for _, neighbor in neighbor_rows.iterrows():
            neighbor_state = current[current["system_code"] == neighbor["neighbor_code"]]
            if neighbor_state.empty:
                continue
            payloads.append(
                self._payload_from_row(
                    neighbor_state.iloc[0],
                    occupancy_delta_ratio=0.04 + 0.015 * step,
                    queue_delta=2 + step,
                    traffic="high",
                    is_special_day=0,
                )
            )
        return payloads

    def _holiday_relief_payloads(self, current: pd.DataFrame, step: int) -> list[dict[str, Any]]:
        targets = current.sort_values("capacity", ascending=False).head(8)
        payloads = []
        for _, row in targets.iterrows():
            payloads.append(
                self._payload_from_row(
                    row,
                    occupancy_delta_ratio=-(0.06 + 0.015 * step),
                    queue_delta=-int(max(float(row.get("queue_length", 0)), 1)),
                    traffic="low",
                    is_special_day=1,
                )
            )
        return payloads

    def _payload_from_row(
        self,
        row: pd.Series,
        occupancy_delta_ratio: float | None = None,
        target_utilization: float | None = None,
        queue_delta: int = 0,
        traffic: str = "average",
        is_special_day: int = 0,
    ) -> dict[str, Any]:
        capacity = float(row["capacity"])
        current_occupancy = float(row.get("occupancy_clean", row.get("occupancy", 0.0)))
        if target_utilization is not None:
            occupancy = capacity * target_utilization
        else:
            occupancy = current_occupancy + capacity * float(occupancy_delta_ratio or 0.0)
        occupancy = max(0.0, min(capacity, occupancy))

        current_queue = int(round(float(row.get("queue_length", 0.0))))
        queue_length = max(0, min(30, current_queue + queue_delta))
        if traffic == "low":
            queue_length = min(queue_length, 2)

        return {
            "system_code": str(row["system_code"]),
            "occupancy": round(occupancy, 2),
            "queue_length": int(queue_length),
            "traffic_condition_nearby": traffic,
            "is_special_day": int(is_special_day),
            "vehicle_type": str(row.get("vehicle_type", "car")),
            "capacity": capacity,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

    @staticmethod
    def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        subset = frame[columns].copy()
        for column in subset.columns:
            if pd.api.types.is_datetime64_any_dtype(subset[column]):
                subset[column] = subset[column].astype(str)
        subset = subset.astype(object).where(pd.notnull(subset), None)
        return subset.to_dict(orient="records")
