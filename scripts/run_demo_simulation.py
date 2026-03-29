from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_parking.service import get_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live demo scenarios for the smart parking platform.")
    parser.add_argument("--list", action="store_true", help="List available demo scenarios.")
    parser.add_argument("--list-playbooks", action="store_true", help="List available guided demo playbooks.")
    parser.add_argument("--scenario", type=str, default="rush_hour_surge", help="Scenario name to run.")
    parser.add_argument("--playbook", type=str, default=None, help="Guided playbook name to run.")
    parser.add_argument("--steps", type=int, default=3, help="Number of scenario steps to execute.")
    parser.add_argument(
        "--keep-live-state",
        action="store_true",
        help="Do not reset the live state before running the scenario.",
    )
    parser.add_argument(
        "--clear-jobs",
        action="store_true",
        help="Clear retraining jobs when resetting without running a scenario.",
    )
    parser.add_argument("--reset-only", action="store_true", help="Reset live state and exit.")
    return parser


def print_scenarios() -> None:
    service = get_service()
    for item in service.demo_scenarios():
        print(f"{item['name']}: {item['title']} ({item['recommended_steps']} steps)")
        print(f"  {item['description']}")


def print_playbooks() -> None:
    service = get_service()
    for item in service.demo_playbooks():
        print(f"{item['name']}: {item['title']} ({item['stage_count']} stages)")
        print(f"  {item['description']}")
        for stage in item["stages"]:
            print(f"  - {stage['scenario_name']} x{stage['steps']}: {stage['narrative']}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    service = get_service()

    if args.list:
        print_scenarios()
        return 0

    if args.list_playbooks:
        print_playbooks()
        return 0

    if args.reset_only:
        result = service.reset_live_state(clear_jobs=args.clear_jobs)
        print(json.dumps(result, indent=2))
        return 0

    if args.playbook:
        result = service.run_demo_playbook(
            playbook_name=args.playbook,
            reset_first=not args.keep_live_state,
        )
        print(json.dumps(result, indent=2))
        return 0

    result = service.run_demo_scenario(
        scenario_name=args.scenario,
        steps=args.steps,
        reset_first=not args.keep_live_state,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
