from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIFT_REPORT = ROOT / "artifacts" / "reports" / "drift_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain the pipeline only when drift crosses a threshold.")
    parser.add_argument("--threshold", type=float, default=0.1, help="PSI threshold for retraining")
    parser.add_argument("--force", action="store_true", help="Retrain regardless of drift")
    args = parser.parse_args()

    if not DRIFT_REPORT.exists():
        print("Drift report missing. Running full pipeline.")
        return subprocess.call([sys.executable, "scripts/run_pipeline.py"], cwd=str(ROOT))

    drift = json.loads(DRIFT_REPORT.read_text())
    should_retrain = args.force or any(item["psi"] >= args.threshold for item in drift.get("features", []))

    if not should_retrain:
        print("No drift threshold breach detected. Skipping retrain.")
        return 0

    print("Drift threshold breached. Retraining pipeline.")
    return subprocess.call([sys.executable, "scripts/run_pipeline.py"], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
