from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_CHECK = ROOT / "artifacts" / "reports" / "model_registry.json"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    if not ARTIFACT_CHECK.exists():
        result = subprocess.run([sys.executable, "scripts/run_pipeline.py"], cwd=str(ROOT))
        if result.returncode != 0:
            return result.returncode
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("smart_parking.api:app", host="0.0.0.0", port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
