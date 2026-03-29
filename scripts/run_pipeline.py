from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_parking.pipeline import run_pipeline


if __name__ == "__main__":
    outputs = run_pipeline()
    print(json.dumps(outputs, indent=2))

