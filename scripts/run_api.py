from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("smart_parking.api:app", host="0.0.0.0", port=port, reload=False)
