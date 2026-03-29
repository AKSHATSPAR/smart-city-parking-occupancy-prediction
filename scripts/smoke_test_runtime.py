from __future__ import annotations

import argparse
import json
import sys

import httpx


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def _check(client: httpx.Client, url: str, name: str, headers: dict[str, str]) -> dict:
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return {
        "name": name,
        "status_code": response.status_code,
        "request_id": response.headers.get("X-Request-ID"),
        "ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the smart parking API and dashboard.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Base URL for the FastAPI service.")
    parser.add_argument("--dashboard-url", default=None, help="Optional URL for the Streamlit dashboard root.")
    parser.add_argument("--api-key", default=None, help="Optional API key for protected API access.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds.")
    args = parser.parse_args()

    checks: list[dict] = []
    headers = _headers(args.api_key)
    try:
        with httpx.Client(timeout=args.timeout) as client:
            checks.append(_check(client, f"{args.api_base_url}/health", "health", headers={}))
            checks.append(_check(client, f"{args.api_base_url}/ready", "ready", headers={}))
            checks.append(_check(client, f"{args.api_base_url}/ops/summary", "ops_summary", headers=headers))
            checks.append(_check(client, f"{args.api_base_url}/ops/activity", "ops_activity", headers=headers))
            if args.dashboard_url:
                dashboard_response = client.get(args.dashboard_url, timeout=args.timeout)
                dashboard_response.raise_for_status()
                checks.append(
                    {
                        "name": "dashboard",
                        "status_code": dashboard_response.status_code,
                        "request_id": None,
                        "ok": True,
                    }
                )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "checks": checks,
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps({"ok": True, "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
