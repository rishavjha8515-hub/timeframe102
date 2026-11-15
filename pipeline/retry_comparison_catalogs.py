"""Retry the IRSA TAP comparison-catalog probe with bounded backoff.

This script is intended for the user's network-connected machine. It runs the
existing catalog-only probe (never the frame retry step), preserving each
attempt's JSON report. The default schedule starts immediately, then waits 15,
20, 20, ... minutes, stopping after two hours or on success.

Run from the repository root:
    python -m pipeline.retry_comparison_catalogs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "pipeline/cache/control-star/comparison_catalogs/comparison_catalog_vetting.json"
ATTEMPTS_DIR = ROOT / "pipeline/cache/control-star/comparison_catalogs/tap_attempts"


def retry_delays(duration_minutes: int, initial_minutes: int = 15,
                 multiplier: float = 1.5, maximum_minutes: int = 20) -> list[int]:
    """Return bounded exponential delays whose attempt times fit the window."""
    if duration_minutes < 1 or initial_minutes < 1 or maximum_minutes < initial_minutes or multiplier < 1:
        raise ValueError("invalid retry schedule parameters")
    delays: list[int] = []
    elapsed = 0
    delay = initial_minutes
    while elapsed + delay < duration_minutes:
        delays.append(delay)
        elapsed += delay
        delay = min(maximum_minutes, max(initial_minutes, int(round(delay * multiplier))))
    return delays


def run_retries(*, duration_minutes: int = 120, sleep: Callable[[float], None] = time.sleep,
                run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[dict]:
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    delays = retry_delays(duration_minutes)
    attempts: list[dict] = []
    for index in range(len(delays) + 1):
        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        process = run([sys.executable, "-m", "pipeline.query_comparison_catalogs"],
                      cwd=ROOT, capture_output=True, text=True, check=False)
        report = {}
        if REPORT.is_file():
            try:
                report = json.loads(REPORT.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {"status": "failed", "error": "catalog probe produced invalid JSON"}
        record = {
            "attempt": index + 1,
            "started_utc": started,
            "status": report.get("status", "failed"),
            "error": report.get("error"),
            "return_code": process.returncode,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:],
            "report": report,
        }
        path = ATTEMPTS_DIR / f"tap_attempt_{index + 1:02d}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        attempts.append(record)
        print(json.dumps({"attempt": index + 1, "status": record["status"],
                          "error": record["error"], "report_path": path.relative_to(ROOT).as_posix()},
                         separators=(",", ":")), flush=True)
        if record["status"] == "complete":
            break
        if index < len(delays):
            sleep(delays[index] * 60)
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-minutes", type=int, default=120)
    args = parser.parse_args()
    attempts = run_retries(duration_minutes=args.duration_minutes)
    return 0 if attempts and attempts[-1]["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
