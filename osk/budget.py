"""
Budgets + the kill switch. Breaching agents are parked, not retried.

LOFI_OS_PAUSED=1 stops the whole OS at the next tick — the loop keeps
running (so unpausing needs no redeploy) but nothing executes.
"""
from __future__ import annotations

import datetime as _dt
import os

from osk.manifest import AgentManifest


def paused() -> bool:
    return os.environ.get("LOFI_OS_PAUSED", "").strip().lower() in (
        "1", "true", "yes", "on")


def blocked_reason(manifest: AgentManifest, blackboard,
                   now: _dt.datetime) -> str | None:
    """None = clear to run; otherwise a human-readable reason for the log."""
    day_start = now.replace(hour=0, minute=0, second=0,
                            microsecond=0).isoformat(timespec="seconds")
    runs = blackboard.runs_since(manifest.name, day_start)
    if runs >= manifest.budget.max_runs_per_day:
        return (f"daily run budget spent ({runs}/"
                f"{manifest.budget.max_runs_per_day})")
    return None
