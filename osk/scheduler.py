"""
Schedule strings + due-ness — pure functions, fully testable.

Grammar (kept deliberately small; croniter can replace this later without
touching agents):

    "every:15m" / "every:6h"      interval since last run
    "hourly"                      once per clock hour  (slot = HH:00)
    "daily"                       once per day         (slot = 00:00)
    "daily@03:10"                 once per day at/after 03:10
    "weekly@mon 04:00"            once per week at/after Mon 04:00

An agent is due when the most recent slot at-or-before `now` is later than
its last run. Interval schedules are due when `now - last_run >= interval`.
"""
from __future__ import annotations

import datetime as _dt
import re

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4,
             "sat": 5, "sun": 6}
_EVERY = re.compile(r"^every:(\d+)([mh])$")
_DAILY = re.compile(r"^daily(?:@(\d{1,2}):(\d{2}))?$")
_WEEKLY = re.compile(r"^weekly(?:@([a-z]{3})(?: (\d{1,2}):(\d{2}))?)?$")


def validate_schedule(schedule: str) -> None:
    if schedule == "hourly":
        return
    m = _EVERY.match(schedule)
    if m:
        if int(m.group(1)) <= 0:
            raise ValueError(f"bad interval: {schedule!r}")
        return
    m = _DAILY.match(schedule)
    if m:
        if m.group(1) and not (int(m.group(1)) < 24 and int(m.group(2)) < 60):
            raise ValueError(f"bad time in: {schedule!r}")
        return
    m = _WEEKLY.match(schedule)
    if m:
        if m.group(1) and m.group(1) not in _WEEKDAYS:
            raise ValueError(f"bad weekday in: {schedule!r}")
        if m.group(2) and not (int(m.group(2)) < 24 and int(m.group(3)) < 60):
            raise ValueError(f"bad time in: {schedule!r}")
        return
    raise ValueError(f"unparseable schedule: {schedule!r}")


def interval_seconds(schedule: str) -> int | None:
    m = _EVERY.match(schedule)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * (60 if unit == "m" else 3600)


def last_slot(schedule: str, now: _dt.datetime) -> _dt.datetime | None:
    """The most recent scheduled moment at-or-before `now`.
    None for interval schedules (they have no anchored slots)."""
    if _EVERY.match(schedule):
        return None
    if schedule == "hourly":
        return now.replace(minute=0, second=0, microsecond=0)
    m = _DAILY.match(schedule)
    if m:
        hh, mm = int(m.group(1) or 0), int(m.group(2) or 0)
        slot = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return slot if slot <= now else slot - _dt.timedelta(days=1)
    m = _WEEKLY.match(schedule)
    if m:
        wd = _WEEKDAYS[m.group(1) or "mon"]
        hh, mm = int(m.group(2) or 0), int(m.group(3) or 0)
        slot = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        slot -= _dt.timedelta(days=(now.weekday() - wd) % 7)
        return slot if slot <= now else slot - _dt.timedelta(days=7)
    raise ValueError(f"unparseable schedule: {schedule!r}")


def is_due(schedule: str | None, last_run: _dt.datetime | None,
           now: _dt.datetime) -> bool:
    """Event-only agents (schedule=None) are never time-due."""
    if schedule is None:
        return False
    secs = interval_seconds(schedule)
    if secs is not None:
        return last_run is None or (now - last_run).total_seconds() >= secs
    slot = last_slot(schedule, now)
    return slot is not None and (last_run is None or last_run < slot)
