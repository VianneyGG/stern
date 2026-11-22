"""stern/watcher/class_hours.py

No cluster job may run during class hours: Monday-Friday 08:00-20:00,
Europe/Paris. The SallesInfo nodes are classroom machines; the admins cancel
anything running then and warn the account holder (2026-09-21).

A job is refused when ANY part of [now, now + walltime] overlaps class hours,
not just when it starts inside them -- a job started at 07:30 would still be
running at 08:00. With the 2h SallesInfo MaxTime the weekday launch window is
therefore 20:00-06:00; weekends are unrestricted.

Deliberately no override: this is a cluster rule, not a preference.

stdlib only and Python 3.9-compatible, because it is called as a plain script
(`python3 class_hours.py`) from job scripts and launchers on nodes whose system
python is 3.9, before any venv exists. Exit 0 = allowed, 1 = blocked.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
CLASS_START = time(8, 0)
CLASS_END = time(20, 0)
# SallesInfo MaxTime (mesh_train.sbatch). Callers that don't know a job's
# walltime assume the longest one the partition allows.
MAX_WALLTIME_S = 2 * 3600


def blocked(now: datetime | float, walltime_s: float = MAX_WALLTIME_S) -> str | None:
    """Why a job spanning [now, now + walltime_s] may not run, or None if it may."""
    if isinstance(now, (int, float)):
        now = datetime.fromtimestamp(now, PARIS)
    start = now.astimezone(PARIS)
    # Add in UTC: aware-datetime + timedelta is wall-clock arithmetic, which is
    # an hour off across a DST switch.
    end = (start.astimezone(timezone.utc) + timedelta(seconds=walltime_s)).astimezone(
        PARIS
    )
    day = start.date()
    while day <= end.date():
        if day.weekday() < 5:
            win_start = datetime.combine(day, CLASS_START, PARIS)
            win_end = datetime.combine(day, CLASS_END, PARIS)
            # `start >= win_start` too: a job starting AT 08:00 overlaps even
            # with zero walltime, which the half-open comparison alone misses.
            if start < win_end and (end > win_start or start >= win_start):
                return (
                    f"class hours: a job from {start:%a %H:%M} running "
                    f"{walltime_s / 3600:.2f}h would overlap {day:%a} "
                    f"{CLASS_START:%H:%M}-{CLASS_END:%H:%M} Europe/Paris "
                    "(no job may run Mon-Fri 08:00-20:00)"
                )
        day += timedelta(days=1)
    return None


def parse_walltime(spec: str) -> float:
    """SLURM time strings ([D-]HH:MM:SS, MM:SS, minutes) -> seconds.

    Anything unparseable (squeue's UNLIMITED / NOT_SET, an empty string)
    falls back to MAX_WALLTIME_S: the gate must fail closed, not open.
    """
    spec = spec.strip()
    try:
        days, _, rest = spec.rpartition("-")
        parts = [int(p) for p in rest.split(":")]
        if len(parts) == 1:
            secs = parts[0] * 60
        elif len(parts) == 2:
            secs = parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            return MAX_WALLTIME_S
        return int(days or 0) * 86400 + secs
    except ValueError:
        return MAX_WALLTIME_S


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument(
        "--walltime",
        default="",
        help="SLURM time string for the job; default is the partition max (2h)",
    )
    args = ap.parse_args(argv)
    reason = blocked(datetime.now(PARIS), parse_walltime(args.walltime))
    if reason:
        print(f"REFUSED -- {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
