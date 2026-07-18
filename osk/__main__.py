"""
osk CLI — operate the Agentic OS.

    python -m osk status            # agents, schedules, last runs, backend
    python -m osk once              # one tick (run whatever is due), then exit
    python -m osk run <agent>       # force one agent now (ignores schedule)
    python -m osk loop              # run forever (the VPS entrypoint)
    python -m osk feed [-n 20] [--kind alert]   # the blackboard, newest first
    python -m osk health            # exit 0 iff the blackboard is reachable
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:
        pass


def _open():
    from osk.blackboard import open_blackboard
    from osk.registry import default_registry
    return default_registry(), open_blackboard()


def _cmd_status() -> int:
    from osk import budget
    registry, bb = _open()
    print(f"backend : {bb.describe()}")
    print(f"paused  : {'YES (LOFI_OS_PAUSED)' if budget.paused() else 'no'}")
    print()
    for agent in registry.all():
        m = agent.manifest
        runs = bb.runs(m.name, limit=1)
        last = runs[0] if runs else None
        line = f"  {m.name:<20} {m.schedule or 'event-only':<18}"
        if last:
            line += (f" last {str(last.get('started_at'))[:16]} "
                     f"→ {last.get('outcome') or 'running'}")
            if last.get("error"):
                line += f" ({last['error'][:60]})"
        else:
            line += " never ran"
        print(line)
    bb.close()
    return 0


def _print_results(results) -> int:
    if not results:
        print("nothing due (or paused).")
    for r in results:
        line = f"[{r['agent']}] {r['outcome']}"
        if r.get("error"):
            line += f" — {r['error']}"
        print(line)
    return 1 if any(r["outcome"] == "error" for r in results) else 0


def _cmd_once() -> int:
    from osk.orchestrator import tick
    registry, bb = _open()
    code = _print_results(tick(registry, bb))
    bb.close()
    return code


def _cmd_run(name: str) -> int:
    from osk.orchestrator import tick
    registry, bb = _open()
    if name not in registry.names():
        print(f"unknown agent {name!r} — registered: "
              f"{', '.join(registry.names())}")
        return 2
    code = _print_results(tick(registry, bb, force=name))
    bb.close()
    return code


def _cmd_loop() -> int:
    from osk.blackboard import open_blackboard
    from osk.orchestrator import loop
    from osk.registry import default_registry
    loop(default_registry(), open_blackboard())
    return 0


def _cmd_feed(limit: int, kind: str | None) -> int:
    from osk.blackboard import open_blackboard
    bb = open_blackboard()
    records = bb.read(kinds=[kind] if kind else None, limit=limit)
    if not records:
        print("blackboard is empty.")
    for r in records:
        who = f" {r.artist_name}" if r.artist_name else ""
        print(f"{r.created_at}  [{r.kind}]{who}  by {r.emitted_by}")
        print(f"    {r.payload}")
    bb.close()
    return 0


def _cmd_health() -> int:
    try:
        from osk.blackboard import open_blackboard
        bb = open_blackboard()
        bb.read(limit=1)
        print(f"ok — {bb.describe()}")
        bb.close()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"unhealthy: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    _load_env()
    p = argparse.ArgumentParser(prog="osk",
                                description="LOFI Agentic OS kernel")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("once")
    sub.add_parser("loop")
    sub.add_parser("health")
    run_p = sub.add_parser("run")
    run_p.add_argument("agent")
    feed_p = sub.add_parser("feed")
    feed_p.add_argument("-n", "--limit", type=int, default=20)
    feed_p.add_argument("--kind", default=None)
    args = p.parse_args(argv)

    if args.cmd == "status":
        return _cmd_status()
    if args.cmd == "once":
        return _cmd_once()
    if args.cmd == "run":
        return _cmd_run(args.agent)
    if args.cmd == "loop":
        return _cmd_loop()
    if args.cmd == "feed":
        return _cmd_feed(args.limit, args.kind)
    if args.cmd == "health":
        return _cmd_health()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
