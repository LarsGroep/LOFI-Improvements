"""
The orchestrator — the kernel's run loop.

tick():   run whatever is due (schedule) or forced (CLI), under budget,
          each run audited to agent_runs whatever happens.
loop():   tick forever; SIGTERM/SIGINT exit cleanly (docker stop / systemd).

Agents receive a Context: their manifest-scoped window onto the blackboard.
emit/read enforce the manifest's writes/reads lists in code — an agent
cannot exceed its declared capabilities by construction.
"""
from __future__ import annotations

import datetime as _dt
import os
import signal
import time
import traceback

from osk import budget as _budget
from osk.blackboard import open_blackboard
from osk.manifest import AgentManifest
from osk.registry import Registry
from osk.scheduler import is_due

DEFAULT_TICK_SECONDS = 60


class Context:
    """What an agent gets to touch during one run — nothing else."""

    def __init__(self, manifest: AgentManifest, blackboard, shared: dict,
                 now: _dt.datetime):
        self.manifest = manifest
        self.now = now
        self.shared = shared          # per-tick cache (e.g. loaded candidates)
        self._bb = blackboard
        self.records_emitted = 0
        # reasoning agents increment these; the audit row picks them up
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def emit(self, kind: str, payload: dict, artist_id: str | None = None,
             artist_name: str | None = None) -> int:
        if kind not in self.manifest.writes:
            raise PermissionError(
                f"{self.manifest.name} may not emit {kind!r} "
                f"(manifest writes: {self.manifest.writes})")
        self.records_emitted += 1
        return self._bb.emit(kind, payload, self.manifest.name,
                             artist_id=artist_id, artist_name=artist_name)

    def read(self, kinds: list[str], **kwargs):
        for kind in kinds:
            if kind not in self.manifest.reads:
                raise PermissionError(
                    f"{self.manifest.name} may not read {kind!r} "
                    f"(manifest reads: {self.manifest.reads})")
        return self._bb.read(kinds=kinds, **kwargs)

    # per-agent persistent state, namespaced so agents can't collide
    def state_get(self, key: str):
        return self._bb.state_get(f"{self.manifest.name}:{key}")

    def state_set(self, key: str, value) -> None:
        self._bb.state_set(f"{self.manifest.name}:{key}", value)

    @property
    def llm(self):
        """Reasoning agents only (Phase D). The single egress point stays
        agents/core.py; pure-Python agents can't reach it by construction."""
        if not self.manifest.llm:
            raise PermissionError(
                f"{self.manifest.name} is a pure-Python agent (llm=False)")
        import agents.core as core
        return core


def run_agent(agent, blackboard, shared: dict | None = None,
              now: _dt.datetime | None = None, trigger: str = "schedule") -> dict:
    """One audited run. Never raises — errors land in agent_runs."""
    manifest: AgentManifest = agent.manifest
    now = now or _dt.datetime.now(_dt.timezone.utc)
    run_id = blackboard.run_start(manifest.name, trigger)
    ctx = Context(manifest, blackboard, shared if shared is not None else {},
                  now)
    try:
        outcome = agent.run(ctx) or "ok"
        blackboard.run_finish(run_id, outcome,
                              llm_calls=ctx.llm_calls,
                              tokens_in=ctx.tokens_in,
                              tokens_out=ctx.tokens_out,
                              records_emitted=ctx.records_emitted)
        return {"agent": manifest.name, "outcome": outcome, "error": None}
    except Exception as exc:  # noqa: BLE001 — the OS must survive any agent
        err = "".join(traceback.format_exception_only(exc)).strip()
        blackboard.run_finish(run_id, "error", error=err,
                              llm_calls=ctx.llm_calls,
                              tokens_in=ctx.tokens_in,
                              tokens_out=ctx.tokens_out,
                              records_emitted=ctx.records_emitted)
        return {"agent": manifest.name, "outcome": "error", "error": err}


def tick(registry: Registry, blackboard,
         now: _dt.datetime | None = None,
         force: str | None = None) -> list[dict]:
    """Run every agent that is due (or the one named by `force`)."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    results: list[dict] = []
    if _budget.paused() and not force:
        return results
    shared: dict = {}
    for agent in registry.all():
        m: AgentManifest = agent.manifest
        if force:
            if m.name != force:
                continue
        elif not is_due(m.schedule, blackboard.last_run_at(m.name), now):
            continue
        reason = _budget.blocked_reason(m, blackboard, now)
        if reason and not force:
            results.append({"agent": m.name, "outcome": "parked",
                            "error": reason})
            continue
        results.append(run_agent(agent, blackboard, shared, now,
                                 trigger="cli" if force else "schedule"))
    return results


def loop(registry: Registry, blackboard=None,
         tick_seconds: int | None = None) -> None:
    blackboard = blackboard or open_blackboard()
    tick_seconds = tick_seconds or int(
        os.environ.get("LOFI_OS_TICK_SECONDS", DEFAULT_TICK_SECONDS))
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    print(f"osk loop — {blackboard.describe()} — agents: "
          f"{', '.join(a.manifest.name for a in registry.all())}", flush=True)
    while not stop["flag"]:
        for r in tick(registry, blackboard):
            line = f"[{r['agent']}] {r['outcome']}"
            if r.get("error"):
                line += f" — {r['error']}"
            print(line, flush=True)
        # sleep in 1s slices so SIGTERM lands promptly
        for _ in range(tick_seconds):
            if stop["flag"]:
                break
            time.sleep(1)
    blackboard.close()
    print("osk loop stopped.", flush=True)
