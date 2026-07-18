"""
Agent manifests — the kernel refuses to run anything without one.

The manifest is the enforcement surface: what an agent may read/write on the
blackboard, whether it may reason (LLM egress), and how much it may spend.
Pure dataclasses + validation; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The blackboard's typed-record vocabulary (docs/agentic_os.md §2.2).
RECORD_KINDS = ("signal", "candidate", "brief", "argument", "verdict",
                "dossier_update", "alert")


@dataclass(frozen=True)
class Budget:
    """Spend limits the kernel enforces. Token fields are recorded from day
    one so Phase D reasoning agents inherit working accounting."""
    max_runs_per_day: int = 24
    max_llm_calls_per_run: int = 0     # 0 for pure-Python agents
    max_tokens_per_day: int = 0


@dataclass(frozen=True)
class AgentManifest:
    name: str
    description: str = ""
    reads: tuple[str, ...] = ()        # record kinds it may read
    writes: tuple[str, ...] = ()       # record kinds it may emit
    llm: bool = False                  # False → the kernel never hands it egress
    schedule: str | None = None        # see osk/scheduler.py; None = event-only
    budget: Budget = field(default_factory=Budget)

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid agent name: {self.name!r}")
        for kind in (*self.reads, *self.writes):
            if kind not in RECORD_KINDS:
                raise ValueError(
                    f"{self.name}: unknown record kind {kind!r} "
                    f"(allowed: {', '.join(RECORD_KINDS)})")
        if not self.llm and self.budget.max_llm_calls_per_run:
            raise ValueError(
                f"{self.name}: llm=False but budget allows LLM calls")
        if self.schedule is not None:
            from osk.scheduler import validate_schedule
            validate_schedule(self.schedule)
