"""
The agent registry — the kernel's process table.

An agent is any object with a `manifest: AgentManifest` attribute and a
`run(ctx) -> str` method returning a one-line outcome summary. Registration
validates the manifest; the orchestrator only ever runs registered agents.
"""
from __future__ import annotations

from osk.manifest import AgentManifest


class Registry:
    def __init__(self):
        self._agents: dict[str, object] = {}

    def register(self, agent) -> None:
        manifest = getattr(agent, "manifest", None)
        if not isinstance(manifest, AgentManifest):
            raise TypeError(f"{agent!r} has no AgentManifest")
        manifest.validate()
        if not callable(getattr(agent, "run", None)):
            raise TypeError(f"{manifest.name}: agent has no run(ctx) method")
        if manifest.name in self._agents:
            raise ValueError(f"duplicate agent name: {manifest.name}")
        self._agents[manifest.name] = agent

    def get(self, name: str):
        return self._agents[name]

    def all(self) -> list:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return sorted(self._agents)


def default_registry() -> Registry:
    """The shipped agent roster: Phase A maintenance agents + the Phase B
    listeners. Later phases append here as they land."""
    from osk.agents_builtin import BUILTIN_AGENTS
    from osk.listeners import LISTENER_AGENTS
    reg = Registry()
    for agent in (*BUILTIN_AGENTS, *LISTENER_AGENTS):
        reg.register(agent())
    return reg
