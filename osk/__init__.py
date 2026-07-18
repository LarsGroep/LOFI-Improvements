"""
osk — the LOFI Agentic OS kernel (docs/agentic_os.md, Phase A).

Pure Python, stdlib-only at the core. Agents are registered processes with a
manifest; they communicate through a typed blackboard (SQLite by default,
Supabase `agentic` schema when configured); the orchestrator runs whatever is
due and audits every run. No LLM calls happen anywhere in this package —
reasoning agents arrive in Phase D and will route through agents/core.py.

Deploy:  python -m osk loop        (or see deploy/ for docker-compose/systemd)
Inspect: python -m osk status | feed
"""
from osk.manifest import AgentManifest, Budget  # noqa: F401
from osk.registry import Registry               # noqa: F401

__version__ = "0.1.0"
