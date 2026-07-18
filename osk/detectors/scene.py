"""
Scene-graph centrality — being pulled toward the centre before the metrics move
(§5.3, §3.2 cartographer).

An artist whose lineup-adjacency to LOFI-core names is rising — opening slots
for benchmark acts, signings to core labels, B2Bs — is being drawn into the
scene's centre, often before any streaming number reacts. The graph is plain
dicts (no networkx, per the dependency rule); this phase ships the two pure
functions so the math is tested and ready, and parks the agent until the
`ra_listener` starts emitting lineup edges.
"""
from __future__ import annotations

from osk.manifest import AgentManifest, Budget


# ── graph math (pure) ─────────────────────────────────────────────────────────

def build_graph(edges: list[tuple[str, str]]) -> dict:
    """Undirected weighted adjacency from co-occurrence edges (a shared lineup,
    a B2B). Weight = how many times two names appeared together."""
    g: dict = {}
    for edge in edges or []:
        a, b = edge
        if not a or not b or a == b:
            continue
        g.setdefault(a, {})
        g.setdefault(b, {})
        g[a][b] = g[a].get(b, 0) + 1
        g[b][a] = g[b].get(a, 0) + 1
    return g


def _centrality(graph: dict, core: set) -> dict:
    """Fraction of each node's weighted degree pointing at the core set — a
    normalised 'how central to the scene' score in [0, 1]."""
    out = {}
    for node, nbrs in graph.items():
        total = sum(nbrs.values())
        if total <= 0:
            out[node] = 0.0
            continue
        toward = sum(w for n, w in nbrs.items() if n in core)
        out[node] = toward / total
    return out


def centrality_delta(graph_then: dict, graph_now: dict,
                     core: set) -> dict:
    """Per-node change in core-directed centrality between two graph snapshots.
    Positive = being pulled toward the scene's centre."""
    then = _centrality(graph_then or {}, core or set())
    now = _centrality(graph_now or {}, core or set())
    nodes = set(then) | set(now)
    return {n: round(now.get(n, 0.0) - then.get(n, 0.0), 4) for n in nodes}


# ── agent shell ───────────────────────────────────────────────────────────────

class SceneCartographer:
    manifest = AgentManifest(
        name="scene_cartographer",
        description="lineup-graph centrality delta toward LOFI-core artists",
        reads=("signal",), writes=("signal",),
        schedule="daily@05:40", budget=Budget(max_runs_per_day=2))

    def run(self, ctx) -> str:
        # No lineup graph flows yet — the cartographer needs ra_listener edges.
        # The pure functions above get real wiring when lineups arrive.
        return "skipped: no lineup graph source yet (needs ra_listener)"
