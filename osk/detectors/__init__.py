"""
Hidden-gem detection engines (docs/agentic_os.md §5).

Pure-Python processes over the `signal` spine of the blackboard — this whole
package can watch the pool nightly for zero LLM cost. Each detector pairs a
pure detection function (the math, fully tested) with a thin agent shell that
reads signals, calls the math, and emits derived `signal` records the sentinel
weighs. Every agent degrades honestly: few/no inputs → "no data yet", a source
that isn't wired yet → "skipped: …", never a fabricated score.
"""
from __future__ import annotations

from osk.detectors.divergence import DivergenceDetector
from osk.detectors.feelag import FeeLagDetector
from osk.detectors.ignition import IgnitionDetector
from osk.detectors.ladder import VenueLadderDetector
from osk.detectors.scene import SceneCartographer

# Ordered by expected value, matching §5's ordering.
DETECTOR_AGENTS = (DivergenceDetector, IgnitionDetector, VenueLadderDetector,
                   FeeLagDetector, SceneCartographer)
