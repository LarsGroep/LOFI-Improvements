"""
Phase B roster — the listeners ("ears on the scene").

Each listener is a thin, pure-Python adapter with the same shape
(docs/agentic_os.md §3.1): fetch → normalise → emit `signal` records. They are
llm=False by construction, so the OS can watch thousands of artists nightly for
zero LLM cost, and they degrade gracefully — a missing API key or empty source
list yields "skipped: ...", never a crash.

All four write the same `signal` payload contract (see any module), so the
Phase C detectors can consume them uniformly.
"""
from __future__ import annotations

from osk.listeners.label_radar import LabelRadar
from osk.listeners.playlists import PlaylistListener
from osk.listeners.press import PressListener
from osk.listeners.soundcloud import SoundCloudListener

LISTENER_AGENTS = (
    SoundCloudListener,
    PlaylistListener,
    LabelRadar,
    PressListener,
)

__all__ = ["LISTENER_AGENTS", "SoundCloudListener", "PlaylistListener",
           "LabelRadar", "PressListener"]
