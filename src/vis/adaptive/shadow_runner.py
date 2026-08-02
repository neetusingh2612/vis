"""Shadow mode (Phase 2/3 bridge).

A surviving candidate runs DETECTION-ONLY: it logs what it *would* have flagged
but has NO authority to act. It gains authority only after the FLEET validates
it (the `activate` hook is called by fleet/ota_distributor on a signed update).
This is the 'learn fast, act slow' boundary.
"""
from __future__ import annotations

from ..shared.contracts import Antibody
from .antibody_generator import matcher_for_artifact


class ShadowRunner:
    def __init__(self) -> None:
        self._shadow: dict[str, Antibody] = {}    # antibody_id -> candidate
        self._active: dict[str, Antibody] = {}     # antibody_id -> activated
        self._hits: dict[str, int] = {}            # antibody_id -> would-be hit count

    def add_candidate(self, ab: Antibody) -> None:
        self._shadow[ab.antibody_id] = ab
        self._hits.setdefault(ab.antibody_id, 0)

    def observe(self, msg, state=None) -> None:
        """Run shadow candidates detection-only; record would-be hits. No action."""
        for ab_id, ab in self._shadow.items():
            try:
                if matcher_for_artifact(ab.artifact)(msg):
                    self._hits[ab_id] += 1
            except (ValueError, KeyError):
                continue   # un-runnable artifact: skip rather than crash the bus copy

    def shadow_report(self) -> dict[str, int]:
        """How many frames each shadow candidate *would* have flagged."""
        return {ab_id: self._hits.get(ab_id, 0) for ab_id in self._shadow}

    def is_active(self, antibody_id: str) -> bool:
        return antibody_id in self._active

    def activate(self, antibody_id: str) -> None:
        """Called by the fleet layer once an antibody is validated + signed."""
        ab = self._shadow.pop(antibody_id, None)
        if ab is not None and ab.is_validated:
            self._active[antibody_id] = ab
