"""Correlation & fusion (Phase 2).

Combines reflex Events + anomaly scores + decoy hits + V2X reports into a
CONFIRMED incident. Multi-signal agreement promotes corroborated signals and
suppresses isolated anomalies -> keeps false alarms low.

Confirmation policy (per arbitration id, within `window_s`):
  * any high-confidence MALICIOUS event (decoy / deterministic reflex check)
    confirms on its own -- those detectors don't guess; OR
  * >= `min_events` corroborating events agree on the same id (e.g. two
    SUSPECTED anomalies, or a SUSPECTED anomaly beside a timing event).
A lone SUSPECTED anomaly never confirms -- that is the false-alarm suppression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..shared.contracts import Event, Label


@dataclass
class Incident:
    """A confirmed attack assembled from several corroborating events."""
    attack_class: str
    events: list[Event] = field(default_factory=list)
    confidence: float = 1.0


class CorrelationEngine:
    name = "correlation_engine"

    def __init__(self, window_s: float = 0.05, min_events: int = 2):
        self.window_s = window_s
        self.min_events = min_events
        self._recent: list[Event] = []

    def reset(self) -> None:
        self._recent.clear()

    def submit(self, event: Event) -> Optional[Incident]:
        """Feed an event; return an Incident if corroboration confirms an attack."""
        now = event.timestamp
        # slide the window
        self._recent = [e for e in self._recent if now - e.timestamp <= self.window_s]
        self._recent.append(event)

        aid = event.features.get("arbitration_id")
        group = [e for e in self._recent if e.features.get("arbitration_id") == aid]
        has_malicious = any(e.label == Label.MALICIOUS for e in group)
        if not (has_malicious or len(group) >= self.min_events):
            return None

        # consume this id's events so we don't re-confirm on every later frame
        self._recent = [e for e in self._recent if e.features.get("arbitration_id") != aid]
        confidence = min(1.0, sum(1.0 if e.label == Label.MALICIOUS else 0.5 for e in group))
        return Incident(
            attack_class=self._attack_class(group, aid),
            events=group,
            confidence=confidence,
        )

    @staticmethod
    def _attack_class(group: list[Event], aid: Optional[int]) -> str:
        for e in group:
            named = e.features.get("attack_class") or e.features.get("attack_type")
            if named:
                return str(named)
        return f"id_{aid:#x}" if isinstance(aid, int) else "unknown"
