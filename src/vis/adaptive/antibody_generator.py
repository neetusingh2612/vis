"""Candidate-antibody generation (Phase 2).

Distils a CONFIRMED incident into a candidate Antibody (rule / threshold /
model delta), then screens it with negative selection. Surviving candidates go
to the shadow runner. See Figure 5 in the paper.

An Antibody's `artifact` is a small declarative dict; `matcher_for_artifact`
compiles it into a runnable ``(Message) -> bool`` so the same rule drives both
negative selection and the shadow runner.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from ..shared.contracts import Antibody, ArtifactType, Message
from .correlation_engine import Incident
from .negative_selection import NegativeSelection

Matcher = Callable[[Message], bool]


def matcher_for_artifact(artifact: dict) -> Matcher:
    """Compile a declarative artifact dict into a runnable matcher."""
    kind = artifact.get("kind")
    if kind == "id_match":
        target = artifact["arbitration_id"]
        return lambda m: m.arbitration_id == target
    raise ValueError(f"unknown artifact kind: {kind!r}")


class AntibodyGenerator:
    def __init__(self, neg_selection: NegativeSelection):
        self.neg = neg_selection

    def synthesize(self, incident: Incident) -> Optional[Antibody]:
        """incident -> candidate Antibody (or None if it fails negative selection)."""
        aid = self._dominant_id(incident)
        if aid is None:
            return None

        artifact = {"kind": "id_match", "arbitration_id": aid}
        # self-tolerance: reject a rule that would fire on known-good traffic
        # (e.g. a masquerade on a legitimate id -> matching that id is unsafe).
        if not self.neg.passes(matcher_for_artifact(artifact)):
            return None

        return Antibody(
            attack_class=incident.attack_class,
            artifact_type=ArtifactType.RULE,
            artifact=artifact,
            provenance={
                "source": "antibody_generator",
                "event_ids": [e.event_id for e in incident.events],
                "incident_confidence": incident.confidence,
            },
        )

    @staticmethod
    def _dominant_id(incident: Incident) -> Optional[int]:
        ids = [e.features.get("arbitration_id") for e in incident.events
               if isinstance(e.features.get("arbitration_id"), int)]
        if not ids:
            return None
        return Counter(ids).most_common(1)[0][0]
