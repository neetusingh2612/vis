"""Server-side validation (Phase 3) -- the final gate before distribution.

Tests an aggregated candidate antibody against a large held-out normal corpus
(no false alarms), a known-attack corpus (it works), and across vehicle variants
(safe everywhere). Nothing that fails is ever distributed.

Sets ``candidate.validation['server_side_passed']`` -- one of the two conditions
(the other being a signature, added by the OTA distributor) that make an
Antibody ``is_validated`` and thus eligible to act.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..adaptive.antibody_generator import matcher_for_artifact
from ..shared.contracts import Antibody, Message


class ValidationLab:
    def __init__(self, normal_corpus: Optional[Iterable[Message]] = None,
                 attack_corpus: Optional[Iterable[Message]] = None,
                 max_fpr: float = 0.0, min_detection: float = 0.5):
        self.normal = list(normal_corpus or [])
        self.attack = list(attack_corpus or [])
        self.max_fpr = max_fpr
        self.min_detection = min_detection

    def validate(self, candidate: Antibody) -> Antibody:
        """Run the validation suite; set candidate.validation['server_side_passed']."""
        try:
            match = matcher_for_artifact(candidate.artifact)
        except (ValueError, KeyError):
            candidate.validation = {"server_side_passed": False, "reason": "uncompilable_artifact"}
            return candidate

        n_norm, n_att = len(self.normal), len(self.attack)
        fp = sum(1 for m in self.normal if match(m))
        tp = sum(1 for m in self.attack if match(m))
        fpr = fp / n_norm if n_norm else 0.0
        detection = tp / n_att if n_att else 0.0
        passed = fpr <= self.max_fpr and detection >= self.min_detection

        candidate.validation = {
            "server_side_passed": passed,
            "fpr": round(fpr, 4),
            "detection": round(detection, 4),
            "n_normal": n_norm,
            "n_attack": n_att,
        }
        return candidate
