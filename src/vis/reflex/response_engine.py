"""Graduated, state-aware response engine (Phase 1).

A fixed, pre-approved escalation policy: log -> filter -> isolate -> minimal-risk
(goal G5). It is a pure function of (event, vehicle_state) so it is trivially
testable and certifiable-by-inspection, and it never chooses an unsafe action
for the current driving state (goal G8). It never *invents* a response.
"""
from __future__ import annotations

from ..shared.contracts import Detector, Event, Label, ResponseAction
from ..shared.state import VehicleState

# Highest-severity deterministic detectors warrant stronger containment.
_HIGH_SEVERITY = {Detector.AUTHENTICATION, Detector.PHYSICS, Detector.FINGERPRINT}


def decide(event: Event, state: VehicleState) -> ResponseAction:
    """Select the safest approved action for this event and driving state."""
    if event.label == Label.SUSPECTED:
        return ResponseAction.LOG            # unconfirmed -> observe only

    if event.detector == Detector.TRAFFIC:
        return ResponseAction.FILTER         # flooding/fuzzing -> rate-limit

    if event.detector in _HIGH_SEVERITY:
        # would isolate or stop; never force a stop unsafely at speed (G8)
        if state.can_safe_stop:
            return ResponseAction.MINIMAL_RISK
        return ResponseAction.ISOLATE

    return ResponseAction.LOG
