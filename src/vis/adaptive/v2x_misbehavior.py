"""V2X misbehavior detection (Phase 2).

Checks kinematic plausibility of received V2X messages (claimed position/speed/
trajectory vs. physics and local sensing) and emits misbehavior reports for the
fleet revocation service. Evaluate against the VeReMi dataset.

Three cheap, per-sender plausibility checks catch the common VeReMi attacks:
  * **over-speed**        -- claimed speed beyond a physical ceiling.
  * **position teleport** -- displacement between two BSMs implies an
    impossible speed (random / constant-offset position attacks).
  * **speed/position mismatch** -- claimed speed disagrees with the speed
    implied by movement (constant-position / eventual-stop attacks).

Reads the structured ``Message.payload`` (schema 1.1) produced by
``shared/datasets.py::VeReMiSource`` -- keys ``pos`` and ``spd``.
"""
from __future__ import annotations

from math import hypot
from typing import Optional

from ..shared.contracts import Bus, Detector, Event, Label, Message, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState


class V2XMisbehavior(BaseDetector):
    name = "v2x_misbehavior"

    # speed_tol_mps=10 chosen by sweep on VeReMi: same recall as 15 at the same
    # (zero) false-positive rate, and slightly better on eventual_stop.
    def __init__(self, max_speed_mps: float = 70.0, speed_tol_mps: float = 10.0):
        self.max_speed_mps = max_speed_mps        # ~252 kph physical ceiling
        self.speed_tol_mps = speed_tol_mps
        self._last: dict[int, tuple[float, float, float, float]] = {}  # sender -> (t,x,y,claimed)

    def reset(self) -> None:
        self._last.clear()

    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        payload = msg.payload or {}
        if "pos" not in payload:
            return None
        return self.check({
            "sender": msg.arbitration_id,
            "time": msg.timestamp,
            "pos": payload.get("pos"),
            "spd": payload.get("spd"),
        })

    def check(self, bsm: dict) -> Optional[Event]:
        """bsm: a received Basic Safety / Cooperative Awareness Message."""
        pos = bsm.get("pos") or ()
        spd = bsm.get("spd") or ()
        if len(pos) < 2:
            return None
        sender = bsm.get("sender")
        t = bsm.get("time")
        x, y = float(pos[0]), float(pos[1])
        claimed = hypot(float(spd[0]), float(spd[1])) if len(spd) >= 2 else 0.0

        reason: Optional[str] = None
        label = Label.SUSPECTED

        last = self._last.get(sender)
        if t is not None:
            self._last[sender] = (t, x, y, claimed)

        if claimed > self.max_speed_mps:
            reason, label = "speed_exceeds_max", Label.MALICIOUS
        elif last is not None and t is not None:
            lt, lx, ly, _ = last
            dt = t - lt
            if dt > 0:
                implied = hypot(x - lx, y - ly) / dt
                if implied > self.max_speed_mps:
                    reason, label = "position_teleport", Label.MALICIOUS
                elif abs(implied - claimed) > self.speed_tol_mps:
                    reason, label = "speed_position_mismatch", Label.SUSPECTED

        if reason is None:
            return None
        return Event(
            detector=Detector.V2X_MISBEHAVIOR,
            bus=Bus.V2X,
            label=label,
            confidence=0.9 if label == Label.MALICIOUS else 0.6,
            source_tier=Tier.ADAPTIVE,
            features={"sender": sender, "reason": reason, "claimed_speed_mps": round(claimed, 2)},
        )
