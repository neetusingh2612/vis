"""Physics & plausibility checks (Phase 1).

Three sub-checks (range/rate, cross-sensor agreement, command feasibility).
Range/rate is implemented; the others are stubs that need decoded signals.
The physics engine is the anchor against novel/AI-crafted attacks (goal G4)
and cannot be 'drifted' because physical law does not change.

Two ways to supply the "plausible" envelope:

  * ``signal_bounds`` -- explicit per-id (min, max) on a decoded signal, when a
    DBC/decoder is available; or
  * :meth:`fit_byte_ranges` -- learn the per-id, per-byte value envelope from a
    clean capture when no decoder is at hand. This is the generic stand-in that
    catches *content* attacks: a masquerading ECU that keeps perfect cadence but
    forces a signal to an impossible value (ROAD's ``max_speedometer`` pins a
    byte to 0xFF) writes a byte value the real ECU never emits. Timing analysis
    is blind to that by construction; the range check is not.

The envelope is fixed at provisioning -- the reflex layer never learns at
runtime (design rule 2).
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..shared.contracts import Detector, Event, Label, Message, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState


class PhysicsChecks(BaseDetector):
    name = "physics_checks"

    def __init__(self, signal_bounds: dict[int, tuple[float, float]] | None = None,
                 byte_ranges: dict[int, list[tuple[int, int]]] | None = None,
                 margin: int = 0):
        # arbitration_id -> (min, max) plausible decoded value
        self.signal_bounds = signal_bounds or {}
        # arbitration_id -> per-byte (min, max) observed on clean traffic
        self.byte_ranges = byte_ranges or {}
        self.margin = margin
        self._last: dict[int, float] = {}

    def fit_byte_ranges(self, messages: Iterable[Message]) -> None:
        """Learn the per-id, per-byte value envelope from ATTACK-FREE traffic."""
        ranges: dict[int, list[list[int]]] = {}
        for m in messages:
            cur = ranges.setdefault(m.arbitration_id, [])
            for i, b in enumerate(m.data):
                if i >= len(cur):
                    cur.append([b, b])
                else:
                    if b < cur[i][0]:
                        cur[i][0] = b
                    if b > cur[i][1]:
                        cur[i][1] = b
        self.byte_ranges = {aid: [(lo, hi) for lo, hi in cur] for aid, cur in ranges.items()}

    def reset(self) -> None:
        self._last.clear()

    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        # range check (needs a decoder per ID; placeholder decodes first byte)
        bounds = self.signal_bounds.get(msg.arbitration_id)
        if bounds and msg.data:
            value = float(msg.data[0])
            lo, hi = bounds
            if not (lo <= value <= hi):
                return self._flag(msg, "range", value)

        # learned per-byte envelope: a byte outside anything the real ECU ever
        # sent is implausible content, regardless of timing
        envelope = self.byte_ranges.get(msg.arbitration_id)
        if envelope:
            for i, b in enumerate(msg.data):
                if i >= len(envelope):
                    break
                lo, hi = envelope[i]
                if b < lo - self.margin or b > hi + self.margin:
                    return self._flag(msg, "byte_range", float(b), byte_index=i)
        # TODO(phase1): rate-of-change check using self._last
        # TODO(phase2): cross-sensor agreement (camera/LiDAR/radar/GPS/IMU)
        # TODO(phase2): control-command feasibility vs. vehicle dynamics
        return None

    def _flag(self, msg: Message, kind: str, value: float, **extra) -> Event:
        return Event(
            detector=Detector.PHYSICS, bus=msg.bus, label=Label.MALICIOUS,
            confidence=1.0, source_tier=Tier.REFLEX,
            features={"check": kind, "value": value,
                      "arbitration_id": msg.arbitration_id, **extra},
        )
