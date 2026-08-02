"""Physical-layer sender fingerprinting (Phase 1 interface / Phase 4 hardware).

Identifies the *physical* transmitter of a message, catching masquerade and
stolen-key impersonation that authentication alone cannot (goal G4, experiment
E3). Two complementary fingerprints:

  * **clock skew** -- derived purely from message *timestamps*, so it runs on
    recorded data (ROAD) with no special hardware. Each ECU's crystal makes its
    periodic frames arrive at a slightly different rate; a frame on a known id
    arriving with the wrong skew was sent by a different clock (CIDS, Cho &
    Shin, USENIX Security 2016). This is the laptop-runnable masquerade detector.
  * **voltage profile** -- the analog transmitter fingerprint, captured via a
    :class:`~vis.reflex.hal.FingerprintSampler`. Real capture is bench-only
    (Phase 4); a simulated sampler exercises the comparison logic here.

Both are O(1)/message and never learn at runtime -- profiles are fixed at
provisioning (`enrol*`), as the reflex layer requires.
"""
from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import Iterable, Optional

from ..shared.contracts import Detector, Event, Label, Message, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState
from .hal import FingerprintSampler, PhysicalSample


class Fingerprinting(BaseDetector):
    name = "fingerprinting"

    def __init__(self, sampler: Optional[FingerprintSampler] = None,
                 skew_threshold_ppm: float = 200.0, voltage_tol: float = 0.5,
                 window: int = 40, min_samples: int = 20):
        self._nominal_period: dict[int, float] = {}        # enrolled clock-skew baseline
        self._voltage_profiles: dict[int, tuple[float, ...]] = {}   # enrolled voltage centroid
        self.sampler = sampler
        self.skew_threshold_ppm = skew_threshold_ppm
        self.voltage_tol = voltage_tol
        self.window = window
        self.min_samples = min_samples
        self._iat: dict[int, deque[float]] = {}
        self._last_ts: dict[int, float] = {}

    # -- provisioning (no runtime learning) -------------------------------- #
    def enrol(self, arbitration_id: int, sample: object) -> None:
        """Record a legitimate sender's fingerprint during provisioning.

        Accepts a :class:`PhysicalSample` (-> voltage profile) or a number
        (-> nominal clock period for the clock-skew baseline).
        """
        if isinstance(sample, PhysicalSample):
            self._voltage_profiles[arbitration_id] = sample.voltage_features
        elif isinstance(sample, (int, float)):
            self._nominal_period[arbitration_id] = float(sample)
        else:
            raise TypeError(f"unsupported enrolment sample: {type(sample).__name__}")

    def enrol_clock_skew(self, messages: Iterable[Message]) -> None:
        """Learn each id's nominal inter-arrival period from a CLEAN corpus."""
        last: dict[int, float] = {}
        acc: dict[int, list[float]] = {}
        for m in messages:
            if m.arbitration_id in last:
                iat = m.timestamp - last[m.arbitration_id]
                if iat > 0:
                    acc.setdefault(m.arbitration_id, []).append(iat)
            last[m.arbitration_id] = m.timestamp
        for aid, iats in acc.items():
            if iats:
                self._nominal_period[aid] = median(iats)

    def reset(self) -> None:
        self._iat.clear()
        self._last_ts.clear()

    # -- detection --------------------------------------------------------- #
    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        ev = self._check_voltage(msg)
        if ev is not None:
            return ev
        return self._check_clock_skew(msg)

    def _skew_ppm(self, msg: Message) -> Optional[float]:
        """Windowed clock-rate deviation (ppm) vs. the enrolled nominal period.

        Advances the per-id state; returns None until there is enough evidence.
        """
        aid = msg.arbitration_id
        nominal = self._nominal_period.get(aid)
        last = self._last_ts.get(aid)
        self._last_ts[aid] = msg.timestamp
        if nominal is None or last is None or nominal <= 0:
            return None
        iat = msg.timestamp - last
        if iat <= 0:
            return None
        dq = self._iat.setdefault(aid, deque(maxlen=self.window))
        dq.append(iat)
        if len(dq) < self.min_samples:
            return None
        # the transmitter's clock rate relative to the enrolled (legitimate) one
        return (median(dq) / nominal - 1.0) * 1e6

    def calibrate_skew(self, messages: Iterable[Message], quantile: float = 0.999,
                       margin: float = 1.5) -> float:
        """Set the skew threshold from ATTACK-FREE traffic (provisioning step).

        Real buses jitter far more than a crystal's true skew: scheduling,
        arbitration and dropped frames move the windowed median by whole
        percent, so a textbook 200 ppm bound fires on almost every frame. Taking
        a high quantile of the deviation actually observed on clean traffic
        yields a threshold with a bounded false-alarm rate on that vehicle.
        """
        self.reset()
        vals = [abs(p) for p in (self._skew_ppm(m) for m in messages) if p is not None]
        self.reset()
        if vals:
            vals.sort()
            idx = min(len(vals) - 1, int(quantile * (len(vals) - 1)))
            self.skew_threshold_ppm = max(1.0, vals[idx] * margin)
        return self.skew_threshold_ppm

    def _check_clock_skew(self, msg: Message) -> Optional[Event]:
        skew_ppm = self._skew_ppm(msg)
        if skew_ppm is None:
            return None
        if abs(skew_ppm) > self.skew_threshold_ppm:
            return self._flag(msg, "clock_skew", round(skew_ppm, 1))
        return None

    def _check_voltage(self, msg: Message) -> Optional[Event]:
        if self.sampler is None:
            return None
        profile = self._voltage_profiles.get(msg.arbitration_id)
        if not profile:
            return None
        feats = self.sampler.sample(msg).voltage_features
        if len(feats) != len(profile):
            return None
        distance = math.dist(feats, profile)
        if distance > self.voltage_tol:
            return self._flag(msg, "voltage", round(distance, 3))
        return None

    def _flag(self, msg: Message, method: str, value: float) -> Event:
        return Event(
            detector=Detector.FINGERPRINT, bus=msg.bus, label=Label.MALICIOUS,
            confidence=0.95, source_tier=Tier.REFLEX,
            features={"method": method, "arbitration_id": msg.arbitration_id,
                      "masquerade": True, "value": value},
        )
