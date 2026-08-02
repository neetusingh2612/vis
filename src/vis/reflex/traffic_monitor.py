"""Lightweight traffic-shape monitor (Phase 1).

Catches flooding/fuzzing from the *shape* of traffic: a sudden spike in message
rate or a break in the normally-regular per-ID inter-arrival timing. O(1) per
message. This is the simplest reflex detector and validates the whole harness.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from ..shared.contracts import Detector, Event, Label, Message, ResponseAction, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState


class TrafficMonitor(BaseDetector):
    name = "traffic_monitor"

    def __init__(self, window: float = 0.1, rate_threshold: int = 50):
        # flag if > rate_threshold messages arrive within `window` seconds
        self.window = window
        self.rate_threshold = rate_threshold
        self._times: deque[float] = deque()

    def reset(self) -> None:
        self._times.clear()

    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        self._times.append(msg.timestamp)
        while self._times and msg.timestamp - self._times[0] > self.window:
            self._times.popleft()
        if len(self._times) > self.rate_threshold:
            return Event(
                detector=Detector.TRAFFIC,
                bus=msg.bus,
                label=Label.MALICIOUS,
                confidence=0.9,
                source_tier=Tier.REFLEX,
                features={"rate": len(self._times), "window_s": self.window,
                          "arbitration_id": msg.arbitration_id},
                response_taken=ResponseAction.NONE,
            )
        return None
