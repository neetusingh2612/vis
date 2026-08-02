"""SecOC-style message authentication (Phase 1 software / Phase 4 HSM).

Prevention, not detection (goal G1): a message that fails its MAC or carries a
stale freshness value is rejected outright. Phase 1 holds keys in software;
Phase 4 moves them into an HSM/SHE and re-measures latency.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Optional

from ..shared.contracts import Detector, Event, Label, Message, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState

MAC_LEN = 4  # truncated MAC bytes


def compute_mac(key: bytes, arbitration_id: int, data: bytes, freshness: int) -> bytes:
    msg = arbitration_id.to_bytes(4, "big") + data + freshness.to_bytes(8, "big")
    return hmac.new(key, msg, hashlib.sha256).digest()[:MAC_LEN]


class Authentication(BaseDetector):
    """Verifies protected messages. Emits an Event when verification fails."""
    name = "authentication"

    def __init__(self, keys: dict[int, bytes] | None = None):
        self.keys = keys or {}                       # arbitration_id -> key
        self._last_freshness: dict[int, int] = {}

    def reset(self) -> None:
        self._last_freshness.clear()

    def is_protected(self, msg: Message) -> bool:
        return msg.arbitration_id in self.keys

    def verify(self, msg: Message) -> bool:
        key = self.keys.get(msg.arbitration_id)
        if key is None or msg.mac is None or msg.freshness is None:
            return False
        # replay check: freshness must strictly increase
        if msg.freshness <= self._last_freshness.get(msg.arbitration_id, -1):
            return False
        expected = compute_mac(key, msg.arbitration_id, msg.data, msg.freshness)
        if not hmac.compare_digest(expected, msg.mac):
            return False
        self._last_freshness[msg.arbitration_id] = msg.freshness
        return True

    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        if self.is_protected(msg) and not self.verify(msg):
            return Event(
                detector=Detector.AUTHENTICATION, bus=msg.bus, label=Label.MALICIOUS,
                confidence=1.0, source_tier=Tier.REFLEX,
                features={"arbitration_id": msg.arbitration_id, "reason": "mac_or_freshness"},
            )
        return None
