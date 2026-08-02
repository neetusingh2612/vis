"""Deception subsystem (Phase 2).

Decoys (decoy IDs, phantom ECUs, decoy diagnostic/UDS services, honeytokens) on
the NON-SAFETY surface, behind strong isolation. Any interaction is hostile by
construction -> emits a high-confidence, auto-labelled Event. Because nothing
legitimate ever touches a decoy, these are the cleanest training data in the
system, and they corroborate soft anomaly scores in the correlation engine.

This module models the deception surface in software (decoy ids / phantom ECU
addresses / honeytokens, plus rotation and per-vehicle diversity). The physical
isolation it must live behind -- hypervisor/VLAN, separate from safety ECUs --
is a Phase-4 deployment concern; TODO(phase4).
"""
from __future__ import annotations

import random
from typing import Iterable, Optional

from ..shared.contracts import Bus, Detector, Event, Label, Message, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState


def _as_bytes(v) -> bytes:
    return v if isinstance(v, (bytes, bytearray)) else str(v).encode()


class DecoyListener(BaseDetector):
    name = "decoys"

    def __init__(
        self,
        decoy_ids: Iterable[int] | None = None,
        phantom_ecu_ids: Iterable[int] | None = None,
        honeytokens: dict[str, bytes | str] | None = None,
        rotation_pool: Iterable[int] | None = None,
        rotation_size: int | None = None,
        vehicle_seed: int = 0,
    ):
        self.decoy_ids: set[int] = set(decoy_ids or ())       # IDs no legit component uses
        self.phantom_ecu_ids: set[int] = set(phantom_ecu_ids or ())   # decoy UDS/diag addresses
        self.honeytokens: dict[str, bytes] = {
            name: _as_bytes(val) for name, val in (honeytokens or {}).items() if _as_bytes(val)
        }
        self._pool: list[int] = sorted(set(rotation_pool or ()))
        self.rotation_size = rotation_size
        self.vehicle_seed = vehicle_seed
        self.epoch = 0
        self._captured: list[Event] = []                       # auto-labelled corpus
        if self._pool and rotation_size:
            self.rotate(0)

    # -- rotation / per-vehicle diversity ---------------------------------- #
    def rotate(self, epoch: int) -> set[int]:
        """Activate a fresh, per-vehicle-deterministic subset of the decoy pool.

        Rotating which ids are live (seeded by vehicle) means an attacker who
        maps the decoys on one car learns nothing about the next, and nothing
        durable about this one.
        """
        if self._pool and self.rotation_size:
            rnd = random.Random(f"{self.vehicle_seed}:{epoch}")
            k = min(self.rotation_size, len(self._pool))
            self.decoy_ids = set(rnd.sample(self._pool, k))
        self.epoch = epoch
        return self.decoy_ids

    # -- detection --------------------------------------------------------- #
    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        """Inline on a traffic COPY: flag any frame touching the decoy surface."""
        aid = msg.arbitration_id
        if aid in self.decoy_ids:
            return self._emit("decoy_id", {"arbitration_id": aid})
        if aid in self.phantom_ecu_ids:
            return self._emit("phantom_ecu", {"arbitration_id": aid})
        token = self._honeytoken_in(msg.data)
        if token is not None:
            return self._emit("honeytoken", {"arbitration_id": aid, "token": token})
        return None

    def on_interaction(self, arbitration_id: int, detail: dict) -> Optional[Event]:
        """Explicit callback for a non-bus interaction with a decoy id."""
        if arbitration_id in self.decoy_ids or arbitration_id in self.phantom_ecu_ids:
            surface = "phantom_ecu" if arbitration_id in self.phantom_ecu_ids else "decoy_id"
            return self._emit(surface, {"arbitration_id": arbitration_id, **detail})
        return None

    def on_honeytoken_read(self, name: str, detail: dict | None = None) -> Optional[Event]:
        """Call when a planted honeytoken is read/exfiltrated -- always hostile."""
        if name in self.honeytokens:
            return self._emit("honeytoken", {"token": name, **(detail or {})})
        return None

    @property
    def captured_events(self) -> list[Event]:
        """Auto-labelled MALICIOUS events captured so far (clean training data)."""
        return list(self._captured)

    # -- internals --------------------------------------------------------- #
    def _honeytoken_in(self, data: bytes) -> Optional[str]:
        for name, value in self.honeytokens.items():
            if value and value in data:
                return name
        return None

    def _emit(self, surface: str, features: dict) -> Event:
        ev = Event(
            detector=Detector.DECOY, bus=Bus.CAN, label=Label.MALICIOUS,
            confidence=1.0, source_tier=Tier.ADAPTIVE,
            features={"surface": surface, **features},
        )
        self._captured.append(ev)
        return ev
