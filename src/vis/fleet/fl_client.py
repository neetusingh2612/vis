"""Federated-learning client (Phase 3, on-vehicle).

Packages locally-validated candidate antibodies / model deltas into an update.
Participates only when admitted by the attestation gate. Asynchronous and
connectivity-tolerant (the vehicle is fully protected offline; FL only improves).

The update is a plain dict (transport-agnostic): identity + attestation evidence
for the gate, a list of candidate Antibodies (the antibody-sharing path), and an
optional numeric weight delta with a sample count (the FedAvg/Krum path).
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..shared.contracts import Antibody


class FLClient:
    def __init__(self, vehicle_id: str, attestation_token: str = "", certificate: str = ""):
        self.vehicle_id = vehicle_id
        self.attestation_token = attestation_token
        self.certificate = certificate

    def make_update(self, candidates: Optional[Sequence[Antibody]] = None,
                    weights: Optional[Sequence[float]] = None,
                    num_samples: int = 1) -> dict:
        """Build a model update / antibody contribution to send to the server."""
        return {
            "vehicle_id": self.vehicle_id,
            "attestation_token": self.attestation_token,
            "certificate": self.certificate,
            "candidates": list(candidates or []),
            "weights": list(weights) if weights is not None else [],
            "num_samples": num_samples,
        }
