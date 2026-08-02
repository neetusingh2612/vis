"""Admission gate (Phase 3).

Only genuine, uncompromised, certified vehicles may contribute to fleet
learning. Blocks Sybil/poisoning at the door by keeping the malicious fraction
f/n small (see paper Section 8, assumption A2).

Software model of the trust checks (real PKI / HSM-backed attestation is
Phase 4): a vehicle is admitted iff (a) its certificate verifies against the CA,
(b) its secure-boot *measurement* is on the known-good list, and (c) it has not
been revoked. The CA/cert here are modelled with HMAC; swap for asymmetric
signatures + remote attestation on the bench.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..shared.keystore import KeyStore, as_keystore
from .revocation_service import RevocationService


class AttestationGate:
    def __init__(self, ca_key: bytes | KeyStore, trusted_measurements: Iterable[str],
                 revocation: Optional[RevocationService] = None):
        # ca_key may be raw bytes (wrapped in a SoftwareKeyStore) or an HSM
        self.ca: KeyStore = as_keystore(ca_key)
        self.trusted_measurements = set(trusted_measurements)
        self.revocation = revocation
        self._admitted: set[str] = set()

    def issue_certificate(self, vehicle_id: str) -> str:
        """CA enrollment helper: a cert binding the vehicle id to the CA key."""
        return self.ca.sign(vehicle_id.encode())

    def admit(self, vehicle_id: str, attestation_token: str, certificate: str) -> bool:
        """Verify integrity attestation + valid certificate before accepting updates."""
        if self.revocation is not None and self.revocation.is_revoked(vehicle_id):
            return False
        if not self.ca.verify(vehicle_id.encode(), certificate):
            return False
        if attestation_token not in self.trusted_measurements:
            return False
        self._admitted.add(vehicle_id)
        return True

    def is_admitted(self, vehicle_id: str) -> bool:
        return vehicle_id in self._admitted
