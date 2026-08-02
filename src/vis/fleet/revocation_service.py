"""Misbehavior revocation (Phase 3).

Collects V2X misbehavior reports; a consistently dishonest vehicle has its
certificates revoked and is removed from the trusted group. FL augments this by
sharing learned misbehavior patterns fleet-wide. The attestation gate consults
this service so a revoked vehicle can no longer contribute to fleet learning.
"""
from __future__ import annotations


class RevocationService:
    def __init__(self) -> None:
        self._reports: dict[str, int] = {}
        self._revoked: set[str] = set()

    def report(self, vehicle_id: str) -> None:
        self._reports[vehicle_id] = self._reports.get(vehicle_id, 0) + 1

    def revoke(self, vehicle_id: str) -> None:
        """Explicitly revoke a vehicle (e.g. after a server-side investigation)."""
        self._revoked.add(vehicle_id)

    def is_revoked(self, vehicle_id: str, threshold: int = 5) -> bool:
        """Revoked if explicitly revoked OR reported at/above the threshold."""
        return vehicle_id in self._revoked or self.should_revoke(vehicle_id, threshold)

    def should_revoke(self, vehicle_id: str, threshold: int = 5) -> bool:
        return self._reports.get(vehicle_id, 0) >= threshold
