"""Signed OTA distribution (Phase 3) -- R156-style managed update.

Signs a validated antibody and ships it; each vehicle verifies the signature
(root of trust) before calling shadow_runner.activate(). Only here does a
detector graduate from shadow mode to ACTING.

The signature is modelled with HMAC over the antibody's identity payload
(class + artifact + validation record), so any tampering in transit invalidates
it. Real deployment uses an asymmetric signature with the maker public key
burned in as the vehicle's root of trust (HSM, Phase 4).
"""
from __future__ import annotations

import json

from ..shared.contracts import Antibody
from ..shared.keystore import KeyStore, as_keystore


class OTADistributor:
    def __init__(self, maker_key: bytes | KeyStore):
        # maker_key may be raw bytes or an HSM holding the non-exportable key
        self.maker: KeyStore = as_keystore(maker_key)

    def _payload(self, ab: Antibody) -> bytes:
        return json.dumps({
            "attack_class": ab.attack_class,
            "artifact_type": ab.artifact_type.value,
            "artifact": ab.artifact,
            "validation": ab.validation,
        }, sort_keys=True).encode()

    def sign(self, ab: Antibody) -> Antibody:
        """Attach the maker's signature to a validated antibody."""
        if not ab.validation.get("server_side_passed"):
            raise ValueError("refusing to sign an antibody that has not passed validation")
        ab.signature = self.maker.sign(self._payload(ab))
        return ab

    def verify_and_apply(self, ab: Antibody, shadow_runner) -> bool:
        """On-vehicle: verify signature against the maker key, then activate."""
        if not self.maker.verify(self._payload(ab), ab.signature or ""):
            return False    # forged or tampered -> never acts
        if not ab.is_validated:
            return False
        shadow_runner.add_candidate(ab)        # install the distributed antibody...
        shadow_runner.activate(ab.antibody_id)  # ...and graduate it out of shadow
        return shadow_runner.is_active(ab.antibody_id)
