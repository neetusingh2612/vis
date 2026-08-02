"""Key storage + signing abstraction (Phase 3 default / Phase 4 HSM).

Centralises the "root of trust" so the attestation gate and OTA distributor do
not hard-code a raw key. Two backends:

  * :class:`SoftwareKeyStore` -- a plain in-process key (the Phase-3 default).
  * :class:`SimulatedHSM` -- models a tamper-resistant HSM: the private key is
    generated inside and NEVER exported (no accessor), and it hands out a
    verify-only :class:`HSMVerifier` capability for vehicles. This lets the full
    signed-OTA / attestation flow run on a laptop as it would with real silicon.

The underlying primitive is HMAC-SHA256 (symmetric) to stay stdlib-only; real
hardware uses an asymmetric key whose *public* half is the vehicle's root of
trust. The interface and capability separation are what matter here.
TODO(phase4): back `SimulatedHSM` with a real PKCS#11 / TPM device.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from abc import ABC, abstractmethod


def measure_firmware(image: bytes) -> str:
    """Secure-boot style measurement of a firmware image (content hash).

    A tampered image yields a different measurement, so the attestation gate can
    reject it by comparing against the fleet's known-good measurement list.
    """
    return hashlib.sha256(image).hexdigest()


class KeyStore(ABC):
    @abstractmethod
    def sign(self, data: bytes) -> str:  # pragma: no cover - interface
        ...

    @abstractmethod
    def verify(self, data: bytes, signature: str) -> bool:  # pragma: no cover - interface
        ...


class SoftwareKeyStore(KeyStore):
    """Plain in-process HMAC key (exportable). The Phase-3 default."""

    def __init__(self, key: bytes):
        self._key = bytes(key)

    def sign(self, data: bytes) -> str:
        return hmac.new(self._key, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        return bool(signature) and hmac.compare_digest(signature, self.sign(data))


class HSMVerifier:
    """Verify-only capability handed to vehicles: verifies signatures from its
    HSM but cannot sign and cannot read the private key."""

    def __init__(self, hsm: "SimulatedHSM"):
        self._hsm = hsm

    def verify(self, data: bytes, signature: str) -> bool:
        return self._hsm.verify(data, signature)


class SimulatedHSM(KeyStore):
    """Tamper-resistant key store simulator -- the private key never leaves."""

    def __init__(self, key: bytes | None = None):
        # name-mangled + no accessor: nothing outside the instance can read it
        self.__key = bytes(key) if key is not None else secrets.token_bytes(32)
        self.key_id = hashlib.sha256(self.__key).hexdigest()[:16]

    def sign(self, data: bytes) -> str:
        return hmac.new(self.__key, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        return bool(signature) and hmac.compare_digest(signature, self.sign(data))

    def verifier(self) -> HSMVerifier:
        """The public verification capability for vehicles (no signing key)."""
        return HSMVerifier(self)


def as_keystore(key_or_store) -> KeyStore:
    """Accept a KeyStore, or wrap raw key bytes in a SoftwareKeyStore."""
    if isinstance(key_or_store, KeyStore):
        return key_or_store
    if isinstance(key_or_store, (bytes, bytearray)):
        return SoftwareKeyStore(bytes(key_or_store))
    raise TypeError(f"expected KeyStore or bytes, got {type(key_or_store).__name__}")
