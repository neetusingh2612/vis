"""Cross-module data contracts for VIS.

These are the schemas that cross module boundaries. EVERY other module either
produces or consumes one of these. Treat this file as a frozen, versioned API:
do not change a field casually, because the whole system depends on it.

Three core artifacts:
  * Message  -- a single bus/V2X message flowing through the system.
  * Event    -- the "antigen": a captured detection, emitted by detectors.
  * Antibody -- a validated detector, shared across the fleet.

See CLAUDE.md (section "Contracts") for the design rationale.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

# 1.1: added Message.payload (additive, backward-compatible).
SCHEMA_VERSION = "1.1"


class Tier(str, Enum):
    REFLEX = "reflex"
    ADAPTIVE = "adaptive"
    FLEET = "fleet"


class Bus(str, Enum):
    CAN = "can"
    CAN_FD = "can_fd"
    ETH = "eth"
    V2X = "v2x"


class Label(str, Enum):
    BENIGN = "benign"
    MALICIOUS = "malicious"     # high-confidence (decoy / deterministic check)
    SUSPECTED = "suspected"     # low-confidence (anomaly score) -- needs corroboration


class Detector(str, Enum):
    AUTHENTICATION = "authentication"
    FINGERPRINT = "fingerprint"
    PHYSICS = "physics"
    TRAFFIC = "traffic"
    ANOMALY = "anomaly"
    DECOY = "decoy"
    CORRELATION = "correlation"
    V2X_MISBEHAVIOR = "v2x_misbehavior"


class ResponseAction(str, Enum):
    NONE = "none"
    LOG = "log"
    FILTER = "filter"
    ISOLATE = "isolate"
    MINIMAL_RISK = "minimal_risk"


class ArtifactType(str, Enum):
    RULE = "rule"
    THRESHOLD_SET = "threshold_set"
    SEQUENCE_SIGNATURE = "sequence_signature"
    MODEL_DELTA = "model_delta"


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Message:
    """A single message on a bus or over V2X.

    `is_attack` / `attack_type` are ground-truth labels used only for evaluation
    against datasets; the live system never reads them.
    """
    arbitration_id: int
    data: bytes = b""
    bus: Bus = Bus.CAN_FD
    timestamp: float = field(default_factory=time.time)
    # optional security fields (populated when SecOC is in use)
    mac: Optional[bytes] = None
    freshness: Optional[int] = None
    # structured payload for buses whose content does not fit raw CAN `data`
    # bytes -- e.g. a V2X BSM ({"pos": (x,y,z), "spd": (x,y,z), ...}). CAN/CAN-FD
    # frames leave this empty and use `data`; consumers must tolerate {}.
    payload: dict[str, Any] = field(default_factory=dict)
    # ground-truth (evaluation only)
    is_attack: bool = False
    attack_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bus"] = self.bus.value
        d["data"] = self.data.hex()
        d["mac"] = self.mac.hex() if self.mac else None
        return d


@dataclass
class Event:
    """The 'antigen': a captured detection emitted by a detector.

    Flows UP the dependency tree (reflex -> adaptive -> fleet). Never flows down.
    """
    detector: Detector
    bus: Bus
    label: Label
    confidence: float = 1.0
    source_tier: Tier = Tier.REFLEX
    features: dict[str, Any] = field(default_factory=dict)
    response_taken: ResponseAction = ResponseAction.NONE
    event_id: str = field(default_factory=_new_id)
    timestamp: float = field(default_factory=time.time)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detector"] = self.detector.value
        d["bus"] = self.bus.value
        d["label"] = self.label.value
        d["source_tier"] = self.source_tier.value
        d["response_taken"] = self.response_taken.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class Antibody:
    """A validated detector, shared across the fleet.

    NOT raw data, NOT an untested guess. Carries provenance + a validation record
    + the maker's signature. Only a fleet-validated antibody ever gains authority
    to act (see fleet/ and adaptive/shadow_runner.py).
    """
    attack_class: str
    artifact_type: ArtifactType
    artifact: dict[str, Any]                       # the actual rule/threshold/delta
    provenance: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    deployment: dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None                # maker's signature (None until signed)
    antibody_id: str = field(default_factory=_new_id)
    created: float = field(default_factory=time.time)
    schema_version: str = SCHEMA_VERSION

    @property
    def is_validated(self) -> bool:
        return bool(self.validation.get("server_side_passed")) and self.signature is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["artifact_type"] = self.artifact_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
