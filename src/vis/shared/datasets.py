"""Phase-1 dataset adapters: native format -> Message stream.

Each public CAN-IDS dataset ships in its own on-disk format, so a single CSV
column map is not enough. This module gives one :class:`TrafficSource` per
dataset that maps the *native* records onto our frozen ``Message`` contract,
so every reflex detector is developed against recorded data and later moved to
hardware unchanged (see ``traffic.py``).

Ground-truth (``is_attack`` / ``attack_type``) is populated here for the eval
harness ONLY. The live system never reads those fields.

Adapters
--------
* :class:`CarHackingSource` -- HCRL Car-Hacking (DoS/Fuzzy/gear/RPM). Per-frame
  ``R``/``T`` flag => exact labels.
* :class:`OtidsSource` -- HCRL OTIDS candump-style text. No per-frame flag; the
  whole file is one scenario, with an optional per-frame injection predicate.
* :class:`RoadSource` -- ORNL ROAD candump ``.log`` + injection metadata; labels
  by ``injection_id`` within ``injection_interval`` (handles masquerade).
* :class:`CanMirguSource` -- CAN-MIRGU labelled CSV (real moving vehicle); a
  per-frame ``Label`` column gives exact labels. Tolerant header auto-detection.
* :class:`VeReMiSource` -- VeReMi V2X misbehavior (simulated BSMs, JSON lines).
  Labels from per-message ``attackerType`` / a GroundTruth file. V2X kinematics
  ride in the structured ``Message.payload`` field (pos/spd/...).

Column / field mappings are documented per class and mirrored in
``datasets/README.md``.

All adapters are stdlib-only (no pandas) to keep the core dependency-light.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Sequence

from .contracts import Bus, Message
from .traffic import TrafficSource

# Labels that mean "no attack", regardless of dataset casing.
_BENIGN_TOKENS = {"r", "normal", "benign", "0", "", "attack-free", "attack_free"}


def _maybe_can_id(raw: str) -> Optional[int]:
    """Like :func:`_parse_can_id` but returns None for placeholders (e.g. ROAD's
    ``"XXX"`` for fuzzing captures, where the injected id is random)."""
    try:
        return _parse_can_id(raw)
    except (ValueError, AttributeError):
        return None


def _parse_can_id(raw: str) -> int:
    """Parse a CAN arbitration id given as hex (``0316``/``0x316``) or decimal.

    Datasets write the id as bare hex without an ``0x`` prefix, so we cannot use
    ``int(raw, 0)``. We try hex first (the common case) and fall back to decimal.
    """
    raw = raw.strip()
    try:
        return int(raw, 16)
    except ValueError:
        return int(raw, 10)


# --------------------------------------------------------------------------- #
# HCRL Car-Hacking
# --------------------------------------------------------------------------- #
class CarHackingSource(TrafficSource):
    """HCRL Car-Hacking dataset (DoS / Fuzzy / gear-spoof / RPM-spoof).

    Native rows are headerless and **variable width** because the number of data
    columns equals the DLC::

        Timestamp,        CAN ID, DLC, DATA[0..DLC-1],            Flag
        1478198376.389427,0316,   8,   05,21,68,09,21,21,00,6f,  R

    Column mapping:
      * ``Timestamp``  -> ``Message.timestamp``  (epoch seconds, float)
      * ``CAN ID``     -> ``Message.arbitration_id``  (bare hex, no ``0x``)
      * ``DLC``        -> count of following data bytes (0..8)
      * ``DATA[i]``    -> ``Message.data``  (each a hex byte)
      * ``Flag``       -> ground truth: ``T`` = injected attack, ``R`` = regular

    Each Car-Hacking file contains exactly one attack class, so ``attack_type``
    names it (used only when the per-frame flag is ``T``).
    """

    def __init__(self, path: str | Path, attack_type: str, bus: Bus = Bus.CAN):
        self.path = Path(path)
        self.attack_type = attack_type
        self.bus = bus

    def __iter__(self) -> Iterator[Message]:
        with self.path.open(newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                fields = [c.strip() for c in row]
                ts = float(fields[0])
                can_id = _parse_can_id(fields[1])
                dlc = int(fields[2])
                data = bytes(int(b, 16) for b in fields[3 : 3 + dlc])
                # The flag is always the final field; with DLC<8 there is no
                # trailing padding, so index 3+dlc lands on it.
                flag = fields[3 + dlc].lower() if len(fields) > 3 + dlc else "r"
                is_attack = flag == "t"
                yield Message(
                    arbitration_id=can_id,
                    data=data,
                    bus=self.bus,
                    timestamp=ts,
                    is_attack=is_attack,
                    attack_type=self.attack_type if is_attack else None,
                )


# --------------------------------------------------------------------------- #
# HCRL OTIDS
# --------------------------------------------------------------------------- #
# Example OTIDS line (candump-derived text):
#   Timestamp: 1479121434.850202        ID: 0260    000    DLC: 8    19 21 22 30 08 8e 6d 3a
_OTIDS_RE = re.compile(
    r"Timestamp:\s*(?P<ts>[\d.]+)\s+"
    r"ID:\s*(?P<id>[0-9A-Fa-f]+)\s+"
    r"(?P<rtr>\d+)\s+"
    r"DLC:\s*(?P<dlc>\d+)\s*"
    r"(?P<data>[0-9A-Fa-f ]*)"
)


class OtidsSource(TrafficSource):
    """HCRL OTIDS dataset (DoS / Fuzzy / Impersonation), candump-style text.

    Native lines look like::

        Timestamp: 1479121434.850202   ID: 0260   000   DLC: 8   19 21 22 30 08 8e 6d 3a

    Field mapping:
      * ``Timestamp`` -> ``Message.timestamp``
      * ``ID``        -> ``Message.arbitration_id``  (bare hex)
      * ``000``/``...`` -> remote/data flag (exposed as ``features`` is not on
        Message, so it only drives the default DoS predicate below)
      * ``DLC``       -> data-byte count
      * trailing hex  -> ``Message.data``

    OTIDS has **no per-frame attack flag** -- each file is one scenario and the
    authors label by injection campaign, not per message. So:

      * ``attack_type=None`` (the Attack-free file) => every frame benign.
      * Otherwise pass ``inject_predicate(msg) -> bool`` to mark injected frames.
        Convenience predicates for the published scenarios are provided as
        :data:`OTIDS_DOS_PREDICATE` etc.

    TODO(phase1): for exact per-frame ground truth, load the authors' injection
    time-windows instead of the heuristic predicates.
    """

    def __init__(
        self,
        path: str | Path,
        attack_type: Optional[str] = None,
        inject_predicate: Optional[Callable[[Message], bool]] = None,
        bus: Bus = Bus.CAN,
    ):
        self.path = Path(path)
        self.attack_type = attack_type
        self.inject_predicate = inject_predicate
        self.bus = bus

    def __iter__(self) -> Iterator[Message]:
        with self.path.open() as f:
            for line in f:
                mt = _OTIDS_RE.search(line)
                if not mt:
                    continue
                dlc = int(mt["dlc"])
                tokens = mt["data"].split()[:dlc]
                msg = Message(
                    arbitration_id=_parse_can_id(mt["id"]),
                    data=bytes(int(b, 16) for b in tokens),
                    bus=self.bus,
                    timestamp=float(mt["ts"]),
                )
                if self.attack_type is not None and self.inject_predicate is not None:
                    if self.inject_predicate(msg):
                        msg.is_attack = True
                        msg.attack_type = self.attack_type
                yield msg


def OTIDS_DOS_PREDICATE(m: Message) -> bool:
    """OTIDS DoS injects a flood of the highest-priority id (0x000)."""
    return m.arbitration_id == 0x000


# --------------------------------------------------------------------------- #
# ORNL ROAD
# --------------------------------------------------------------------------- #
# Example ROAD candump line: "(1597695883.156677) can0 0D0#FFFF0000000000A8"
_ROAD_RE = re.compile(
    r"\((?P<ts>[\d.]+)\)\s+\S+\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]*)"
)


class RoadSource(TrafficSource):
    r"""ORNL ROAD dataset, candump ``.log`` + injection metadata.

    Native lines are Linux ``candump`` format::

        (1597695883.156677) can0 0D0#FFFF0000000000A8
         \_ epoch ts ____/  iface  \_id/ \__ data hex __/

    Field mapping:
      * ``ts``   -> ``Message.timestamp``
      * ``id``   -> ``Message.arbitration_id``  (bare hex before ``#``)
      * ``data`` -> ``Message.data``  (hex after ``#``)

    ROAD frames carry no inline label. Each attack capture ships a metadata
    JSON describing the injection, e.g.::

        {"injection_id": "0xD0",
         "injection_data_str": "ffff0000000000a8",
         "injection_interval": [9.19, 30.06]}

    A frame is ground-truth attack iff its id == ``injection_id`` and its time
    (relative to the first frame in the capture) falls inside an
    ``injection_interval``. This labels masquerade attacks correctly (experiment
    E3): the injected id is a legitimate id, so only the in-window frames count.

    Pass either an explicit ``injection_id``/``injection_interval(s)`` or a
    ``metadata`` dict / path to the capture's JSON. With neither, every frame is
    benign (use this for the ambient/attack-free captures).
    """

    def __init__(
        self,
        path: str | Path,
        injection_id: Optional[int] = None,
        injection_intervals: Optional[Sequence[tuple[float, float]]] = None,
        attack_type: str = "road_injection",
        metadata: Optional[dict | str | Path] = None,
        bus: Bus = Bus.CAN,
    ):
        self.path = Path(path)
        self.attack_type = attack_type
        self.bus = bus
        if metadata is not None:
            meta = metadata if isinstance(metadata, dict) else json.loads(Path(metadata).read_text())
            if injection_id is None:
                # Fuzzing captures use a placeholder ("XXX") because the attack
                # injects RANDOM ids -- there is no single id to key on. Degrade
                # to "no id-based ground truth" rather than mislabelling frames.
                injection_id = _maybe_can_id(str(meta.get("injection_id", "")))
            if injection_intervals is None:
                raw = meta.get("injection_interval")
                # accept a single [start, end] or a list of them
                if raw and isinstance(raw[0], (int, float)):
                    injection_intervals = [tuple(raw)]
                elif raw:
                    injection_intervals = [tuple(iv) for iv in raw]
        self.injection_id = injection_id
        self.injection_intervals = list(injection_intervals or [])

    def _in_attack_window(self, can_id: int, rel_t: float) -> bool:
        if self.injection_id is None or can_id != self.injection_id:
            return False
        return any(start <= rel_t <= end for start, end in self.injection_intervals)

    def __iter__(self) -> Iterator[Message]:
        t0: Optional[float] = None
        with self.path.open() as f:
            for line in f:
                mt = _ROAD_RE.search(line)
                if not mt:
                    continue
                ts = float(mt["ts"])
                if t0 is None:
                    t0 = ts
                can_id = _parse_can_id(mt["id"])
                is_attack = self._in_attack_window(can_id, ts - t0)
                yield Message(
                    arbitration_id=can_id,
                    data=bytes.fromhex(mt["data"]),
                    bus=self.bus,
                    timestamp=ts,
                    is_attack=is_attack,
                    attack_type=self.attack_type if is_attack else None,
                )


# --------------------------------------------------------------------------- #
# CAN-MIRGU
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    """Normalise a header cell for tolerant matching: lower, strip non-alnum."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _find_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first header in ``fieldnames`` matching any candidate name."""
    norm = {_norm(f): f for f in fieldnames}
    for c in candidates:
        if _norm(c) in norm:
            return norm[_norm(c)]
    return None


def _parse_data_field(raw: str, dlc: Optional[int] = None) -> bytes:
    """Parse a CAN-MIRGU ``Data`` cell: contiguous or space-separated hex.

    Handles ``"00 11 22"``, ``"001122"`` and ``"0x00 0x11"``; trims to ``dlc``.
    """
    cleaned = raw.strip().lower().replace("0x", "")
    tokens = cleaned.split() if " " in cleaned else re.findall("..", cleaned)
    if dlc is not None:
        tokens = tokens[:dlc]
    return bytes(int(b, 16) for b in tokens if b)


# CAN-MIRGU candump line WITH a trailing per-frame label:
#   "(1683206351.761563) can0 421#000000FFE37F0065 0"
_MIRGU_RE = re.compile(
    r"\((?P<ts>[\d.]+)\)\s+\S+\s+(?P<id>[0-9A-Fa-f]+)#+(?P<data>[0-9A-Fa-f]*)"
    r"(?:\s+(?P<label>\S+))?"
)


class CanMirguSource(TrafficSource):
    """CAN-MIRGU dataset -- a broad, recent attack set captured on a *moving*
    real vehicle (masquerade, suspension, fabrication, DoS, fuzzing, ...).

    Ships as Linux ``candump`` ``.log`` files with a **trailing per-frame label**
    column, which is what this adapter parses::

        (1683206351.761563) can0 421#000000FFE37F0065 0
         \\_ epoch ts ____/  iface  \\_id/ \\__ data hex __/ ^-- 0=benign 1=attack

    Field mapping:
      * ``ts``    -> ``Message.timestamp``
      * ``id``    -> ``Message.arbitration_id``  (bare hex before ``#``)
      * ``data``  -> ``Message.data``  (hex after ``#``; length varies, no DLC col)
      * ``label`` -> ground truth; anything not in :data:`_BENIGN_TOKENS` is an
        attack. Benign captures carry an all-zero label column.

    A labelled-CSV variant (as some redistributions ship) is auto-detected and
    handled too; the format is sniffed from the first non-empty line.
    """

    def __init__(
        self,
        path: str | Path,
        bus: Bus = Bus.CAN,
        attack_type: str = "attack",
        id_col: Optional[str] = None,
        data_col: Optional[str] = None,
        label_col: Optional[str] = None,
        category_col: Optional[str] = None,
        time_col: Optional[str] = None,
        dlc_col: Optional[str] = None,
    ):
        self.path = Path(path)
        self.bus = bus
        self.attack_type = attack_type
        self._cols = dict(
            id=id_col, data=data_col, label=label_col,
            category=category_col, time=time_col, dlc=dlc_col,
        )

    # -- format sniffing --------------------------------------------------- #
    def _is_candump(self) -> bool:
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    return _MIRGU_RE.search(line) is not None
        return False

    def _resolve(self, fieldnames: Sequence[str]) -> dict[str, Optional[str]]:
        aliases = {
            "time": ("timestamp", "time", "ts"),
            "id": ("arbitration_id", "can_id", "id", "arbitrationid"),
            "dlc": ("dlc",),
            "data": ("data", "payload"),
            "label": ("label", "flag", "attack", "class"),
            "category": ("category", "attack_type", "attacktype", "label_name", "type"),
        }
        resolved: dict[str, Optional[str]] = {}
        for key, cands in aliases.items():
            resolved[key] = self._cols.get(key) or _find_col(fieldnames, cands)
        if resolved["id"] is None or resolved["data"] is None:
            raise ValueError(
                f"CAN-MIRGU: could not find id/data columns in header {list(fieldnames)}; "
                "pass id_col=/data_col= explicitly."
            )
        return resolved

    def _is_attack_label(self, label: Optional[str]) -> bool:
        if label is None:
            return False
        return _norm(label) not in {_norm(t) for t in _BENIGN_TOKENS}

    def __iter__(self) -> Iterator[Message]:
        if self._is_candump():
            yield from self._iter_candump()
        else:
            yield from self._iter_csv()

    def _iter_candump(self) -> Iterator[Message]:
        with self.path.open() as f:
            for line in f:
                mt = _MIRGU_RE.search(line)
                if not mt:
                    continue
                data = mt["data"] or ""
                if len(data) % 2:          # guard against odd-length hex
                    data = data[:-1]
                is_attack = self._is_attack_label(mt["label"])
                yield Message(
                    arbitration_id=_parse_can_id(mt["id"]),
                    data=bytes.fromhex(data),
                    bus=self.bus,
                    timestamp=float(mt["ts"]),
                    is_attack=is_attack,
                    attack_type=self.attack_type if is_attack else None,
                )

    def _iter_csv(self) -> Iterator[Message]:
        with self.path.open(newline="") as f:
            reader = csv.DictReader(f)
            cols = self._resolve(reader.fieldnames or [])
            for row in reader:
                dlc = int(row[cols["dlc"]]) if cols["dlc"] and row.get(cols["dlc"]) else None
                label = (row.get(cols["label"]) or "0") if cols["label"] else "0"
                is_attack = self._is_attack_label(label.strip())
                category = row.get(cols["category"]) if cols["category"] else None
                ts = float(row[cols["time"]]) if cols["time"] and row.get(cols["time"]) else 0.0
                yield Message(
                    arbitration_id=_parse_can_id(row[cols["id"]]),
                    data=_parse_data_field(row[cols["data"]], dlc),
                    bus=self.bus,
                    timestamp=ts,
                    is_attack=is_attack,
                    attack_type=(category or self.attack_type) if is_attack else None,
                )


# --------------------------------------------------------------------------- #
# VeReMi (V2X)
# --------------------------------------------------------------------------- #
# VeReMi attacker-type codes -> names (original VeReMi + common extension set).
VEREMI_ATTACKER_TYPES: dict[int, str] = {
    0: "genuine",
    1: "const_pos",
    2: "const_pos_offset",
    4: "random_pos",
    8: "random_pos_offset",
    16: "eventual_stop",
}

def load_veremi_ground_truth(path: str | Path) -> dict[int, int]:
    """Build a ``messageID -> attackerType`` map from a VeReMi GroundTruth log.

    The GroundTruth file is JSON-lines like the receiver logs; each record
    carries ``messageID`` and ``attackerType``.
    """
    gt: dict[int, int] = {}
    for obj in _iter_json_lines(Path(path)):
        if "messageID" in obj and "attackerType" in obj:
            gt[int(obj["messageID"])] = int(obj["attackerType"])
    return gt


def _iter_json_lines(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class VeReMiSource(TrafficSource):
    """VeReMi V2X misbehavior dataset (simulated BSMs; feeds v2x_misbehavior).

    Each receiver's log is JSON-lines. Two record types matter::

        {"type":2, ...}                                  # receiver's own GPS (skipped)
        {"type":3,"rcvTime":25201.5,"sendTime":25201.4,  # a received BSM
         "sender":11,"messageID":4783,
         "pos":[4101.4,5482.6,0],"spd":[12.3,4.5,0], ...}

    Field mapping:
      * ``rcvTime``   -> ``Message.timestamp``
      * ``sender``    -> ``Message.arbitration_id``  (sender pseudonym, not a CAN id)
      * ``pos``/``spd``/``sendTime``/``messageID`` -> ``Message.payload``
        (keys ``pos``, ``spd``, ``send_time``, ``message_id``)
      * ``bus``       -> :attr:`Bus.V2X`

    Ground truth: type-3 records may embed ``attackerType``; otherwise pass
    ``ground_truth`` (a GroundTruth log path/dict, or a pre-built
    ``messageID -> attackerType`` map). ``attackerType == 0`` is benign; other
    codes map to names via :data:`VEREMI_ATTACKER_TYPES`.
    """

    def __init__(
        self,
        log_path: str | Path,
        ground_truth: Optional[str | Path | Mapping[int, int]] = None,
        bus: Bus = Bus.V2X,
    ):
        self.log_path = Path(log_path)
        self.bus = bus
        if ground_truth is None or isinstance(ground_truth, Mapping):
            self.gt: Mapping[int, int] = ground_truth or {}
        else:
            self.gt = load_veremi_ground_truth(ground_truth)

    def __iter__(self) -> Iterator[Message]:
        for obj in _iter_json_lines(self.log_path):
            if obj.get("type") != 3:          # skip own-position (type 2) records
                continue
            atype = obj.get("attackerType")
            if atype is None:
                atype = self.gt.get(int(obj.get("messageID", -1)), 0)
            atype = int(atype)
            is_attack = atype != 0
            yield Message(
                arbitration_id=int(obj.get("sender", 0)),
                bus=self.bus,
                timestamp=float(obj.get("rcvTime", 0.0)),
                payload={
                    "pos": tuple(obj.get("pos", ())),
                    "spd": tuple(obj.get("spd", ())),
                    "send_time": float(obj["sendTime"]) if "sendTime" in obj else None,
                    "message_id": int(obj.get("messageID", -1)),
                },
                is_attack=is_attack,
                attack_type=VEREMI_ATTACKER_TYPES.get(atype, f"attacker_{atype}")
                if is_attack else None,
            )
