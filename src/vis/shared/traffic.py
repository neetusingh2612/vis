"""Traffic abstraction + replay harness.

A single interface that yields Messages, backed interchangeably by a recorded
dataset, a simulator, or (later) the real bus -- so every detector is developed
against recorded data and moved to hardware unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator

from .contracts import Bus, Message


class TrafficSource(Iterable[Message]):
    """Base class for anything that yields Messages."""

    def __iter__(self) -> Iterator[Message]:  # pragma: no cover - interface
        raise NotImplementedError


class CsvReplaySource(TrafficSource):
    """Replay a simple CSV of CAN frames.

    Expected columns (header): timestamp, arbitration_id, data (hex),
    optionally label (e.g. 'normal'/attack-type). Adapt the column mapping to
    each dataset (Car-Hacking, OTIDS, ROAD, CAN-MIRGU) in datasets/README.md.
    """

    def __init__(self, path: str | Path, bus: Bus = Bus.CAN, attack_labels: set[str] | None = None):
        self.path = Path(path)
        self.bus = bus
        # any label not in this set is treated as benign
        self.attack_labels = attack_labels or set()

    def __iter__(self) -> Iterator[Message]:
        with self.path.open(newline="") as f:
            for row in csv.DictReader(f):
                label = (row.get("label") or "normal").strip().lower()
                is_attack = label in self.attack_labels or label not in ("normal", "benign", "")
                yield Message(
                    arbitration_id=int(row["arbitration_id"], 0),
                    data=bytes.fromhex(row.get("data", "") or ""),
                    bus=self.bus,
                    timestamp=float(row["timestamp"]),
                    is_attack=is_attack,
                    attack_type=None if not is_attack else label,
                )


class SyntheticSource(TrafficSource):
    """Generate periodic benign traffic with an optional injected flood.

    Useful for unit tests and for exercising the harness without a dataset.
    """

    def __init__(self, n: int = 1000, period: float = 0.01, flood_at: int | None = None,
                 flood_len: int = 100, flood_id: int = 0x000):
        self.n, self.period = n, period
        self.flood_at, self.flood_len, self.flood_id = flood_at, flood_len, flood_id

    def __iter__(self) -> Iterator[Message]:
        t = 0.0
        for i in range(self.n):
            in_flood = self.flood_at is not None and self.flood_at <= i < self.flood_at + self.flood_len
            if in_flood:
                yield Message(arbitration_id=self.flood_id, data=b"\x00" * 8,
                              timestamp=t, is_attack=True, attack_type="dos")
                t += self.period / 20      # flood = much higher rate
            else:
                yield Message(arbitration_id=0x100 + (i % 5), data=bytes([i % 256]),
                              timestamp=t, is_attack=False)
                t += self.period
