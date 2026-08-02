"""Detection / latency metrics used by the eval harness."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies_ms: list[float] | None = None

    @property
    def detection_rate(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def false_positive_rate(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def mean_latency_ms(self) -> float:
        xs = self.latencies_ms or []
        return sum(xs) / len(xs) if xs else 0.0

    @property
    def p99_latency_ms(self) -> float:
        xs = sorted(self.latencies_ms or [])
        return xs[int(0.99 * (len(xs) - 1))] if xs else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "detection_rate": round(self.detection_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "precision": round(self.precision, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 4),
            "p99_latency_ms": round(self.p99_latency_ms, 4),
        }
