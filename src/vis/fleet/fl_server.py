"""Federated-learning server + robust aggregation (Phase 3, backend).

Combines client updates with Byzantine-robust aggregation (Krum / trimmed-mean /
median). Tolerates f malicious clients while n >= 2f + 3 (the Krum condition).
Start with plain FedAvg, then swap in robust aggregation to run experiment E5
(detection vs. % poisoned clients).

Two aggregation paths:
  * numeric weight deltas -> `fedavg` / `krum`;
  * candidate antibodies   -> `tally_candidates` (fleet quorum: many vehicles
    independently proposing the same rule is itself strong corroboration).

Pure Python (no numpy/flwr) so it runs in core; a real backend can move behind
the `.[fl]` extra.
"""
from __future__ import annotations

import json
from typing import Sequence

from ..shared.contracts import Antibody


def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def fedavg(updates: Sequence[dict]) -> dict:
    """Sample-weighted average of the client weight vectors (no robustness)."""
    vecs = [(u["weights"], u.get("num_samples", 1)) for u in updates if u.get("weights")]
    if not vecs:
        return {"weights": []}
    dim = len(vecs[0][0])
    total = sum(n for _, n in vecs) or 1
    agg = [sum(n * vec[d] for vec, n in vecs) / total for d in range(dim)]
    return {"weights": agg}


def krum(updates: Sequence[dict], f: int) -> dict:
    """Robust aggregation tolerant of f malicious updates (n >= 2f + 3).

    Returns the single update whose weight vector is closest (by summed squared
    distance to its n-f-2 nearest neighbours) to the honest majority -- outliers
    from poisoning clients are geometrically isolated and never selected.
    """
    vecs = [u["weights"] for u in updates]
    n = len(vecs)
    if n < 2 * f + 3:
        raise ValueError(f"krum needs n >= 2f+3 (got n={n}, f={f})")
    m = n - f - 2
    best_i, best_score = 0, float("inf")
    for i in range(n):
        dists = sorted(_sqdist(vecs[i], vecs[j]) for j in range(n) if j != i)
        score = sum(dists[:m])
        if score < best_score:
            best_i, best_score = i, score
    return updates[best_i]


def tally_candidates(updates: Sequence[dict], quorum: int = 2) -> list[Antibody]:
    """Promote candidate antibodies seen (identically) by >= `quorum` vehicles."""
    groups: dict[tuple, list[tuple]] = {}
    for u in updates:
        for ab in u.get("candidates", []):
            key = (ab.attack_class, ab.artifact_type.value, json.dumps(ab.artifact, sort_keys=True))
            groups.setdefault(key, []).append((u.get("vehicle_id"), ab))

    promoted: list[Antibody] = []
    for (attack_class, _, _), members in groups.items():
        voters = sorted({vid for vid, _ in members if vid is not None})
        if len(voters) >= quorum:
            rep = members[0][1]
            promoted.append(Antibody(
                attack_class=attack_class,
                artifact_type=rep.artifact_type,
                artifact=rep.artifact,
                provenance={"source": "fleet_quorum", "contributors": voters, "votes": len(voters)},
            ))
    return promoted


class FLServer:
    def __init__(self, aggregator=krum, byzantine_f: int = 1):
        self.aggregator = aggregator
        self.byzantine_f = byzantine_f

    def aggregate(self, updates) -> dict:
        updates = list(updates)
        if self.aggregator is krum:
            return krum(updates, self.byzantine_f)
        return self.aggregator(updates)
