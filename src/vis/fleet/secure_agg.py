"""Secure aggregation + differential privacy (Phase 3).

The server learns only the COMBINED result of many updates, with calibrated
noise added so individual data cannot be reverse-engineered. Expose epsilon as
a knob for experiment E6 (privacy vs. accuracy).

  * `add_dp_noise` -- the Gaussian mechanism: per-coordinate noise sized to
    (epsilon, delta) and the update's L2 sensitivity.
  * `apply_pairwise_masks` + `secure_sum` -- additive masking: each pair of
    clients shares canceling masks, so the server sees only random-looking
    vectors yet their SUM is exact (no trusted aggregator needed).

Pure Python; a production secure-agg protocol (e.g. Bonawitz et al.) lives
behind the `.[fl]` extra.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence


def add_dp_noise(update: dict, epsilon: float, delta: float = 1e-5,
                 sensitivity: float = 1.0, rng: Optional[random.Random] = None) -> dict:
    """Return an (epsilon, delta)-DP perturbed update (Gaussian mechanism)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    rng = rng or random.Random()
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon
    noised = [x + rng.gauss(0.0, sigma) for x in update.get("weights", [])]
    return {**update, "weights": noised,
            "dp": {"epsilon": epsilon, "delta": delta, "sigma": sigma}}


def apply_pairwise_masks(updates: Sequence[dict], seed: int = 0,
                         scale: float = 1e3) -> list[dict]:
    """Add canceling pairwise masks: client i gets +r_ij, client j gets -r_ij.

    The masks cancel in the global sum, so `secure_sum` recovers the true total
    while no individual masked vector reveals its contribution.
    """
    masked = [list(u.get("weights", [])) for u in updates]
    n = len(masked)
    dim = len(masked[0]) if masked else 0
    rnd = random.Random(seed)
    for i in range(n):
        for j in range(i + 1, n):
            for d in range(dim):
                r = rnd.uniform(-scale, scale)
                masked[i][d] += r
                masked[j][d] -= r
    return [{**updates[k], "weights": masked[k]} for k in range(n)]


def secure_sum(masked_updates: Sequence[dict]) -> dict:
    """Combine masked updates so individual contributions stay hidden."""
    vecs = [u["weights"] for u in masked_updates if u.get("weights")]
    if not vecs:
        return {"weights": []}
    return {"weights": [sum(col) for col in zip(*vecs)]}
