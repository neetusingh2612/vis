"""Fleet-layer experiments over an EMULATED fleet on real data (E5, E6, E7).

`experiments.py` only exercises proxies for these (weight-vector error, RMS
noise, pipeline stages). This module runs the fleet layer end to end and reports
the three numbers the paper table asks for:

  E5  detection RETAINED at f/n = 10/20/30 %, Krum vs. averaging
  E6  detection vs. privacy budget epsilon
  E7  T_immunity vs. fleet size N

The federated model
------------------
For FedAvg/Krum/DP to mean anything, the thing being aggregated must be a real
detector whose parameters are a numeric vector. We use the per-id **minimum
plausible inter-arrival**:

    theta[k] = period_ratio * median_IAT(id_k)     for the K monitored ids

A frame on id_k arriving sooner than theta[k] is flagged: this is exactly the
compressed-gap rule that catches injection on a *legitimate* id (Car-Hacking
gear/RPM spoofing, ~0.97 standalone). Each vehicle estimates theta from its own
shard of benign traffic; the server aggregates; detection is then measured on the
real attack capture. So "detection retained" is a genuine end-to-end number, not
a distance in weight space.

Threat model for E5: a *scaling* model-poisoning attack. Malicious clients submit
-(n_honest/f) x their honest estimate, which drags the arithmetic mean to ~0.
Thresholds clamp at 0 (negative gaps are unphysical), so a mean near zero means
the detector never fires -- the attacker's goal is to BLIND the fleet, and the
metric is how much detection survives.

Run: python eval/run_fleet.py
"""
from __future__ import annotations

import argparse
import itertools
import random
import statistics
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from harness import evaluate

from vis.adaptive.shadow_runner import ShadowRunner
from vis.fleet.attestation_gate import AttestationGate
from vis.fleet.fl_client import FLClient
from vis.fleet.fl_server import fedavg, krum, tally_candidates
from vis.fleet.ota_distributor import OTADistributor
from vis.fleet.secure_agg import add_dp_noise
from vis.fleet.validation_lab import ValidationLab
from vis.shared.contracts import Antibody, ArtifactType, Detector, Event, Label, Message, Tier
from vis.shared.datasets import CarHackingSource
from vis.shared.detector import BaseDetector
from vis.shared.keystore import SimulatedHSM, measure_firmware
from vis.shared.state import VehicleState

ROOT = Path(__file__).resolve().parents[1]
CAR_HACKING = ROOT / "datasets" / "Car-Hacking Dataset"
ATTACK_FILE = "RPM_dataset.csv"          # spoofing on a LEGITIMATE id -> timing model matters

PERIOD_RATIO = 0.5                        # same convention as AnomalyDetector


# --------------------------------------------------------------------------- #
# the federated detector
# --------------------------------------------------------------------------- #
class ThresholdDetector(BaseDetector):
    """Per-id minimum-plausible-gap detector parameterised by a weight vector."""

    name = "fleet_threshold"

    def __init__(self, ids: Sequence[int], theta: Sequence[float]):
        self.index = {aid: i for i, aid in enumerate(ids)}
        # negative/zero thresholds are unphysical -> clamp (a blinded detector)
        self.theta = [max(0.0, t) for t in theta]
        self._last: dict[int, float] = {}

    def reset(self) -> None:
        self._last.clear()

    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        i = self.index.get(msg.arbitration_id)
        last = self._last.get(msg.arbitration_id)
        self._last[msg.arbitration_id] = msg.timestamp
        if i is None or last is None:
            return None
        gap = msg.timestamp - last
        thr = self.theta[i]
        if thr > 0.0 and 0.0 <= gap < thr:
            return Event(detector=Detector.ANOMALY, bus=msg.bus, label=Label.SUSPECTED,
                         confidence=0.8, source_tier=Tier.ADAPTIVE,
                         features={"arbitration_id": msg.arbitration_id, "gap": gap})
        return None


def local_theta(messages: Iterable[Message], ids: Sequence[int]) -> list[float]:
    """One vehicle's local estimate of the threshold vector from its own traffic."""
    gaps: dict[int, list[float]] = {aid: [] for aid in ids}
    last: dict[int, float] = {}
    for m in messages:
        aid = m.arbitration_id
        if aid in gaps:
            if aid in last:
                d = m.timestamp - last[aid]
                if d > 0:
                    gaps[aid].append(d)
            last[aid] = m.timestamp
    return [PERIOD_RATIO * statistics.median(gaps[aid]) if gaps[aid] else 0.0 for aid in ids]


# --------------------------------------------------------------------------- #
# data: shard benign traffic across an emulated fleet
# --------------------------------------------------------------------------- #
def load_fleet_data(n_vehicles: int, limit: int):
    path = CAR_HACKING / ATTACK_FILE
    if not path.exists():
        return None
    all_msgs = list(itertools.islice(CarHackingSource(path, attack_type="rpm-spoof"), limit))
    benign = [m for m in all_msgs if not m.is_attack]
    # monitored ids: those periodic enough to model, present throughout
    counts: dict[int, int] = {}
    for m in benign:
        counts[m.arbitration_id] = counts.get(m.arbitration_id, 0) + 1
    ids = sorted(aid for aid, c in counts.items() if c >= 200)
    # each vehicle observes a different contiguous stretch of driving (non-IID)
    shard = max(1, len(benign) // n_vehicles)
    shards = [benign[i * shard:(i + 1) * shard] for i in range(n_vehicles)]
    return all_msgs, benign, ids, shards


def metrics_of(theta, ids, attack_msgs, clean_msgs) -> dict[str, float]:
    """Recall on the attack capture, FPR on clean traffic, plus utility scores.

    Recall alone is gameable: a detector with absurdly large thresholds flags
    *everything* and scores recall 1.0 while being useless (E6 at small epsilon
    hits recall 1.0 at FPR 0.60). So a utility number is required.

    **balanced accuracy** = (TPR + TNR)/2 is the headline, because it is
    independent of how much of the capture is attack traffic. F1 is reported too
    but is *not* a safe guard on its own: its precision term depends on the
    attack/benign ratio of the evaluated capture, so on an all-attack slice a
    blanket detector still scores F1 ~= 1.0 while balanced accuracy correctly
    collapses to 0.5 (no discriminative power).
    """
    a = evaluate(ThresholdDetector(ids, theta), attack_msgs)
    c = evaluate(ThresholdDetector(ids, theta), clean_msgs)
    rec, prec = a.detection_rate, a.precision
    fpr = c.false_positive_rate
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"recall": rec, "fpr": fpr, "precision": prec, "f1": f1,
            "bal_acc": 0.5 * (rec + (1.0 - fpr))}


def detection_of(theta, ids, attack_msgs, clean_msgs) -> tuple[float, float]:
    m = metrics_of(theta, ids, attack_msgs, clean_msgs)
    return m["recall"], m["fpr"]


# --------------------------------------------------------------------------- #
# E5 -- poisoning: detection retained, Krum vs. averaging
# --------------------------------------------------------------------------- #
def e5(n_vehicles: int, limit: int) -> None:
    data = load_fleet_data(n_vehicles, limit)
    if data is None:
        print("\n[E5] Car-Hacking not found -- skipped")
        return
    attack_msgs, benign, ids, shards = data

    honest = [local_theta(s, ids) for s in shards]
    baseline_theta = fedavg([{"weights": h, "num_samples": 1} for h in honest])["weights"]
    base = metrics_of(baseline_theta, ids, attack_msgs, benign)
    base_det, base_fpr = base["recall"], base["fpr"]

    print(f"\n[E5] emulated fleet N={n_vehicles} on {ATTACK_FILE} "
          f"({len(ids)} monitored ids, {len(attack_msgs)} frames)")
    print(f"     clean-fleet baseline: detection={base_det:.4f} fpr={base_fpr:.4f}")

    rows = []
    for frac in (0.10, 0.20, 0.30):
        f = max(1, round(frac * n_vehicles))
        n_honest = n_vehicles - f
        # scaling attack: drag the arithmetic mean to ~0 (blind the detector)
        scale = n_honest / f
        updates = [{"weights": honest[i], "num_samples": 1} for i in range(n_honest)]
        for j in range(f):
            evil = [-scale * v for v in honest[(n_honest + j) % len(honest)]]
            updates.append({"weights": evil, "num_samples": 1})

        am = metrics_of(fedavg(updates)["weights"], ids, attack_msgs, benign)
        avg_det, avg_fpr = am["recall"], am["fpr"]
        try:
            km = metrics_of(krum(updates, f)["weights"], ids, attack_msgs, benign)
            krum_det, krum_fpr = km["recall"], km["fpr"]
            krum_cell = f"{krum_det:.4f}"
            krum_fpr_cell = f"{krum_fpr:.4f}"
        except ValueError as exc:                       # n < 2f+3
            krum_det = float("nan")
            krum_cell, krum_fpr_cell = f"n/a ({exc})", "-"

        rows.append({
            "f/n": f"{100 * f / n_vehicles:.0f}%", "f": f,
            "avg_detection": f"{avg_det:.4f}",
            "avg_retained": f"{100 * avg_det / base_det:.1f}%" if base_det else "-",
            "krum_detection": krum_cell,
            "krum_retained": (f"{100 * krum_det / base_det:.1f}%"
                              if base_det and krum_det == krum_det else "-"),
            "avg_fpr": f"{avg_fpr:.4f}", "krum_fpr": krum_fpr_cell,
        })
    _table(rows)
    print("     scaling attack: malicious clients submit -(n_honest/f) x honest theta, which")
    print("     pulls the mean to ~0; clamped thresholds then never fire (a blinded detector).")


# --------------------------------------------------------------------------- #
# E6 -- privacy: detection vs epsilon
# --------------------------------------------------------------------------- #
def e6(n_vehicles: int, limit: int, delta: float = 1e-5) -> None:
    data = load_fleet_data(n_vehicles, limit)
    if data is None:
        print("\n[E6] Car-Hacking not found -- skipped")
        return
    attack_msgs, benign, ids, shards = data
    honest = [local_theta(s, ids) for s in shards]
    clean_theta = fedavg([{"weights": h, "num_samples": 1} for h in honest])["weights"]
    base = metrics_of(clean_theta, ids, attack_msgs, benign)

    # sensitivity: clip to the typical per-coordinate threshold magnitude, so
    # epsilon is expressed relative to the quantity actually being protected
    scale = statistics.mean(t for t in clean_theta if t > 0)
    print(f"\n[E6] distributed DP (local Gaussian noise, then average over N={n_vehicles})")
    print(f"     no-DP baseline: recall={base['recall']:.4f} fpr={base['fpr']:.4f} "
          f"bal_acc={base['bal_acc']:.4f} f1={base['f1']:.4f}; "
          f"sensitivity={scale:.6f}s (mean threshold), delta={delta}")

    rows = []
    for eps in (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0):
        rng = random.Random(0)
        noisy = [add_dp_noise({"weights": h}, epsilon=eps, delta=delta,
                              sensitivity=scale, rng=rng)["weights"] for h in honest]
        agg = fedavg([{"weights": w, "num_samples": 1} for w in noisy])["weights"]
        m = metrics_of(agg, ids, attack_msgs, benign)
        sigma_client = add_dp_noise({"weights": [0.0]}, epsilon=eps, delta=delta,
                                    sensitivity=scale, rng=random.Random(0))["dp"]["sigma"]
        rows.append({"epsilon": eps,
                     "sigma_agg_s": f"{sigma_client / (n_vehicles ** 0.5):.5f}",
                     "recall": f"{m['recall']:.4f}",
                     "fpr": f"{m['fpr']:.4f}",
                     "bal_acc": f"{m['bal_acc']:.4f}",
                     "f1": f"{m['f1']:.4f}",
                     "utility_retained": (f"{100 * m['bal_acc'] / base['bal_acc']:.1f}%"
                                          if base["bal_acc"] else "-")})
    _table(rows)
    print("     Read recall WITH fpr: at eps<=1 the noise inflates thresholds so the detector")
    print("     flags nearly everything -- recall ~1.0 but fpr up to 0.60, i.e. useless.")
    print("     bal_acc = (TPR+TNR)/2 is the honest utility curve (class-balance independent)")
    print("     and it IS monotone in epsilon; utility_retained is relative to no-DP.")
    print("     Averaging N local noises shrinks the aggregate sigma by sqrt(N): privacy is")
    print("     paid per-vehicle, utility is recovered by the fleet.")


# --------------------------------------------------------------------------- #
# E7 -- time to fleet immunity vs fleet size
# --------------------------------------------------------------------------- #
def _rule() -> Antibody:
    return Antibody(attack_class="rpm-spoof", artifact_type=ArtifactType.RULE,
                    artifact={"kind": "id_match", "arbitration_id": 0x316})


def e7(limit: int, quorum: int = 3, p_encounter: float = 0.10, trials: int = 200) -> None:
    """Rounds until a validated+signed antibody is active fleet-wide.

    Uses the REAL fleet components: attestation gate, candidate quorum,
    validation lab, HSM signing, OTA verify+activate. Only the per-round
    exposure of each vehicle to the attack is stochastic.
    """
    ca, maker = SimulatedHSM(), SimulatedHSM()
    fw = measure_firmware(b"ecu-fw-v1")
    lab = ValidationLab(
        normal_corpus=[Message(arbitration_id=0x100) for _ in range(200)],
        attack_corpus=[Message(arbitration_id=0x316) for _ in range(40)],
        min_detection=0.9,
    )
    ota = OTADistributor(maker)

    print(f"\n[E7] time-to-fleet-immunity (quorum={quorum}, p_encounter={p_encounter}/round/"
          f"vehicle, {trials} trials)")
    rows = []
    for N in (5, 10, 20, 50, 100):
        gate = AttestationGate(ca, trusted_measurements={fw})
        clients = [FLClient(f"v{i}", fw, gate.issue_certificate(f"v{i}")) for i in range(N)]
        for c in clients:                                  # admission happens once
            gate.admit(c.vehicle_id, c.attestation_token, c.certificate)

        durations, wall = [], []
        for t in range(trials):
            rng = random.Random(1000 + t)
            reporters: set[str] = set()
            rounds = 0
            t0 = time.perf_counter()
            while True:
                rounds += 1
                for c in clients:
                    if rng.random() < p_encounter:
                        reporters.add(c.vehicle_id)
                if len(reporters) >= quorum:
                    break
                if rounds > 10_000:                        # safety valve
                    break
            # --- real pipeline: quorum -> validate -> sign -> OTA -> activate
            updates = [FLClient(v, fw, gate.issue_certificate(v)).make_update(
                candidates=[_rule()]) for v in sorted(reporters)]
            promoted = tally_candidates(updates, quorum=quorum)
            assert promoted, "quorum reached but nothing promoted"
            cand = lab.validate(promoted[0])
            assert cand.validation["server_side_passed"]
            ota.sign(cand)
            shadows = [ShadowRunner() for _ in range(N)]
            applied = all(ota.verify_and_apply(cand, s) for s in shadows)
            assert applied and all(s.is_active(cand.antibody_id) for s in shadows)
            wall.append((time.perf_counter() - t0) * 1000.0)
            # +4 fixed rounds: quorum, validation, signing, OTA distribution
            durations.append(rounds + 4)

        durations.sort()
        rows.append({"N": N,
                     "rounds_to_quorum_p50": statistics.median(d - 4 for d in durations),
                     "T_immunity_rounds_mean": f"{statistics.mean(durations):.2f}",
                     "T_immunity_p50": durations[len(durations) // 2],
                     "T_immunity_p95": durations[int(0.95 * (len(durations) - 1))],
                     "backend_ms_mean": f"{statistics.mean(wall):.2f}"})
    _table(rows)
    print("     T_immunity = rounds to quorum + 4 fixed backend rounds (quorum, validation,")
    print("     signing, OTA). Larger fleets reach quorum sooner: herd immunity is a")
    print("     function of exposure count, so T falls with N and then floors at the")
    print("     irreducible backend cost.")


# --------------------------------------------------------------------------- #
def _table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("     " + "  ".join(c.ljust(w[c]) for c in cols))
    print("     " + "  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("     " + "  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--limit", type=int, default=300_000)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    print("=" * 78)
    print("VIS -- fleet-layer experiments over an emulated fleet (E5, E6, E7)")
    print("=" * 78)
    t0 = time.time()
    e5(args.vehicles, args.limit)
    e6(args.vehicles, args.limit)
    e7(args.limit, trials=args.trials)
    print(f"\n{'-' * 78}\ntotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
