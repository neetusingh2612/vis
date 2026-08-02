"""Phase 5 -- integration & evaluation (experiments E1-E7 + Section 8 claims).

Runs the whole VIS stack end-to-end against synthetic/simulated data (no
downloads, no hardware) and reports Section-9-style result tables. Each
experiment returns an :class:`ExperimentResult` with a pass/fail verdict so the
suite doubles as a regression test.

  E1  Reflex detection efficacy (flood, across sources)
  E2  Real-time latency (reflex compute/frame -- software proxy)
  E3  Masquerade detection (clock-skew + voltage fingerprinting)
  E4  Adaptive layer (observe-only anomaly + correlation false-alarm suppression)
  E5  Byzantine robustness (Krum vs FedAvg as % poisoned clients rises)
  E6  Privacy vs. accuracy (DP epsilon sweep)
  E7  Time-to-fleet-immunity (local detect -> quorum -> validate -> sign -> act)

`adversarial_claims()` checks the Section 8 security claims directly.
Run `python eval/experiments.py` for the full report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from harness import evaluate

from vis.adaptive.anomaly_detector import AnomalyDetector
from vis.adaptive.antibody_generator import AntibodyGenerator, matcher_for_artifact
from vis.adaptive.correlation_engine import CorrelationEngine, Incident
from vis.adaptive.llm_assistant import LLMAssistant
from vis.adaptive.negative_selection import NegativeSelection
from vis.adaptive.shadow_runner import ShadowRunner
from vis.fleet.attestation_gate import AttestationGate
from vis.fleet.fl_client import FLClient
from vis.fleet.fl_server import fedavg, krum, tally_candidates
from vis.fleet.ota_distributor import OTADistributor
from vis.fleet.secure_agg import add_dp_noise, apply_pairwise_masks, secure_sum
from vis.fleet.validation_lab import ValidationLab
from vis.reflex.fingerprinting import Fingerprinting
from vis.reflex.hal import Ecu, PhysicalSample, SimulatedCanFdBench
from vis.reflex.traffic_monitor import TrafficMonitor
from vis.shared.contracts import (
    Antibody,
    ArtifactType,
    Bus,
    Detector,
    Event,
    Label,
    Message,
    ResponseAction,
    Tier,
)
from vis.shared.keystore import SimulatedHSM, measure_firmware
from vis.shared.state import VehicleState
from vis.shared.traffic import SyntheticSource

import random


@dataclass
class ExperimentResult:
    eid: str
    name: str
    rows: list[dict] = field(default_factory=list)
    passed: bool = True
    note: str = ""

    def render(self) -> str:
        head = f"[{self.eid}] {self.name}  --  {'PASS' if self.passed else 'FAIL'}"
        if not self.rows:
            return head
        cols = list(self.rows[0].keys())
        widths = {c: max(len(c), *(len(str(r[c])) for r in self.rows)) for c in cols}
        line = "  ".join(c.ljust(widths[c]) for c in cols)
        sep = "  ".join("-" * widths[c] for c in cols)
        body = "\n".join("  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in self.rows)
        note = f"\n  {self.note}" if self.note else ""
        return f"{head}\n  {line}\n  {sep}\n" + "\n".join("  " + b for b in body.splitlines()) + note


def _ecus() -> list[Ecu]:
    return [Ecu("engine", {0x111: 0.01}, clock_skew_ppm=+50, voltage_features=(1.0, 2.0, 3.0)),
            Ecu("brake", {0x222: 0.02}, clock_skew_ppm=-30, voltage_features=(4.0, 5.0, 6.0))]


def _norm(vec) -> float:
    return math.hypot(*vec) if vec else 0.0


# --------------------------------------------------------------------------- #
# E1 -- reflex detection efficacy
# --------------------------------------------------------------------------- #
def e1_reflex_detection() -> ExperimentResult:
    # Detection is measured under attack; the false-alarm rate is measured on
    # attack-free traffic (operational FPR). A bus-level rate monitor flags every
    # frame in a high-rate window, so benign frames riding inside an active flood
    # are collateral -- not a fair FPR sample. This is the honest methodology.
    cases = [
        ("synthetic flood", lambda: TrafficMonitor(),
         SyntheticSource(n=2000, flood_at=1000, flood_len=200),
         SyntheticSource(n=2000, flood_at=None)),
        ("canfd flood", lambda: TrafficMonitor(window=0.01, rate_threshold=25),
         _flood_bench(), SimulatedCanFdBench(_ecus(), duration_s=1.0)),
    ]
    rows, ok = [], True
    for name, make_det, attack_src, clean_src in cases:
        detection = evaluate(make_det(), attack_src).detection_rate
        fpr = evaluate(make_det(), clean_src).false_positive_rate
        rows.append({"scenario": name, "detection": round(detection, 3),
                     "fpr_clean": round(fpr, 4)})
        ok = ok and detection > 0.5 and fpr < 0.05
    return ExperimentResult("E1", "Reflex detection efficacy", rows, ok,
                            note="detection under attack; FPR on attack-free traffic")


def _flood_bench() -> SimulatedCanFdBench:
    bench = SimulatedCanFdBench(_ecus(), duration_s=1.0)
    bench.add_flood(0x000, period=0.0002, start=0.5, end=1.0)
    return bench


# --------------------------------------------------------------------------- #
# E2 -- real-time latency (software proxy)
# --------------------------------------------------------------------------- #
def e2_latency(budget_ms: float = 1.0) -> ExperimentResult:
    m = evaluate(TrafficMonitor(window=0.01, rate_threshold=25), _flood_bench())
    rows = [{"metric": "mean_ms", "value": round(m.mean_latency_ms, 5)},
            {"metric": "p99_ms", "value": round(m.p99_latency_ms, 5)},
            {"metric": "budget_ms", "value": budget_ms}]
    passed = m.p99_latency_ms < budget_ms
    return ExperimentResult("E2", "Reflex latency (per-frame compute)", rows, passed,
                            note="software proxy for on-ECU E2; bench measures real-time")


# --------------------------------------------------------------------------- #
# E3 -- masquerade detection (fingerprinting)
# --------------------------------------------------------------------------- #
def e3_masquerade() -> ExperimentResult:
    # clock-skew path
    skew = Fingerprinting(skew_threshold_ppm=200.0)
    skew.enrol_clock_skew(SimulatedCanFdBench(_ecus(), duration_s=2.0))
    attacked = SimulatedCanFdBench(_ecus(), duration_s=2.0)
    attacked.add_masquerade(0x111, attacker_skew_ppm=-600, start=1.0, end=2.0)
    ms = evaluate(skew, attacked)

    # voltage path
    vbench = SimulatedCanFdBench(_ecus(), duration_s=0.5)
    vbench.add_masquerade(0x111, attacker_skew_ppm=0.0, start=0.0, end=0.5,
                          attacker_features=(9.0, 9.0, 9.0))
    volt = Fingerprinting(sampler=vbench.voltage_sampler(seed=1), voltage_tol=0.5)
    for ecu in _ecus():
        for aid in ecu.sends:
            volt.enrol(aid, PhysicalSample(aid, 0.0, ecu.voltage_features))
    mv = evaluate(volt, vbench)

    rows = [
        {"method": "clock_skew", "detection": round(ms.detection_rate, 3),
         "fpr": round(ms.false_positive_rate, 4)},
        {"method": "voltage", "detection": round(mv.detection_rate, 3),
         "fpr": round(mv.false_positive_rate, 4)},
    ]
    passed = ms.detection_rate > 0.2 and ms.false_positive_rate < 0.05 \
        and mv.detection_rate > 0.9 and mv.false_positive_rate < 0.05
    return ExperimentResult("E3", "Masquerade detection (fingerprinting)", rows, passed)


# --------------------------------------------------------------------------- #
# E4 -- adaptive layer (observe-only anomaly + correlation suppression)
# --------------------------------------------------------------------------- #
def e4_adaptive() -> ExperimentResult:
    det = AnomalyDetector()
    det.fit(SyntheticSource(n=2000, flood_at=None))
    m = evaluate(det, SyntheticSource(n=2000, flood_at=1000, flood_len=200))

    # correlation: a lone anomaly is suppressed; two agreeing ones confirm
    eng = CorrelationEngine(window_s=0.05, min_events=2)

    def _ev(aid, ts):
        return Event(detector=Detector.ANOMALY, bus=Bus.CAN, label=Label.SUSPECTED,
                     source_tier=Tier.ADAPTIVE, timestamp=ts, features={"arbitration_id": aid})
    lone_suppressed = eng.submit(_ev(0x000, 0.0)) is None
    corroborated = eng.submit(_ev(0x000, 0.01)) is not None

    rows = [
        {"check": "anomaly detection", "value": round(m.detection_rate, 3)},
        {"check": "anomaly fpr", "value": round(m.false_positive_rate, 4)},
        {"check": "lone anomaly suppressed", "value": lone_suppressed},
        {"check": "corroborated confirmed", "value": corroborated},
    ]
    passed = (m.detection_rate > 0.5 and m.false_positive_rate < 0.05
              and lone_suppressed and corroborated)
    return ExperimentResult("E4", "Adaptive: anomaly + correlation", rows, passed)


# --------------------------------------------------------------------------- #
# E5 -- Byzantine robustness (Krum vs FedAvg)
# --------------------------------------------------------------------------- #
def e5_byzantine_robustness() -> ExperimentResult:
    n, dim = 15, 10
    honest = [0.0] * dim
    poison = [10.0] * dim
    rows, ok = [], True
    for f in range(0, 7):                      # n >= 2f+3 holds up to f=6
        updates = ([{"weights": list(honest)} for _ in range(n - f)]
                   + [{"weights": list(poison)} for _ in range(f)])
        fed_err = _norm(fedavg(updates)["weights"])
        krum_err = _norm(krum(updates, f)["weights"])
        rows.append({"poison_%": round(100 * f / n, 1), "f": f,
                     "fedavg_err": round(fed_err, 2), "krum_err": round(krum_err, 2)})
        if f >= 1:
            ok = ok and krum_err < 1e-6 and fed_err > krum_err
    return ExperimentResult("E5", "Byzantine robustness (Krum vs FedAvg)", rows, ok,
                            note="Krum stays on the honest target while FedAvg is dragged")


# --------------------------------------------------------------------------- #
# E6 -- privacy vs. accuracy (DP epsilon sweep)
# --------------------------------------------------------------------------- #
def e6_privacy_accuracy() -> ExperimentResult:
    dim = 500
    true = {"weights": [0.0] * dim}
    rows = []
    errors = []
    for eps in (0.1, 0.5, 1.0, 5.0, 10.0):
        noised = add_dp_noise(true, epsilon=eps, sensitivity=1.0, rng=random.Random(0))
        rms = math.sqrt(sum(x * x for x in noised["weights"]) / dim)
        rows.append({"epsilon": eps, "sigma": round(noised["dp"]["sigma"], 3),
                     "rms_error": round(rms, 3)})
        errors.append(rms)
    # stronger privacy (smaller epsilon) must cost more accuracy
    passed = all(a >= b for a, b in zip(errors, errors[1:]))
    return ExperimentResult("E6", "Privacy vs. accuracy (DP)", rows, passed,
                            note="smaller epsilon -> more noise -> lower accuracy")


# --------------------------------------------------------------------------- #
# E7 -- time-to-fleet-immunity
# --------------------------------------------------------------------------- #
def e7_fleet_immunity() -> ExperimentResult:
    ca, maker = SimulatedHSM(), SimulatedHSM()
    good_fw = measure_firmware(b"ecu-fw-v1")
    gate = AttestationGate(ca, trusted_measurements={good_fw})

    def _rule():
        return Antibody("dos", ArtifactType.RULE, {"kind": "id_match", "arbitration_id": 0x000})

    clients = [FLClient(f"veh{i}", good_fw, gate.issue_certificate(f"veh{i}")) for i in range(3)]
    updates = [c.make_update(candidates=[_rule()]) for c in clients]

    stages, rounds = [], 0
    admitted = [u for u in updates
                if gate.admit(u["vehicle_id"], u["attestation_token"], u["certificate"])]
    stages.append(("admit", len(admitted) == 3))
    rounds += 1

    promoted = tally_candidates(admitted, quorum=2)
    stages.append(("quorum", len(promoted) == 1))
    rounds += 1
    candidate = promoted[0]

    lab = ValidationLab(normal_corpus=[Message(arbitration_id=0x100) for _ in range(100)],
                        attack_corpus=[Message(arbitration_id=0x000) for _ in range(20)],
                        min_detection=0.9)
    lab.validate(candidate)
    stages.append(("validate", candidate.validation["server_side_passed"]))
    rounds += 1

    ota = OTADistributor(maker)
    ota.sign(candidate)
    stages.append(("sign", candidate.is_validated))
    rounds += 1

    shadow = ShadowRunner()
    activated = ota.verify_and_apply(candidate, shadow)
    stages.append(("verify+activate", activated and shadow.is_active(candidate.antibody_id)))
    rounds += 1

    rows = [{"stage": s, "ok": ok} for s, ok in stages]
    rows.append({"stage": "rounds_to_immunity", "ok": rounds})
    passed = all(ok for _, ok in stages)
    return ExperimentResult("E7", "Time-to-fleet-immunity", rows, passed,
                            note="candidate graduates shadow->acting only after the full gauntlet")


# --------------------------------------------------------------------------- #
# Section 8 -- adversarial claim checks
# --------------------------------------------------------------------------- #
def adversarial_claims() -> ExperimentResult:
    rows = []

    def claim(text, ok):
        rows.append({"claim": text, "holds": bool(ok)})

    # C1 Sybil/forged vehicles are blocked at admission
    gate = AttestationGate(b"ca", trusted_measurements={"good-fw"})
    claim("C1 forged-cert vehicle is rejected at the gate",
          gate.admit("evil", "good-fw", "forged") is False
          and gate.admit("v1", "good-fw", gate.issue_certificate("v1")) is True)

    # C2 the adaptive layer never acts (observe-only): anomalies are SUSPECTED, no response
    det = AnomalyDetector()
    det.fit(SyntheticSource(n=1000, flood_at=None))
    det.reset()
    evs = [det.inspect(m, VehicleState())
           for m in SyntheticSource(n=1500, flood_at=1000, flood_len=200)]
    evs = [e for e in evs if e is not None]
    claim("C2 adaptive emits only SUSPECTED with no response action",
          bool(evs) and all(e.label == Label.SUSPECTED
                            and e.response_taken == ResponseAction.NONE for e in evs))

    # C3 learn-fast/act-slow: a candidate cannot act until validated AND signed
    ab = Antibody("dos", ArtifactType.RULE, {"kind": "id_match", "arbitration_id": 0x000})
    ota = OTADistributor(b"maker")
    before = ota.verify_and_apply(ab, ShadowRunner())          # unvalidated -> no
    ab.validation = {"server_side_passed": True}
    ota.sign(ab)
    after = ota.verify_and_apply(ab, ShadowRunner())           # validated+signed -> yes
    claim("C3 candidate acts only after fleet validation + signature",
          before is False and after is True)

    # C4 tampered OTA never activates
    ab2 = Antibody("dos", ArtifactType.RULE, {"kind": "id_match", "arbitration_id": 0x000})
    ab2.validation = {"server_side_passed": True}
    ota.sign(ab2)
    ab2.artifact["arbitration_id"] = 0x111                      # tamper post-signature
    claim("C4 tampered antibody fails signature verification",
          ota.verify_and_apply(ab2, ShadowRunner()) is False)

    # C5 Byzantine: f poisoned clients cannot move the robust aggregate
    updates = ([{"weights": [0.0, 0.0]} for _ in range(12)]
               + [{"weights": [50.0, 50.0]} for _ in range(3)])
    claim("C5 Krum ignores poisoned updates (f=3 of 15)",
          _norm(krum(updates, 3)["weights"]) < 1e-6)

    # C6 the LLM has no authority: an adversarial suggestion must pass negative selection
    neg = NegativeSelection([Message(arbitration_id=0x100)])    # 0x100 is known-good
    gen = AntibodyGenerator(neg)
    llm = LLMAssistant()
    evil_suggestion = llm.suggest_candidate(Incident("masq", events=[_anom(0x100)]))
    self_id_rejected = neg.passes(matcher_for_artifact(evil_suggestion)) is False
    good_suggestion = llm.suggest_candidate(Incident("dos", events=[_anom(0x000)]))
    legit_accepted = gen.synthesize(Incident("dos", events=[_anom(0x000)])) is not None
    claim("C6 LLM suggestion blaming a legit id is rejected by self-tolerance",
          self_id_rejected and legit_accepted and good_suggestion["advisory"] is True)

    # C7 masquerade with a foreign clock is caught by fingerprinting
    fp = Fingerprinting(skew_threshold_ppm=200.0)
    fp.enrol_clock_skew(SimulatedCanFdBench(_ecus(), duration_s=2.0))
    att = SimulatedCanFdBench(_ecus(), duration_s=2.0)
    att.add_masquerade(0x111, attacker_skew_ppm=-600, start=1.0, end=2.0)
    claim("C7 stolen-key masquerade caught by clock-skew fingerprint",
          evaluate(fp, att).detection_rate > 0.2)

    # C8 secure aggregation hides individual updates but recovers the exact sum
    ups = [{"weights": [1.0, 2.0]}, {"weights": [3.0, 4.0]}, {"weights": [5.0, 6.0]}]
    masked = apply_pairwise_masks(ups, seed=3)
    recovered = secure_sum(masked)["weights"]
    claim("C8 secure-agg hides contributions yet recovers the true sum",
          masked[0]["weights"] != ups[0]["weights"]
          and all(abs(a - b) < 1e-6 for a, b in zip(recovered, [9.0, 12.0])))

    passed = all(r["holds"] for r in rows)
    return ExperimentResult("S8", "Adversarial claim checks", rows, passed)


def _anom(aid):
    return Event(detector=Detector.ANOMALY, bus=Bus.CAN, label=Label.SUSPECTED,
                 source_tier=Tier.ADAPTIVE, features={"arbitration_id": aid})


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
ALL_EXPERIMENTS = [e1_reflex_detection, e2_latency, e3_masquerade, e4_adaptive,
                   e5_byzantine_robustness, e6_privacy_accuracy, e7_fleet_immunity,
                   adversarial_claims]


def run_all() -> list[ExperimentResult]:
    return [fn() for fn in ALL_EXPERIMENTS]


if __name__ == "__main__":
    print("=" * 72)
    print("VIS -- Phase 5 integration & evaluation (E1-E7 + Section 8 claims)")
    print("=" * 72)
    results = run_all()
    for r in results:
        print()
        print(r.render())
    print()
    print("-" * 72)
    n_pass = sum(1 for r in results if r.passed)
    verdict = "ALL PASS" if n_pass == len(results) else f"{n_pass}/{len(results)} PASS"
    print(f"summary: {verdict}")
