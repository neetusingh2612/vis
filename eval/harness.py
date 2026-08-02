"""Evaluation harness -- the centre of gravity for the project.

Feeds labelled traffic through a detector and computes detection rate, FPR, and
latency. Build everything to report into this so you always know where you stand
against the design goals. Populates the Section 9 result tables.
"""
from __future__ import annotations

import time

from vis.shared.detector import BaseDetector
from vis.shared.state import VehicleState

from metrics import Metrics


def evaluate(detector: BaseDetector, source, state: VehicleState | None = None) -> Metrics:
    state = state or VehicleState()
    m = Metrics(latencies_ms=[])
    detector.reset()
    for msg in source:
        t0 = time.perf_counter()
        event = detector.inspect(msg, state)
        m.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        flagged = event is not None
        if msg.is_attack and flagged:
            m.tp += 1
        elif msg.is_attack and not flagged:
            m.fn += 1
        elif not msg.is_attack and flagged:
            m.fp += 1
        else:
            m.tn += 1
    return m


if __name__ == "__main__":
    from pathlib import Path

    from vis.adaptive.anomaly_detector import AnomalyDetector
    from vis.adaptive.v2x_misbehavior import V2XMisbehavior
    from vis.reflex.fingerprinting import Fingerprinting
    from vis.reflex.hal import Ecu, SimulatedCanFdBench
    from vis.reflex.traffic_monitor import TrafficMonitor
    from vis.shared.datasets import CarHackingSource, VeReMiSource
    from vis.shared.traffic import SyntheticSource

    repo_root = Path(__file__).resolve().parents[1]

    # 1) Reflex smoke run: traffic monitor vs. synthetic flood (always available).
    synth = SyntheticSource(n=2000, flood_at=1000, flood_len=200)
    print("synthetic flood   :", evaluate(TrafficMonitor(), synth).summary())

    # 2) Adaptive: anomaly detector vs. synthetic flood. It is observe-only and
    # must learn 'self' from a clean corpus first (fit), *then* be evaluated --
    # the harness only resets the live cursor, it never fits for you.
    anomaly = AnomalyDetector()
    anomaly.fit(SyntheticSource(n=2000, flood_at=None))            # clean self corpus
    flood = SyntheticSource(n=2000, flood_at=1000, flood_len=200)
    print("anomaly (synth)   :", evaluate(anomaly, flood).summary())

    # 3) Reflex on a real dataset: HCRL Car-Hacking DoS -- runs only if the
    # capture has been downloaded (datasets/ is git-ignored; see datasets/README).
    dos_csv = repo_root / "datasets" / "car-hacking" / "DoS_dataset.csv"
    if dos_csv.exists():
        # NB: thresholds here are illustrative and must be calibrated against
        # the recorded benign baseline -- that calibration is part of the eval
        # work, not a constant. Rough Car-Hacking numbers: benign ~2000 msg/s
        # => ~20 frames per 10ms window; the DoS flood (injected id 0x000)
        # pushes the aggregate to ~50/window, so a threshold of ~35 brackets
        # them. Tune against your own capture's baseline.
        src = CarHackingSource(dos_csv, attack_type="dos")
        print("car-hacking DoS   :", evaluate(TrafficMonitor(window=0.01, rate_threshold=35), src).summary())
    else:
        print(f"car-hacking DoS   : skipped -- place dataset at {dos_csv}")

    # 4) V2X: misbehavior detector vs. VeReMi -- runs only if a receiver log is
    # present. Drop a JSON-lines log at the path below (+ optional GroundTruth);
    # see datasets/README.md for the format.
    veremi_log = repo_root / "datasets" / "veremi" / "receiver.json"
    veremi_gt = repo_root / "datasets" / "veremi" / "GroundTruth.json"
    if veremi_log.exists():
        src = VeReMiSource(veremi_log, ground_truth=veremi_gt if veremi_gt.exists() else None)
        print("veremi misbehavior:", evaluate(V2XMisbehavior(), src).summary())
    else:
        print(f"veremi misbehavior: skipped -- place log at {veremi_log}")

    # 5) Phase-4 simulated CAN-FD bench (no hardware needed): clock-skew
    # fingerprinting catches a masquerade (foreign ECU clock on a legit id). The
    # p99 latency line doubles as the E2 software proxy (reflex compute/frame).
    def _bench_ecus():
        return [Ecu("engine", {0x111: 0.01}, clock_skew_ppm=+50),
                Ecu("brake", {0x222: 0.02}, clock_skew_ppm=-30)]

    fp = Fingerprinting(skew_threshold_ppm=200.0)
    fp.enrol_clock_skew(SimulatedCanFdBench(_bench_ecus(), duration_s=2.0))
    attacked = SimulatedCanFdBench(_bench_ecus(), duration_s=2.0)
    attacked.add_masquerade(0x111, attacker_skew_ppm=-600, start=1.0, end=2.0)
    print("canfd masquerade  :", evaluate(fp, attacked).summary())
