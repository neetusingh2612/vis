"""Phase-5 integration suite: every experiment (E1-E7) and the Section 8
adversarial claims must pass. This keeps the end-to-end story a regression test."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from experiments import (  # noqa: E402
    adversarial_claims,
    e1_reflex_detection,
    e2_latency,
    e3_masquerade,
    e4_adaptive,
    e5_byzantine_robustness,
    e6_privacy_accuracy,
    e7_fleet_immunity,
    run_all,
)


def test_e1_reflex_detection():
    assert e1_reflex_detection().passed


def test_e2_latency():
    assert e2_latency().passed


def test_e3_masquerade():
    assert e3_masquerade().passed


def test_e4_adaptive():
    assert e4_adaptive().passed


def test_e5_byzantine_robustness():
    r = e5_byzantine_robustness()
    assert r.passed
    # FedAvg error must grow with the poisoned fraction while Krum stays at 0
    fed = [row["fedavg_err"] for row in r.rows]
    assert fed == sorted(fed) and fed[-1] > fed[0]


def test_e6_privacy_accuracy():
    assert e6_privacy_accuracy().passed


def test_e7_fleet_immunity():
    assert e7_fleet_immunity().passed


def test_section8_adversarial_claims_all_hold():
    res = adversarial_claims()
    failing = [r["claim"] for r in res.rows if not r["holds"]]
    assert res.passed, f"claims that did not hold: {failing}"


def test_run_all_reports_every_experiment_passing():
    results = run_all()
    assert len(results) == 8
    assert all(r.passed for r in results)
