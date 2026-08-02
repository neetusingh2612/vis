"""Fleet-layer emulation (E5/E6/E7) — properties that must hold regardless of
whether the real datasets are present."""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from run_fleet import (  # noqa: E402
    PERIOD_RATIO,
    ThresholdDetector,
    local_theta,
    metrics_of,
)
from vis.fleet.fl_server import fedavg, krum  # noqa: E402
from vis.shared.contracts import Message  # noqa: E402
from vis.shared.state import VehicleState  # noqa: E402


def _periodic(aid, period, n, start=0.0, attack=False):
    return [Message(arbitration_id=aid, timestamp=start + i * period, is_attack=attack)
            for i in range(n)]


def test_local_theta_recovers_half_the_period():
    msgs = _periodic(0x100, 0.01, 500)
    (theta,) = local_theta(msgs, [0x100])
    assert theta == pytest.approx(PERIOD_RATIO * 0.01, rel=1e-6)


def test_threshold_detector_flags_only_compressed_gaps():
    det = ThresholdDetector([0x100], [0.005])       # min plausible gap 5 ms
    state = VehicleState()
    # on-cadence traffic (10 ms) is clean
    assert all(det.inspect(m, state) is None for m in _periodic(0x100, 0.01, 50))
    # injected frames at 1 ms are flagged
    det.reset()
    flags = [det.inspect(m, state) for m in _periodic(0x100, 0.001, 50)]
    assert sum(f is not None for f in flags) > 40


def test_clamped_nonpositive_thresholds_blind_the_detector():
    # this is the mechanism E5 measures: a mean dragged to <=0 never fires
    det = ThresholdDetector([0x100], [-1.0])
    state = VehicleState()
    assert all(det.inspect(m, state) is None for m in _periodic(0x100, 0.0001, 100))


def test_scaling_attack_destroys_fedavg_but_not_krum():
    honest = [[0.005, 0.010] for _ in range(14)]
    f = 6
    scale = len(honest) / f
    updates = [{"weights": h, "num_samples": 1} for h in honest]
    for _ in range(f):
        updates.append({"weights": [-scale * 0.005, -scale * 0.010], "num_samples": 1})

    avg = fedavg(updates)["weights"]
    assert min(avg) <= 0.0                    # mean dragged to/below zero -> blinded
    kru = krum(updates, f)["weights"]
    assert kru == [0.005, 0.010]              # an honest vector survives


def test_balanced_accuracy_exposes_that_recall_alone_is_gameable():
    """A blanket detector scores recall ~1.0 while being useless.

    balanced accuracy must collapse to ~0.5 (no discriminative power). F1 must
    NOT be trusted for this: on an all-attack slice its precision term is 1.0,
    so F1 stays ~1.0 -- which is exactly why E6 reports bal_acc as the headline.
    """
    attack = _periodic(0x100, 0.01, 100, attack=True)
    clean = _periodic(0x100, 0.01, 100)
    blanket = metrics_of([10.0], [0x100], attack, clean)

    assert blanket["recall"] > 0.9 and blanket["fpr"] > 0.9
    assert blanket["bal_acc"] == pytest.approx(0.5, abs=0.05)   # no skill
    assert blanket["f1"] > 0.9                                  # F1 is fooled here

    # a genuinely discriminating threshold scores well on bal_acc
    good = metrics_of([0.005], [0x100], _periodic(0x100, 0.001, 100, attack=True), clean)
    assert good["bal_acc"] > 0.9
