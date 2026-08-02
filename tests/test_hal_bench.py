"""Phase-4 simulated CAN-FD bench: full reflex pipeline on multi-ECU timing,
plus an E2 latency sanity check."""
import sys
import pathlib

from vis.reflex.fingerprinting import Fingerprinting
from vis.reflex.hal import Ecu, PhysicalSample, SimulatedCanFdBench
from vis.reflex.traffic_monitor import TrafficMonitor
from vis.shared.contracts import Bus

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from harness import evaluate  # noqa: E402


def _ecus():
    return [
        Ecu("engine", {0x111: 0.01}, clock_skew_ppm=+50, voltage_features=(1.0, 2.0, 3.0)),
        Ecu("brake", {0x222: 0.02}, clock_skew_ppm=-30, voltage_features=(4.0, 5.0, 6.0)),
    ]


def test_bench_streams_time_ordered_canfd_frames():
    bench = SimulatedCanFdBench(_ecus(), duration_s=0.5)
    msgs = list(bench)
    assert len(msgs) > 0
    assert all(m.bus == Bus.CAN_FD for m in msgs)
    assert [m.timestamp for m in msgs] == sorted(m.timestamp for m in msgs)


def test_bench_clock_skew_masquerade_detected_end_to_end():
    fp = Fingerprinting(skew_threshold_ppm=200.0, window=40, min_samples=20)
    fp.enrol_clock_skew(SimulatedCanFdBench(_ecus(), duration_s=2.0))   # clean enrolment

    attacked = SimulatedCanFdBench(_ecus(), duration_s=2.0)
    # attacker sends the engine id from a foreign clock for the 2nd half
    attacked.add_masquerade(0x111, attacker_skew_ppm=-600, start=1.0, end=2.0)

    m = evaluate(fp, attacked)
    assert m.detection_rate > 0.2          # masquerade frames are flagged
    assert m.false_positive_rate < 0.05    # genuine ECUs are not
    # E2 (software proxy): reflex per-frame compute is far under a CAN-frame budget
    assert m.p99_latency_ms < 1.0


def test_bench_voltage_masquerade_detected_end_to_end():
    bench = SimulatedCanFdBench(_ecus(), duration_s=0.5)
    bench.add_masquerade(0x111, attacker_skew_ppm=0.0, start=0.0, end=0.5,
                         attacker_features=(9.0, 9.0, 9.0), suspend_victim=True)

    fp = Fingerprinting(sampler=bench.voltage_sampler(seed=1), voltage_tol=0.5)
    for ecu in _ecus():
        for aid in ecu.sends:
            fp.enrol(aid, PhysicalSample(aid, 0.0, ecu.voltage_features))

    m = evaluate(fp, bench)
    assert m.detection_rate > 0.9          # foreign transmitter voltage stands out
    assert m.false_positive_rate < 0.05


def test_bench_flood_is_visible_to_traffic_monitor():
    bench = SimulatedCanFdBench(_ecus(), duration_s=1.0)
    bench.add_flood(0x000, period=0.0002, start=0.5, end=1.0)   # ~5000 msg/s burst
    m = evaluate(TrafficMonitor(window=0.01, rate_threshold=25), bench)
    assert m.detection_rate > 0.5

    # the injected frames are labelled attack ground-truth
    attack_frames = [msg for msg in bench if msg.is_attack]
    assert attack_frames and all(msg.attack_type == "dos" for msg in attack_frames)
