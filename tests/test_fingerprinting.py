"""Phase-4 (software-portable) sender fingerprinting: clock-skew + voltage
masquerade detection. Hardware capture itself is bench-only; these exercise the
detector logic on simulated physical signals."""
from vis.reflex.fingerprinting import Fingerprinting
from vis.reflex.hal import PhysicalSample, SimulatedVoltageSampler
from vis.shared.contracts import Detector, Label, Message
from vis.shared.state import VehicleState


def _periodic(aid, period, n, start=0.0, attack=False, attack_type=None):
    """Emit n frames for `aid` at a fixed period (period encodes clock rate)."""
    t = start
    for _ in range(n):
        yield Message(arbitration_id=aid, timestamp=t, is_attack=attack, attack_type=attack_type)
        t += period


# --------------------------------------------------------------------------- #
# clock-skew fingerprinting (timestamp-only -- runs on recorded data)
# --------------------------------------------------------------------------- #
def test_clock_skew_flags_masquerade_from_a_different_clock():
    # genuine ECU for id 0x11 runs ~ +50 ppm fast: period 0.1 * (1 + 50e-6)
    genuine_period = 0.1 * (1 + 50e-6)
    det = Fingerprinting(skew_threshold_ppm=200.0, window=40, min_samples=20)
    det.enrol_clock_skew(_periodic(0x11, genuine_period, n=500))

    state = VehicleState()
    det.reset()

    # genuine traffic raises no alarm
    flags = [det.inspect(m, state) for m in _periodic(0x11, genuine_period, n=300)]
    assert all(ev is None for ev in flags)

    # masquerade ECU sends id 0x11 at its own clock (~ -400 ppm) -> skew mismatch
    det.reset()
    attacker_period = 0.1 * (1 - 400e-6)
    events = [det.inspect(m, state) for m in _periodic(0x11, attacker_period, n=300, attack=True)]
    fired = [e for e in events if e is not None]
    assert fired, "masquerade with a foreign clock skew should be detected"
    ev = fired[0]
    assert ev.detector == Detector.FINGERPRINT and ev.label == Label.MALICIOUS
    assert ev.features["method"] == "clock_skew" and ev.features["masquerade"] is True


def test_clock_skew_silent_without_enrolment_and_below_min_samples():
    det = Fingerprinting(min_samples=20)
    state = VehicleState()
    # not enrolled -> no opinion
    assert all(det.inspect(m, state) is None for m in _periodic(0x22, 0.1, n=50))

    det.enrol_clock_skew(_periodic(0x22, 0.1, n=200))
    det.reset()
    # fewer than min_samples observed -> not enough evidence to flag
    early = [det.inspect(m, state) for m in _periodic(0x22, 0.05, n=10, attack=True)]
    assert all(ev is None for ev in early)


# --------------------------------------------------------------------------- #
# voltage fingerprinting (via the simulated HAL sampler)
# --------------------------------------------------------------------------- #
def test_voltage_fingerprint_flags_masquerade_and_passes_genuine():
    ecu_features = {0x11: (1.0, 2.0, 3.0)}            # the legitimate ECU's profile
    sampler = SimulatedVoltageSampler(ecu_features, attacker_features=(5.0, 5.0, 5.0),
                                      jitter=0.01, seed=1)
    det = Fingerprinting(sampler=sampler, voltage_tol=0.5)
    det.enrol(0x11, PhysicalSample(0x11, 0.0, (1.0, 2.0, 3.0)))   # provisioning
    state = VehicleState()

    genuine = det.inspect(Message(arbitration_id=0x11, timestamp=0.0, is_attack=False), state)
    assert genuine is None                            # same ECU -> close fingerprint

    masq = det.inspect(Message(arbitration_id=0x11, timestamp=0.1, is_attack=True), state)
    assert masq is not None and masq.features["method"] == "voltage"
    assert masq.label == Label.MALICIOUS


def test_voltage_check_inert_without_sampler_or_profile():
    det = Fingerprinting(sampler=None)
    # no sampler -> voltage path is inert (clock-skew path also unenrolled)
    assert det.inspect(Message(arbitration_id=0x11, is_attack=True), VehicleState()) is None
