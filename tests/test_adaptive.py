"""Phase-2 adaptive layer: the detection -> antibody -> shadow vertical slice."""
from vis.adaptive.anomaly_detector import AnomalyDetector
from vis.adaptive.antibody_generator import AntibodyGenerator, matcher_for_artifact
from vis.adaptive.correlation_engine import CorrelationEngine, Incident
from vis.adaptive.negative_selection import NegativeSelection
from vis.adaptive.shadow_runner import ShadowRunner
from vis.adaptive.v2x_misbehavior import V2XMisbehavior
from vis.shared.contracts import (
    Antibody,
    ArtifactType,
    Bus,
    Detector,
    Event,
    Label,
    Message,
    Tier,
)
from vis.shared.state import VehicleState
from vis.shared.traffic import SyntheticSource


def _suspected(aid, ts):
    return Event(detector=Detector.ANOMALY, bus=Bus.CAN, label=Label.SUSPECTED,
                 source_tier=Tier.ADAPTIVE, timestamp=ts, features={"arbitration_id": aid})


def _malicious(aid, ts):
    return Event(detector=Detector.DECOY, bus=Bus.CAN, label=Label.MALICIOUS,
                 source_tier=Tier.ADAPTIVE, timestamp=ts, features={"arbitration_id": aid})


# --------------------------------------------------------------------------- #
# anomaly_detector
# --------------------------------------------------------------------------- #
def test_anomaly_flags_unknown_id_flood_not_benign():
    det = AnomalyDetector()
    # learn 'self' from clean periodic traffic (ids 0x100..0x104, no 0x000)
    det.fit(SyntheticSource(n=1000, flood_at=None))

    state = VehicleState()
    det.reset()
    benign_flags, flood_flags = 0, 0
    for msg in SyntheticSource(n=1500, flood_at=1000, flood_len=200):
        ev = det.inspect(msg, state)
        if msg.is_attack and ev is not None:
            flood_flags += 1
            assert ev.label == Label.SUSPECTED   # observe-only, never a decision
        elif not msg.is_attack and ev is not None:
            benign_flags += 1
    assert flood_flags > 0          # the 0x000 flood (unknown id) is flagged
    assert benign_flags == 0        # regular ids raise no suspicion


# --------------------------------------------------------------------------- #
# negative_selection
# --------------------------------------------------------------------------- #
def test_negative_selection_rejects_matcher_that_hits_self():
    self_corpus = [Message(arbitration_id=0x100), Message(arbitration_id=0x200)]
    neg = NegativeSelection(self_corpus)
    assert neg.passes(matcher_for_artifact({"kind": "id_match", "arbitration_id": 0x000})) is True
    assert neg.passes(matcher_for_artifact({"kind": "id_match", "arbitration_id": 0x100})) is False


# --------------------------------------------------------------------------- #
# correlation_engine
# --------------------------------------------------------------------------- #
def test_correlation_suppresses_lone_anomaly_but_confirms_corroboration():
    eng = CorrelationEngine(window_s=0.05, min_events=2)
    assert eng.submit(_suspected(0x000, 0.00)) is None       # lone anomaly -> no incident
    inc = eng.submit(_suspected(0x000, 0.01))                 # second agreeing anomaly -> confirm
    assert isinstance(inc, Incident) and len(inc.events) == 2

    # a single high-confidence MALICIOUS event confirms on its own
    eng2 = CorrelationEngine()
    assert isinstance(eng2.submit(_malicious(0x123, 0.0)), Incident)


def test_correlation_drops_events_outside_window():
    eng = CorrelationEngine(window_s=0.05, min_events=2)
    assert eng.submit(_suspected(0x000, 0.0)) is None
    # second anomaly arrives too late -> first has aged out -> still no incident
    assert eng.submit(_suspected(0x000, 1.0)) is None


# --------------------------------------------------------------------------- #
# antibody_generator (+ negative_selection)
# --------------------------------------------------------------------------- #
def test_antibody_generated_for_unknown_id_but_rejected_for_self_id():
    neg = NegativeSelection([Message(arbitration_id=0x100)])   # 0x100 is known-good
    gen = AntibodyGenerator(neg)

    # incident on the injected id 0x000 -> a candidate antibody survives
    inc = Incident(attack_class="dos", events=[_suspected(0x000, 0.0), _suspected(0x000, 0.01)],
                   confidence=1.0)
    ab = gen.synthesize(inc)
    assert ab is not None
    assert ab.artifact == {"kind": "id_match", "arbitration_id": 0x000}
    assert ab.is_validated is False            # candidate only -- no signature yet

    # incident blaming a legitimate id (masquerade) -> rejected by self-tolerance
    inc_self = Incident(attack_class="masquerade", events=[_suspected(0x100, 0.0)], confidence=1.0)
    assert gen.synthesize(inc_self) is None


# --------------------------------------------------------------------------- #
# shadow_runner
# --------------------------------------------------------------------------- #
def test_shadow_runner_logs_hits_and_gates_activation_on_validation():
    ab = Antibody(attack_class="dos", artifact_type=ArtifactType.RULE,
                  artifact={"kind": "id_match", "arbitration_id": 0x000})
    shadow = ShadowRunner()
    shadow.add_candidate(ab)

    for msg in SyntheticSource(n=300, flood_at=100, flood_len=50):
        shadow.observe(msg)
    assert shadow.shadow_report()[ab.antibody_id] == 50      # would-be hits, no action taken

    # unvalidated candidate cannot be activated
    shadow.activate(ab.antibody_id)
    assert shadow.is_active(ab.antibody_id) is False

    # only a server-validated + signed antibody gains authority
    ab.validation = {"server_side_passed": True}
    ab.signature = "sig"
    shadow.add_candidate(ab)
    shadow.activate(ab.antibody_id)
    assert shadow.is_active(ab.antibody_id) is True


# --------------------------------------------------------------------------- #
# end-to-end: anomaly -> correlation -> antibody -> shadow
# --------------------------------------------------------------------------- #
def test_adaptive_pipeline_end_to_end():
    clean = list(SyntheticSource(n=1000, flood_at=None))
    det = AnomalyDetector()
    det.fit(clean)
    eng = CorrelationEngine(window_s=0.05, min_events=2)
    gen = AntibodyGenerator(NegativeSelection(clean))
    shadow = ShadowRunner()

    state = VehicleState()
    det.reset()
    produced = None
    for msg in SyntheticSource(n=1500, flood_at=1000, flood_len=200):
        ev = det.inspect(msg, state)
        if ev is not None:
            inc = eng.submit(ev)
            if inc is not None and produced is None:
                produced = gen.synthesize(inc)
                if produced is not None:
                    shadow.add_candidate(produced)
        shadow.observe(msg)

    assert produced is not None                                  # a candidate was learned
    assert produced.artifact["arbitration_id"] == 0x000          # blames the flood id
    assert shadow.shadow_report()[produced.antibody_id] > 0      # and would have caught it


# --------------------------------------------------------------------------- #
# v2x_misbehavior
# --------------------------------------------------------------------------- #
def _bsm(sender, t, pos, spd):
    return {"sender": sender, "time": t, "pos": pos, "spd": spd}


def test_v2x_detects_overspeed_teleport_mismatch_and_passes_benign():
    det = V2XMisbehavior(max_speed_mps=70.0, speed_tol_mps=15.0)

    # over-speed: claimed 200 m/s -> MALICIOUS
    ev = det.check(_bsm(1, 0.0, (0, 0, 0), (200, 0, 0)))
    assert ev is not None and ev.label == Label.MALICIOUS and ev.features["reason"] == "speed_exceeds_max"

    # benign motion: 20 m/s, position consistent over 1s -> no event
    det.reset()
    assert det.check(_bsm(2, 0.0, (0, 0, 0), (20, 0, 0))) is None
    assert det.check(_bsm(2, 1.0, (20, 0, 0), (20, 0, 0))) is None

    # teleport: jumps 1000 m in 1s -> impossible implied speed -> MALICIOUS
    det.reset()
    assert det.check(_bsm(3, 0.0, (0, 0, 0), (20, 0, 0))) is None
    tele = det.check(_bsm(3, 1.0, (1000, 0, 0), (20, 0, 0)))
    assert tele is not None and tele.features["reason"] == "position_teleport"

    # constant-position attack: claims 20 m/s but never moves -> SUSPECTED mismatch
    det.reset()
    assert det.check(_bsm(4, 0.0, (5, 5, 0), (20, 0, 0))) is None
    mism = det.check(_bsm(4, 1.0, (5, 5, 0), (20, 0, 0)))
    assert mism is not None and mism.label == Label.SUSPECTED
    assert mism.features["reason"] == "speed_position_mismatch"


def test_v2x_inspect_reads_message_payload():
    det = V2XMisbehavior()
    msg = Message(arbitration_id=7, bus=Bus.V2X, timestamp=0.0,
                  payload={"pos": (0, 0, 0), "spd": (999, 0, 0)})
    ev = det.inspect(msg, VehicleState())
    assert ev is not None and ev.detector == Detector.V2X_MISBEHAVIOR


# --------------------------------------------------------------------------- #
# stretched-gap / silence check (suspension attacks)
# --------------------------------------------------------------------------- #
def _periodic_bus(duration, ids_periods, drop=None, drop_window=None):
    """Interleave periodic ids over the SAME time span; optionally silence one
    id over a window (all ids must span the same duration, otherwise an id that
    simply ends early is legitimately 'silent')."""
    msgs = []
    for aid, period in ids_periods.items():
        t = 0.0
        while t < duration:
            in_window = bool(drop_window and drop_window[0] <= t <= drop_window[1])
            if not (drop == aid and in_window):
                msgs.append(Message(arbitration_id=aid, timestamp=t, is_attack=in_window))
            t += period
    return sorted(msgs, key=lambda m: m.timestamp)


def test_gap_check_detects_suspended_id_and_is_quiet_on_clean_traffic():
    ids = {0x100: 0.01, 0x200: 0.02}
    clean = _periodic_bus(10.0, ids)

    det = AnomalyDetector(gap_check=True)
    det.fit(clean)
    # a suspended ECU stops transmitting -> its silence must be noticed while
    # inspecting OTHER traffic
    attacked = _periodic_bus(10.0, ids, drop=0x200, drop_window=(2.0, 4.0))
    det.reset()
    flagged = [det.inspect(m, VehicleState()) for m in attacked]
    hits = [e for e in flagged if e is not None and e.features.get("check") == "stretched_gap"]
    assert hits, "suspension (absent frames) should raise a stretched-gap event"
    assert "0x200" in hits[0].features["silent_ids"]

    # and no silence alarms on clean traffic
    det.reset()
    clean_hits = [e for e in (det.inspect(m, VehicleState()) for m in clean)
                  if e is not None and e.features.get("check") == "stretched_gap"]
    assert clean_hits == []


def test_fit_sessions_does_not_measure_gaps_across_capture_seams():
    # two sessions recorded days apart; concatenating them would invent a
    # multi-day inter-arrival and inflate the silence budget to uselessness
    s1 = [Message(arbitration_id=0x100, timestamp=t * 0.01) for t in range(200)]
    s2 = [Message(arbitration_id=0x100, timestamp=1_000_000 + t * 0.01) for t in range(200)]

    naive = AnomalyDetector(gap_check=True)
    naive.fit(s1 + s2)                       # seam counted as one huge gap
    seam_aware = AnomalyDetector(gap_check=True)
    seam_aware.fit_sessions([s1, s2])        # continuity tracked per session

    assert naive._fit[0x100].max_iat > 1000          # poisoned by the seam
    assert seam_aware._fit[0x100].max_iat < 1.0      # real worst-case gap
    assert seam_aware._deadline[0x100] < naive._deadline[0x100]
