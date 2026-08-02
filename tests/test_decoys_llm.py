"""Phase-2 deception subsystem (decoys) + advisory LLM assistant.

The LLM tests deliberately assert the *no-authority* property: a suggestion is
just an unproven draft that must survive negative selection.
"""
from vis.adaptive.antibody_generator import matcher_for_artifact
from vis.adaptive.correlation_engine import Incident
from vis.adaptive.decoys import DecoyListener
from vis.adaptive.llm_assistant import LLMAssistant
from vis.adaptive.negative_selection import NegativeSelection
from vis.shared.contracts import Bus, Detector, Event, Label, Message, Tier
from vis.shared.state import VehicleState


def _event(aid):
    return Event(detector=Detector.ANOMALY, bus=Bus.CAN, label=Label.SUSPECTED,
                 source_tier=Tier.ADAPTIVE, features={"arbitration_id": aid})


# --------------------------------------------------------------------------- #
# decoys
# --------------------------------------------------------------------------- #
def test_decoy_surfaces_flag_interactions_and_pass_benign():
    dec = DecoyListener(
        decoy_ids={0x7FF},
        phantom_ecu_ids={0x710},
        honeytokens={"vin_token": b"\xde\xad\xbe\xef"},
    )
    state = VehicleState()

    assert dec.inspect(Message(arbitration_id=0x100, data=b"\x01\x02"), state) is None

    hit = dec.inspect(Message(arbitration_id=0x7FF), state)
    assert hit is not None and hit.label == Label.MALICIOUS and hit.confidence == 1.0
    assert hit.features["surface"] == "decoy_id"

    ecu = dec.inspect(Message(arbitration_id=0x710), state)
    assert ecu is not None and ecu.features["surface"] == "phantom_ecu"

    tok = dec.inspect(Message(arbitration_id=0x123, data=b"\x00\xde\xad\xbe\xef\x00"), state)
    assert tok is not None and tok.features["surface"] == "honeytoken"
    assert tok.features["token"] == "vin_token"

    # everything that touched the surface is captured as clean labelled data
    assert len(dec.captured_events) == 3
    assert all(e.label == Label.MALICIOUS for e in dec.captured_events)


def test_decoy_rotation_is_per_vehicle_deterministic():
    pool = {0x700, 0x701, 0x702, 0x703, 0x704}
    a = DecoyListener(rotation_pool=pool, rotation_size=2, vehicle_seed=1)
    b = DecoyListener(rotation_pool=pool, rotation_size=2, vehicle_seed=2)

    assert len(a.decoy_ids) == 2 and a.decoy_ids <= pool   # initialised at epoch 0
    # same (seed, epoch) is reproducible
    assert a.rotate(7) == a.rotate(7)
    # two different vehicles get different live-decoy sequences -> diversity
    seq_a = [frozenset(a.rotate(e)) for e in range(5)]
    seq_b = [frozenset(b.rotate(e)) for e in range(5)]
    assert seq_a != seq_b
    assert all(s <= pool and len(s) == 2 for s in seq_a + seq_b)


def test_decoy_on_interaction_callback():
    dec = DecoyListener(decoy_ids={0x7FF})
    assert dec.on_interaction(0x100, {"svc": "uds"}) is None
    ev = dec.on_interaction(0x7FF, {"svc": "uds"})
    assert ev is not None and ev.features["svc"] == "uds"


# --------------------------------------------------------------------------- #
# llm_assistant
# --------------------------------------------------------------------------- #
def test_llm_drafts_advisory_candidate_grounded_in_evidence():
    inc = Incident(attack_class="dos", events=[_event(0x000), _event(0x000), _event(0x111)])
    draft = LLMAssistant().suggest_candidate(inc)
    assert draft["kind"] == "id_match"
    assert draft["arbitration_id"] == 0x000          # dominant id, from evidence
    assert draft["advisory"] is True
    assert isinstance(draft["rationale"], str) and draft["rationale"]


def test_llm_suggestion_has_no_authority_must_pass_negative_selection():
    llm = LLMAssistant()
    neg = NegativeSelection([Message(arbitration_id=0x100)])   # 0x100 is known-good

    # a suggestion blaming the injected id survives the gate
    good = llm.suggest_candidate(Incident(attack_class="dos", events=[_event(0x000)]))
    assert neg.passes(matcher_for_artifact(good)) is True

    # a suggestion blaming a legitimate id is rejected by self-tolerance
    bad = llm.suggest_candidate(Incident(attack_class="masq", events=[_event(0x100)]))
    assert neg.passes(matcher_for_artifact(bad)) is False


def test_llm_no_signal_yields_non_runnable_draft():
    # no usable arbitration id -> draft cannot be compiled into a matcher,
    # so it can never fire (fails safe rather than acting on nothing).
    draft = LLMAssistant().suggest_candidate(Incident(attack_class="?", events=[]))
    assert draft["advisory"] is True
    try:
        matcher_for_artifact(draft)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_llm_backend_shapes_rationale_only():
    llm = LLMAssistant(backend=lambda prompt: "BACKEND-SAYS-HI")
    draft = llm.suggest_candidate(Incident(attack_class="dos", events=[_event(0x000)]))
    assert draft["rationale"] == "BACKEND-SAYS-HI"
    assert draft["arbitration_id"] == 0x000          # structured target still from evidence
