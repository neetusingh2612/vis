"""Phase-3 fleet layer: robust aggregation, secure-agg/DP, and the antibody
graduation pipeline (attestation -> quorum -> validation -> signed OTA -> activate)."""
from vis.adaptive.shadow_runner import ShadowRunner
from vis.fleet.attestation_gate import AttestationGate
from vis.fleet.fl_client import FLClient
from vis.fleet.fl_server import FLServer, fedavg, krum, tally_candidates
from vis.fleet.ota_distributor import OTADistributor
from vis.fleet.revocation_service import RevocationService
from vis.fleet.secure_agg import add_dp_noise, apply_pairwise_masks, secure_sum
from vis.fleet.validation_lab import ValidationLab
from vis.shared.contracts import Antibody, ArtifactType, Message

CA_KEY = b"ca-secret"
MAKER_KEY = b"maker-secret"


def _rule(aid, attack_class="dos"):
    return Antibody(attack_class=attack_class, artifact_type=ArtifactType.RULE,
                    artifact={"kind": "id_match", "arbitration_id": aid})


# --------------------------------------------------------------------------- #
# attestation_gate (+ revocation)
# --------------------------------------------------------------------------- #
def test_attestation_admits_genuine_rejects_sybil_bad_measurement_and_revoked():
    rev = RevocationService()
    gate = AttestationGate(CA_KEY, trusted_measurements={"good-fw"}, revocation=rev)

    cert = gate.issue_certificate("veh1")
    assert gate.admit("veh1", "good-fw", cert) is True
    assert gate.is_admitted("veh1")

    # forged certificate
    assert gate.admit("evil", "good-fw", "deadbeef") is False
    # genuine cert but un-trusted secure-boot measurement (tampered firmware)
    assert gate.admit("veh2", "bad-fw", gate.issue_certificate("veh2")) is False
    # revoked vehicle is turned away even with a valid cert
    rev.revoke("veh3")
    assert gate.admit("veh3", "good-fw", gate.issue_certificate("veh3")) is False


# --------------------------------------------------------------------------- #
# fl_server aggregation
# --------------------------------------------------------------------------- #
def test_fedavg_is_sample_weighted():
    updates = [{"weights": [0.0, 0.0], "num_samples": 1},
               {"weights": [10.0, 20.0], "num_samples": 3}]
    out = fedavg(updates)
    assert out["weights"] == [7.5, 15.0]      # (0*1 + 10*3)/4, (0*1 + 20*3)/4


def test_krum_rejects_poisoned_outliers():
    honest = [{"weights": [1.0, 1.0]}, {"weights": [1.1, 0.9]},
              {"weights": [0.9, 1.1]}, {"weights": [1.0, 1.2]}]
    poison = [{"weights": [100.0, 100.0]}]     # f = 1 malicious outlier
    chosen = krum(honest + poison, f=1)
    assert chosen["weights"][0] < 2.0 and chosen["weights"][1] < 2.0   # an honest vector won

    server = FLServer(aggregator=krum, byzantine_f=1)
    assert server.aggregate(honest + poison)["weights"][0] < 2.0


def test_krum_requires_enough_clients():
    try:
        krum([{"weights": [1.0]}, {"weights": [1.0]}], f=1)   # n=2 < 2f+3
        raised = False
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# secure_agg
# --------------------------------------------------------------------------- #
def test_pairwise_masks_cancel_in_secure_sum():
    updates = [{"weights": [1.0, 2.0]}, {"weights": [3.0, 4.0]}, {"weights": [5.0, 6.0]}]
    true_sum = [9.0, 12.0]
    masked = apply_pairwise_masks(updates, seed=7)
    # individual masked vectors are perturbed...
    assert masked[0]["weights"] != updates[0]["weights"]
    # ...but their sum is exact
    got = secure_sum(masked)["weights"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, true_sum))


def test_dp_noise_is_calibrated_to_epsilon():
    import random
    base = {"weights": [0.0] * 2000}
    tight = add_dp_noise(base, epsilon=0.1, rng=random.Random(0))   # strong privacy -> big sigma
    loose = add_dp_noise(base, epsilon=10.0, rng=random.Random(0))  # weak privacy -> small sigma
    assert tight["dp"]["sigma"] > loose["dp"]["sigma"]
    spread_tight = max(tight["weights"]) - min(tight["weights"])
    spread_loose = max(loose["weights"]) - min(loose["weights"])
    assert spread_tight > spread_loose


# --------------------------------------------------------------------------- #
# validation_lab
# --------------------------------------------------------------------------- #
def test_validation_passes_clean_rule_and_fails_false_positive_rule():
    normal = [Message(arbitration_id=0x100) for _ in range(50)]
    attack = [Message(arbitration_id=0x000) for _ in range(10)]
    lab = ValidationLab(normal_corpus=normal, attack_corpus=attack, min_detection=0.9)

    good = lab.validate(_rule(0x000))
    assert good.validation["server_side_passed"] is True
    assert good.validation["fpr"] == 0.0 and good.validation["detection"] == 1.0

    # a rule matching a legitimate id raises false alarms -> rejected
    bad = lab.validate(_rule(0x100))
    assert bad.validation["server_side_passed"] is False


# --------------------------------------------------------------------------- #
# ota_distributor
# --------------------------------------------------------------------------- #
def test_ota_refuses_to_sign_unvalidated():
    ota = OTADistributor(MAKER_KEY)
    try:
        ota.sign(_rule(0x000))     # no validation record
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_ota_tampered_antibody_does_not_activate():
    ota = OTADistributor(MAKER_KEY)
    ab = _rule(0x000)
    ab.validation = {"server_side_passed": True}
    ota.sign(ab)
    assert ab.is_validated

    # tamper with the artifact after signing -> signature no longer matches
    ab.artifact["arbitration_id"] = 0x111
    assert ota.verify_and_apply(ab, ShadowRunner()) is False


# --------------------------------------------------------------------------- #
# end-to-end: fleet immunity (the candidate graduates to acting)
# --------------------------------------------------------------------------- #
def test_fleet_immunity_pipeline_end_to_end():
    rev = RevocationService()
    gate = AttestationGate(CA_KEY, trusted_measurements={"good-fw"}, revocation=rev)

    # three genuine vehicles independently propose the SAME rule for id 0x000
    genuine = [FLClient(f"veh{i}", "good-fw", gate.issue_certificate(f"veh{i}")) for i in range(3)]
    updates = [c.make_update(candidates=[_rule(0x000)]) for c in genuine]
    # plus a Sybil with a forged cert, also pushing a (different, poisoned) rule
    sybil = FLClient("evil", "good-fw", "forged")
    updates.append(sybil.make_update(candidates=[_rule(0x100, attack_class="poison")]))

    # 1) admission: only genuine vehicles get through
    admitted = [u for u in updates
                if gate.admit(u["vehicle_id"], u["attestation_token"], u["certificate"])]
    assert {u["vehicle_id"] for u in admitted} == {"veh0", "veh1", "veh2"}

    # 2) quorum: the rule seen by >=2 admitted vehicles is promoted
    promoted = tally_candidates(admitted, quorum=2)
    assert len(promoted) == 1
    candidate = promoted[0]
    assert candidate.artifact == {"kind": "id_match", "arbitration_id": 0x000}
    assert candidate.is_validated is False        # not yet -- still must pass the lab

    # 3) server-side validation against holdout corpora
    lab = ValidationLab(
        normal_corpus=[Message(arbitration_id=0x100) for _ in range(100)],
        attack_corpus=[Message(arbitration_id=0x000) for _ in range(20)],
        min_detection=0.9,
    )
    candidate = lab.validate(candidate)
    assert candidate.validation["server_side_passed"] is True

    # 4) sign + 5) on-vehicle verify -> graduate from shadow to ACTING
    ota = OTADistributor(MAKER_KEY)
    ota.sign(candidate)
    assert candidate.is_validated is True

    vehicle_shadow = ShadowRunner()
    assert ota.verify_and_apply(candidate, vehicle_shadow) is True
    assert vehicle_shadow.is_active(candidate.antibody_id) is True
