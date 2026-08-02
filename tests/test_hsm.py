"""Phase-4 simulated HSM + keystore, and its use by attestation / signed OTA."""
from vis.adaptive.shadow_runner import ShadowRunner
from vis.fleet.attestation_gate import AttestationGate
from vis.fleet.ota_distributor import OTADistributor
from vis.shared.contracts import Antibody, ArtifactType
from vis.shared.keystore import (
    SimulatedHSM,
    SoftwareKeyStore,
    as_keystore,
    measure_firmware,
)


def _validated_rule(aid):
    ab = Antibody(attack_class="dos", artifact_type=ArtifactType.RULE,
                  artifact={"kind": "id_match", "arbitration_id": aid})
    ab.validation = {"server_side_passed": True}
    return ab


def test_firmware_measurement_changes_on_tamper():
    good = measure_firmware(b"firmware-v1")
    assert good == measure_firmware(b"firmware-v1")          # deterministic
    assert good != measure_firmware(b"firmware-v1-trojaned")  # tamper-evident


def test_software_keystore_sign_verify():
    ks = SoftwareKeyStore(b"k")
    sig = ks.sign(b"hello")
    assert ks.verify(b"hello", sig) is True
    assert ks.verify(b"hello!", sig) is False
    assert ks.verify(b"hello", "") is False


def test_hsm_private_key_is_not_exportable():
    hsm = SimulatedHSM()
    assert not hasattr(hsm, "key")               # no public accessor for the key
    sig = hsm.sign(b"data")
    assert hsm.verify(b"data", sig) is True


def test_hsm_verifier_is_verify_only_and_bound_to_its_hsm():
    hsm = SimulatedHSM()
    other = SimulatedHSM()
    v = hsm.verifier()
    sig = hsm.sign(b"payload")
    assert v.verify(b"payload", sig) is True
    assert not hasattr(v, "sign")                # vehicles cannot sign
    # a verifier from a different HSM rejects the signature (root-of-trust binding)
    assert other.verifier().verify(b"payload", sig) is False


def test_as_keystore_wraps_bytes_and_passes_through_stores():
    assert isinstance(as_keystore(b"raw"), SoftwareKeyStore)
    hsm = SimulatedHSM()
    assert as_keystore(hsm) is hsm


# --------------------------------------------------------------------------- #
# integration: HSM-backed attestation + OTA
# --------------------------------------------------------------------------- #
def test_attestation_gate_backed_by_hsm_rejects_tampered_firmware():
    ca = SimulatedHSM()
    good_fw = measure_firmware(b"ecu-firmware-v1")
    gate = AttestationGate(ca, trusted_measurements={good_fw})

    cert = gate.issue_certificate("veh1")
    assert gate.admit("veh1", good_fw, cert) is True
    # genuine cert but a tampered boot measurement -> rejected
    bad_fw = measure_firmware(b"ecu-firmware-v1-trojaned")
    assert gate.admit("veh1", bad_fw, cert) is False


def test_ota_signed_by_hsm_activates_only_under_correct_root_of_trust():
    maker = SimulatedHSM()
    ota = OTADistributor(maker)                  # server signs with the HSM
    ab = _validated_rule(0x000)
    ota.sign(ab)
    assert ab.is_validated is True

    # vehicle with the correct maker root of trust applies it
    shadow = ShadowRunner()
    assert ota.verify_and_apply(ab, shadow) is True
    assert shadow.is_active(ab.antibody_id) is True

    # a vehicle trusting a DIFFERENT maker key refuses to activate
    foreign = OTADistributor(SimulatedHSM())
    assert foreign.verify_and_apply(ab, ShadowRunner()) is False
