from vis.shared.contracts import Antibody, ArtifactType, Bus, Detector, Event, Label


def test_event_json_roundtrip():
    e = Event(detector=Detector.TRAFFIC, bus=Bus.CAN, label=Label.MALICIOUS,
              features={"rate": 99})
    d = e.to_dict()
    assert d["detector"] == "traffic"
    assert d["label"] == "malicious"
    assert "rate" in d["features"]


def test_antibody_not_validated_by_default():
    ab = Antibody(attack_class="dos", artifact_type=ArtifactType.RULE, artifact={})
    assert ab.is_validated is False
