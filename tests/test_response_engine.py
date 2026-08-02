from vis.reflex.response_engine import decide
from vis.shared.contracts import Bus, Detector, Event, Label, ResponseAction
from vis.shared.state import DriveMode, VehicleState


def test_traffic_event_filters():
    e = Event(detector=Detector.TRAFFIC, bus=Bus.CAN, label=Label.MALICIOUS)
    assert decide(e, VehicleState()) == ResponseAction.FILTER


def test_no_unsafe_stop_at_highway_speed():
    e = Event(detector=Detector.PHYSICS, bus=Bus.CAN, label=Label.MALICIOUS)
    highway = VehicleState(speed_kph=110, mode=DriveMode.HIGHWAY)
    # must NOT choose a full stop at speed (G8) -> isolate instead
    assert decide(e, highway) == ResponseAction.ISOLATE


def test_suspected_only_logs():
    e = Event(detector=Detector.ANOMALY, bus=Bus.CAN, label=Label.SUSPECTED)
    assert decide(e, VehicleState()) == ResponseAction.LOG
