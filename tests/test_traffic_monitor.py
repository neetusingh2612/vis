from vis.reflex.traffic_monitor import TrafficMonitor
from vis.shared.traffic import SyntheticSource
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from harness import evaluate  # noqa: E402


def test_flood_is_detected():
    src = SyntheticSource(n=2000, flood_at=1000, flood_len=200)
    m = evaluate(TrafficMonitor(window=0.1, rate_threshold=50), src)
    # the injected flood should be caught with few false positives
    assert m.detection_rate > 0.5
    assert m.false_positive_rate < 0.05
