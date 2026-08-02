"""Hardware abstraction layer (Phase 4) -- the seam to the physical testbed.

The reflex detectors are bus-agnostic: they consume ``Message`` and never touch
hardware directly. "Porting the reflex layer to the bench" therefore means
implementing two seams behind this interface:

  * a :class:`~vis.shared.traffic.TrafficSource` backed by a real CAN-FD adapter
    (SocketCAN / vendor SDK) -- :class:`BenchCanSource` marks that seam; and
  * a :class:`FingerprintSampler` that reads the physical layer (transceiver
    voltage / ADC) to fingerprint the *transmitter*.

On a laptop we provide a SIMULATED sampler so the fingerprinting *logic* is
testable; it fabricates stable per-ECU features (and a distinct attacker
profile) from ground truth. That is a stand-in for bench capture, NOT a
substitute: only the testbed yields real voltage fingerprints. The clock-skew
fingerprint, by contrast, is derived from message timestamps and needs no
special hardware -- see ``fingerprinting.py``.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

from ..shared.contracts import Bus, Message
from ..shared.traffic import TrafficSource


@dataclass
class PhysicalSample:
    """One physical-layer observation of a transmitted frame.

    `voltage_features` is a small feature vector summarising the analog signal
    (e.g. dominant-level mean, std, rising-edge slope, overshoot) -- bench-only.
    """
    arbitration_id: int
    timestamp: float
    voltage_features: tuple[float, ...] = field(default_factory=tuple)


class FingerprintSampler(ABC):
    """Reads the physical layer to fingerprint a frame's transmitter."""

    @abstractmethod
    def sample(self, msg: Message) -> PhysicalSample:  # pragma: no cover - interface
        ...


class SimulatedVoltageSampler(FingerprintSampler):
    """Laptop stand-in for the bench ADC.

    Produces a stable per-ECU feature vector (plus small Gaussian jitter), so the
    same ECU fingerprints consistently and a masquerading ECU fingerprints
    differently. It reads ``msg.is_attack`` -- the ground-truth label -- to model
    which ECU *physically* sent the frame, exactly as the bench would observe it.
    The detector itself never sees this; it judges only the returned sample.
    """

    def __init__(self, ecu_features: dict[int, tuple[float, ...]],
                 attacker_features: tuple[float, ...] = (5.0, 5.0, 5.0),
                 jitter: float = 0.01, seed: int = 0):
        self.ecu_features = ecu_features
        self.attacker_features = attacker_features
        self.jitter = jitter
        self._rng = random.Random(seed)

    def sample(self, msg: Message) -> PhysicalSample:
        base = self.attacker_features if msg.is_attack \
            else self.ecu_features.get(msg.arbitration_id, (0.0, 0.0, 0.0))
        feats = tuple(x + self._rng.gauss(0.0, self.jitter) for x in base)
        return PhysicalSample(msg.arbitration_id, msg.timestamp, feats)


class BenchCanSource(TrafficSource):
    """Real CAN-FD bench frame source (Phase 4 -- requires hardware).

    Implement `__iter__` against SocketCAN / a vendor SDK to stream live frames
    as ``Message`` objects; every reflex detector then runs unchanged. Left
    unimplemented because it cannot run without the testbed.
    """

    def __init__(self, channel: str = "can0", fd: bool = True):
        self.channel = channel
        self.fd = fd

    def __iter__(self):  # pragma: no cover - requires hardware
        raise NotImplementedError(
            "BenchCanSource requires the CAN-FD testbed (Phase 4); "
            "wire SocketCAN/vendor SDK here to stream live frames as Messages."
        )


@dataclass
class Ecu:
    """A simulated ECU on the CAN-FD bench.

    `clock_skew_ppm` is the crystal's rate error vs the receiver clock -- it is
    what clock-skew fingerprinting keys on. `voltage_features` is the analog
    transmitter fingerprint the voltage sampler would measure.
    """
    name: str
    sends: dict[int, float]                    # arbitration_id -> nominal period (s)
    clock_skew_ppm: float = 0.0
    voltage_features: tuple[float, ...] = (1.0, 1.0, 1.0)
    dlc: int = 8                               # CAN-FD payload length (up to 64)


@dataclass
class _Masquerade:
    target_id: int
    skew_ppm: float
    start: float
    end: float
    attack_type: str
    features: tuple[float, ...]
    suspend_victim: bool


@dataclass
class _Flood:
    arbitration_id: int
    period: float
    start: float
    end: float
    attack_type: str


class SimulatedCanFdBench(TrafficSource):
    """Simulated CAN-FD bus: a software stand-in for the Phase-4 bench.

    Several ECUs each transmit periodic frames on their OWN (slightly skewed)
    clock; attacks can be injected. Yields time-ordered ``Bus.CAN_FD`` Messages,
    so it plugs straight into ``eval/harness.py`` and every reflex detector. Pair
    with :meth:`voltage_sampler` to drive voltage fingerprinting too.

    This exercises the full reflex pipeline on realistic multi-ECU timing; it is
    a simulator, not a substitute for the real bench's analog behaviour.
    """

    def __init__(self, ecus: list[Ecu], duration_s: float = 1.0, fd: bool = True):
        self.ecus = list(ecus)
        self.duration = duration_s
        self.fd = fd
        self._masq: list[_Masquerade] = []
        self._flood: list[_Flood] = []
        self._attacker_features: tuple[float, ...] = (5.0, 5.0, 5.0)

    def add_masquerade(self, target_id: int, attacker_skew_ppm: float, start: float, end: float,
                       attacker_features: tuple[float, ...] = (5.0, 5.0, 5.0),
                       attack_type: str = "masquerade", suspend_victim: bool = True) -> None:
        """Attacker transmits `target_id` from a foreign clock over [start, end].

        With `suspend_victim` (the classic masquerade) the genuine ECU goes
        silent on that id for the window, so only the attacker's clock is seen.
        """
        self._attacker_features = attacker_features
        self._masq.append(_Masquerade(target_id, attacker_skew_ppm, start, end,
                                      attack_type, attacker_features, suspend_victim))

    def add_flood(self, arbitration_id: int, period: float, start: float, end: float,
                  attack_type: str = "dos") -> None:
        """Inject a high-rate flood on `arbitration_id` over [start, end]."""
        self._flood.append(_Flood(arbitration_id, period, start, end, attack_type))

    def voltage_sampler(self, jitter: float = 0.01, seed: int = 0) -> SimulatedVoltageSampler:
        """A voltage sampler bound to this bench's per-ECU profiles."""
        feats = {aid: ecu.voltage_features for ecu in self.ecus for aid in ecu.sends}
        return SimulatedVoltageSampler(feats, attacker_features=self._attacker_features,
                                       jitter=jitter, seed=seed)

    def _nominal_period(self, arbitration_id: int) -> Optional[float]:
        for ecu in self.ecus:
            if arbitration_id in ecu.sends:
                return ecu.sends[arbitration_id]
        return None

    def _suspended(self, arbitration_id: int, t: float) -> bool:
        return any(m.target_id == arbitration_id and m.suspend_victim and m.start <= t < m.end
                   for m in self._masq)

    def __iter__(self) -> Iterator[Message]:
        events: list[tuple[float, int, int, bool, Optional[str]]] = []  # (t, aid, dlc, atk, type)

        # genuine periodic traffic (each ECU on its own skewed clock)
        for ecu in self.ecus:
            for aid, period in ecu.sends.items():
                actual = period * (1.0 + ecu.clock_skew_ppm / 1e6)
                t = 0.0
                while t < self.duration:
                    if not self._suspended(aid, t):
                        events.append((t, aid, ecu.dlc, False, None))
                    t += actual

        # masquerade injections (attacker's clock on a victim id)
        for m in self._masq:
            nominal = self._nominal_period(m.target_id) or 0.01
            actual = nominal * (1.0 + m.skew_ppm / 1e6)
            t = m.start
            while t < m.end:
                events.append((t, m.target_id, 8, True, m.attack_type))
                t += actual

        # flood injections
        for fl in self._flood:
            t = fl.start
            while t < fl.end:
                events.append((t, fl.arbitration_id, 8, True, fl.attack_type))
                t += fl.period

        events.sort(key=lambda e: e[0])
        for t, aid, dlc, atk, atype in events:
            n = dlc if self.fd else min(dlc, 8)
            yield Message(arbitration_id=aid, data=bytes(n), bus=Bus.CAN_FD,
                          timestamp=t, is_attack=atk, attack_type=atype)

