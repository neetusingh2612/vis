"""Learning-based anomaly detector (Phase 2).

Runs on the central compute over a COPY of traffic (observe-only). Emits a
*suspicion score* (Label.SUSPECTED), never a decision.

This is the dependency-light core implementation: a per-arbitration-id timing
model learned from a clean "self" corpus. Two cheap, effective CAN signals:

  * **unknown id** -- an arbitration id never seen in `self` (fuzzing/DoS
    inject ids the bus never normally carries); and
  * **inter-arrival compression** -- a frame arriving much *sooner* than the
    id's learned period. Injecting extra frames on an id (spoofing/fabrication)
    roughly halves its gap even when the id itself is perfectly legitimate.

Why a median ratio and not a z-score: real CAN inter-arrival distributions are
heavy-tailed (an id with a 10 ms period shows occasional 200 ms gaps when frames
are dropped or the ECU is busy). Those outliers inflate the standard deviation
so much that `mean - k*std` goes negative and a Gaussian rule can never fire.
The median is robust to that tail, and "arrived at more than 1/period_ratio
times the expected rate" is the physically meaningful statement anyway.

Stretched gaps (suspension attacks) are covered by the separate silence check
(`gap_check=True`): a suspended ECU emits nothing, so its absence is only
visible while inspecting OTHER traffic against a per-id deadline.

Pure Python, no numpy -- a richer model (IsolationForest / autoencoder) belongs
behind the `.[ml]` extra and can replace `_score` without touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable, Optional

from ..shared.contracts import Detector, Event, Label, Message, Tier
from ..shared.detector import BaseDetector
from ..shared.state import VehicleState

_MAX_SAMPLES = 20_000       # per id; plenty for a stable median, bounds memory


@dataclass
class _IatStats:
    """Per-id inter-arrival statistics: online mean/variance + a capped sample
    reservoir for the robust median."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    last_t: Optional[float] = None
    samples: list[float] = field(default_factory=list)
    max_iat: float = 0.0          # longest gap this id showed on clean traffic
    _median: Optional[float] = None

    def update(self, iat: float) -> None:
        self.n += 1
        delta = iat - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (iat - self.mean)
        if len(self.samples) < _MAX_SAMPLES:
            self.samples.append(iat)
        if iat > self.max_iat:
            self.max_iat = iat
        self._median = None

    @property
    def std(self) -> float:
        return sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def median(self) -> float:
        """Robust nominal period (cached)."""
        if self._median is None:
            if not self.samples:
                self._median = 0.0
            else:
                s = sorted(self.samples)
                mid = len(s) // 2
                self._median = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
        return self._median


class AnomalyDetector(BaseDetector):
    name = "anomaly_detector"

    def __init__(self, period_ratio: float = 0.5, min_samples: int = 5,
                 flag_unknown_ids: bool = True, z_threshold: float = 4.0,
                 gap_check: bool = False, gap_margin: float = 2.0,
                 gap_min_periods: float = 10.0, scan_interval: float = 0.05):
        # fire when iat < period_ratio * nominal period (0.5 => "twice the rate")
        self.period_ratio = period_ratio
        self.min_samples = min_samples
        self.flag_unknown_ids = flag_unknown_ids
        self.z_threshold = z_threshold          # retained for telemetry/compat
        # --- stretched-gap / silence check (suspension attacks) --------------
        self.gap_check = gap_check
        self.gap_margin = gap_margin            # x the longest gap seen on clean data
        self.gap_min_periods = gap_min_periods  # and at least this many periods
        self.scan_interval = scan_interval
        self._fit: dict[int, _IatStats] = {}    # learned per-id timing ("self")
        self._live: dict[int, float] = {}        # last-seen ts per id at inspect time
        self._fitted = False
        self._deadline: dict[int, float] = {}    # aid -> silence budget (seconds)
        self._overdue: set[int] = set()
        self._next_scan: Optional[float] = None

    def fit(self, messages: Iterable[Message]) -> None:
        """Learn 'normal' per-id timing from ONE continuous clean capture."""
        self.fit_sessions([messages])

    def fit_sessions(self, sessions: Iterable[Iterable[Message]]) -> None:
        """Learn from SEVERAL captures without measuring gaps across their seams.

        Concatenating captures is not the same as one long capture: the seam
        between two recordings invents an inter-arrival of whatever separates
        them (days, for captures recorded on different dates; or a negative
        value for datasets like ROAD that re-base every capture to the same
        start time). Those phantom gaps destroy `max_iat`, so silence budgets
        become effectively infinite. Continuity is therefore tracked per session.
        """
        stats: dict[int, _IatStats] = {}
        for session in sessions:
            last: dict[int, float] = {}          # per-session continuity only
            for m in session:
                s = stats.setdefault(m.arbitration_id, _IatStats())
                prev = last.get(m.arbitration_id)
                if prev is not None:
                    iat = m.timestamp - prev
                    if iat >= 0:
                        s.update(iat)
                last[m.arbitration_id] = m.timestamp
        self._fit = stats
        self._fitted = True
        self._build_deadlines()

    def _build_deadlines(self) -> None:
        """Per-id silence budget, calibrated from the clean corpus.

        A budget is the longest gap the id actually showed on clean traffic
        (x ``gap_margin``), floored at ``gap_min_periods`` nominal periods. Using
        each id's own observed worst case means ids that legitimately pause --
        event-triggered frames, diagnostics -- do not raise false alarms.
        """
        self._deadline = {}
        for aid, s in self._fit.items():
            if s.n < self.min_samples or s.median <= 0:
                continue           # not reliably periodic -> do not police it
            self._deadline[aid] = max(s.max_iat * self.gap_margin,
                                      s.median * self.gap_min_periods)

    def reset(self) -> None:
        # keep the learned model; only clear the live replay cursor
        self._live.clear()
        self._overdue.clear()
        self._next_scan = None

    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        # NB: _score must run first -- it advances the per-id last-seen cursor
        # the silence check reads. Returning early would freeze that cursor and
        # make the id look silent to itself.
        score = self._score(msg)

        # a suspended ECU emits nothing, so the evidence for it is visible only
        # while inspecting OTHER traffic
        if self.gap_check and self._fitted:
            missing = self._silence_check(msg.timestamp)
            if missing:
                return Event(
                    detector=Detector.ANOMALY, bus=msg.bus, label=Label.SUSPECTED,
                    confidence=0.7, source_tier=Tier.ADAPTIVE,
                    features={"arbitration_id": msg.arbitration_id,
                              "check": "stretched_gap",
                              "silent_ids": [hex(i) for i in sorted(missing)[:8]],
                              "silent_count": len(missing)},
                )

        if score is None:
            return None
        trigger = 1.0 / self.period_ratio if self.period_ratio > 0 else 2.0
        return Event(
            detector=Detector.ANOMALY,
            bus=msg.bus,
            label=Label.SUSPECTED,                       # observe-only: never a decision
            confidence=min(1.0, score / (2.0 * trigger)),
            source_tier=Tier.ADAPTIVE,
            features={"arbitration_id": msg.arbitration_id, "anomaly_score": round(score, 3),
                      "check": "compressed_gap"},
        )

    def _silence_check(self, now: float) -> set[int]:
        """Ids whose silence budget has been exceeded as of `now`.

        Rescans only every ``scan_interval`` seconds of bus time, so the cost is
        amortised rather than O(#ids) on every frame.
        """
        if self._next_scan is None or now >= self._next_scan:
            self._next_scan = now + self.scan_interval
            overdue = set()
            for aid, budget in self._deadline.items():
                seen = self._live.get(aid)
                # only police ids we have actually observed in this run
                if seen is not None and now - seen > budget:
                    overdue.add(aid)
            self._overdue = overdue
        return self._overdue

    def _score(self, msg: Message) -> Optional[float]:
        """Return a compression ratio above the trigger, or None if it looks normal.

        The score is `nominal_period / observed_gap`, so 1.0 is exactly on
        cadence and larger means more frames than the bus should carry.
        """
        aid = msg.arbitration_id
        last = self._live.get(aid)
        self._live[aid] = msg.timestamp
        stats = self._fit.get(aid)
        trigger = 1.0 / self.period_ratio if self.period_ratio > 0 else 2.0

        # an id never observed in the self corpus
        if stats is None or stats.n == 0:
            if self._fitted and self.flag_unknown_ids:
                return trigger + 1.0
            return None

        # need a previous live sighting and a usable learned distribution
        if last is None or stats.n < self.min_samples:
            return None
        nominal = stats.median
        if nominal <= 0.0:
            return None
        iat = msg.timestamp - last
        if iat <= 0.0:
            return trigger + 1.0        # same-instant duplicate: maximal compression
        ratio = nominal / iat
        return ratio if ratio >= trigger else None
