# Key findings — VIS against the real datasets

What we learned running the VIS stack against Car-Hacking, OTIDS, ROAD,
CAN-MIRGU and VeReMi. Raw numbers live in [RESULTS.md](RESULTS.md)
(`python eval/run_datasets.py`); this file records the *conclusions*, including
the ones that are inconvenient.

**Method, applied identically everywhere.** Every detector is fitted /
calibrated / enrolled on **attack-free** traffic only. Detection is measured on
attack captures; the false-positive rate is measured on a **held-out** clean
capture — a different driving session (ROAD) or a different day (CAN-MIRGU).
FPR is never measured on benign frames sitting inside an active attack window,
because a bus-level rate monitor flags everything in a high-rate window and that
would flatter the numbers.

---

## 1. Each attack class needs a different detector — the layering is the result, not the framing

No single detector wins. Mapping attack → signal → detector is the substantive
finding:

Detection is always paired with the false-positive rate it was bought at,
measured on held-out clean traffic. A detection figure quoted on its own is not
a result — the masquerade row is the clearest case: perfect recall, at a cost.

| Attack shape | Observable signal | Detector that works | Detection | FPR (held-out) |
|---|---|---|---|---|
| DoS / flooding (new id) | id never seen in `self` | AnomalyDetector (unknown-id) | 1.00 | 0.002 |
| Fuzzing | random ids, compressed gaps | AnomalyDetector | 0.93–0.99 | 0.002 |
| Fabrication / spoofing (legit id) | inter-arrival **compressed** | AnomalyDetector (median ratio) | 0.51–0.97 | 0.000–0.024 |
| Masquerade (legit id, right cadence) | **payload content** implausible | PhysicsChecks (byte envelope) | 1.00 | **0.085 (ROAD) / 0.159 (MIRGU)** |
| Suspension (frames removed) | **absence** past a deadline | AnomalyDetector (stretched-gap) | 0.99 | 0.002 |
| V2X position forgery | kinematic self-inconsistency | V2XMisbehavior | 0.91–0.92 | 0.0003 |
| V2X constant offset | *(none — see §5)* | — | ~0.00 | — |

The timing-based rows are cheap (≤2.4 % FPR); the content-based row is not. That
gap is the main open engineering problem (§4), and it is invisible if only
detection is reported.

## 2. Aggregate rate monitoring is nearly useless on real buses

`TrafficMonitor` scores **0.00–0.17** everywhere on real data, despite being the
detector that looks best on synthetic floods.

On Car-Hacking DoS the window occupancy during attack frames (p50 = 17 per 10 ms)
*overlaps* benign traffic (p50 = 19). No threshold separates them — at the most
generous setting we measured 25.7 % detection for 7.3 % false positives. The
injections are bursty and the bus is already busy, so total load barely moves.
On ROAD and CAN-MIRGU, targeted injection on one id is invisible in aggregate.

**Conclusion:** bus-level rate is a coarse liveness signal, not an IDS. Per-id
timing is what carries the information. This is evidence *for* the layered
design, not against it — but it does mean a rate monitor alone is not a defence.

## 3. Masquerade is a content attack, and E3 needed the physics layer

Clock-skew fingerprinting scores **0.00** on ROAD masquerade — correctly, not as
a bug. ROAD's masquerade suspends the real ECU and transmits at the *same
cadence*, so the timing channel carries no signal by construction. Calibrating
the skew threshold on clean ambient data (as we do for every other threshold)
pushes it to ~2.3 M ppm, i.e. real bus jitter dwarfs any crystal-skew effect:
a textbook 200 ppm bound fires on **71 %** of clean frames.

What does work: the masquerade pins a byte to an impossible value (ROAD's
`max_speedometer` forces `0xFF`). A per-byte value envelope learned from ambient
traffic gets **1.00 detection on all six ROAD masquerade captures, at 8.5 % FPR**
on a held-out ambient session. Precision on the attack captures ranges 0.04–0.68.

That FPR is not incidental — it is the same generalisation problem as §4, just
milder. The envelope is trained on 11 dyno-based ambient sessions and tested on
`ambient_highway_street_driving_long`, so highway signal values the dyno never
reached read as "implausible". Measured the sloppy way — fitting and scoring FPR
on the *same* ambient log — it looks like 0.00, which is exactly the flattering
artefact the held-out methodology exists to prevent.

**Conclusion:** for masquerade, *content* plausibility beats *timing* — timing
gets 0.00, content gets 1.00. But at 8.5 % FPR the byte envelope is a
demonstration that the content channel carries the signal, **not** a deployable
detector: one benign frame in twelve would raise an alarm. The voltage
fingerprint (hardware, Phase 4) remains the intended discriminator, with decoded
signal bounds (§4) as the software path to an acceptable FPR.

## 4. The byte envelope does not generalise to real-world driving

The same check costs **8.5 % FPR on ROAD** and **~16 % on CAN-MIRGU**, and
training on three benign days instead of one barely helped (19.7 % → 15.9 %).

The severity tracks how far the test traffic strays from training: ROAD is
dyno-based and largely controlled (8.5 %), CAN-MIRGU is a real vehicle on real
roads across different days (16 %), where signals reach values no prior day
contained. The failure mode is the same in both, only the magnitude differs.

A naive per-byte min/max envelope is brittle because it conflates two different
statements: **unseen ≠ implausible**. It has no notion of which byte encodes
which physical quantity, so it cannot tell "this speed is impossible" from
"we happened not to drive that fast during training".

**Conclusion:** honest limitation, and the detection figures in §3 should always
be quoted with their FPR. A deployable version needs decoded signals with
physical bounds (DBC + engineering limits), not raw byte ranges — exactly what
`signal_bounds` is for. We did **not** tune this away.

## 5. Some V2X attacks are undetectable by self-consistency — provably

Per attacker type, aggregated over ~19 scenarios and ~700–1150 receivers each:

| Attacker type | What it fakes | Detection | FPR |
|---|---|---|---|
| `random_pos` | random absolute position | **0.92** | 0.0003 |
| `random_pos_offset` | random offset from truth | **0.92** | 0.0003 |
| `const_pos` | frozen position | 0.61 | 0.0002 |
| `eventual_stop` | truthful, then freezes | 0.52 | 0.0002 |
| `const_pos_offset` | truth + **fixed** offset | **~0.00** | 0.0004 |

The speed tolerance was chosen by sweep (2 / 5 / 10 / 15 m/s) on clean and
attack data: 10 m/s matches 15 m/s on recall at the same near-zero FPR and does
better on `eventual_stop`. Tightening it lifted `const_pos` 0.38 → 0.61 and
`eventual_stop` 0.29 → 0.52 — but moved `const_pos_offset` not at all
(~0.006 even at 2 m/s).
The reason is structural: adding a constant offset to a real trajectory
preserves every internal relationship — speed still matches position deltas,
acceleration stays plausible. A single-vehicle plausibility check *cannot* see
it. Detecting it requires an external reference: cross-checking neighbours'
reports, local sensing (radar/LiDAR), or RSSI/ToF ranging.

`eventual_stop` is hard for a related reason — the attacker reports true
positions first, so early BSMs are genuinely indistinguishable from a car that
legitimately stopped. And the first BSM from any sender can never be judged
kinematically (no prior position), which puts a hard floor on recall.

**Conclusion:** report `const_pos_offset` as out of scope for the current
detector rather than claiming coverage. This is a limitation of the *approach*,
not of the implementation.

## 6. Fleet layer over an emulated fleet (E5/E6/E7)

`experiments.py` only checks *proxies* for these (weight-vector error, RMS noise,
pipeline stages passing). [RESULTS_FLEET.md](RESULTS_FLEET.md)
(`python eval/run_fleet.py`) runs them end to end: N vehicles each estimate a
per-id minimum-plausible-gap vector from their own shard of real Car-Hacking
benign traffic, the server aggregates, and **detection is then measured on the
real attack capture** (RPM spoofing — injection on a *legitimate* id, so the
learned timing model is what carries the signal).

**E5 — poisoning (scaling attack, N = 20, baseline recall 0.966):**

| f/n | FedAvg recall | retained | Krum recall | retained |
|---|---|---|---|---|
| 10 % | 0.000 | **0 %** | 0.966 | **100 %** |
| 20 % | 0.000 | **0 %** | 0.966 | **100 %** |
| 30 % | 0.000 | **0 %** | 0.966 | **100 %** |

Malicious clients submit −(n_honest/f) × their honest vector, dragging the mean
to ≈0; thresholds clamp at zero, so averaging yields a **completely blinded**
detector at even 10 % compromise. Krum selects an honest vector and retains full
detection through 30 %. Krum's condition n ≥ 2f+3 holds to f/n = 40 % at N = 20;
beyond that it is reported `n/a` rather than silently wrong.

**E6 — privacy (distributed DP: local Gaussian noise, then average):**

| ε | σ_agg (s) | recall | FPR | balanced acc | utility retained |
|---|---|---|---|---|---|
| 0.1 | 0.134 | 1.000 | 0.602 | 0.699 | 71 % |
| 0.5 | 0.027 | 0.997 | 0.419 | 0.789 | 80 % |
| 1.0 | 0.013 | 0.992 | 0.247 | 0.872 | 89 % |
| 5.0 | 0.003 | 0.974 | 0.007 | 0.983 | **100 %** |
| ≥10 | ≤0.001 | 0.966 | ≤0.0001 | 0.983 | 100 % |

**Recall is the wrong metric here and would invert the conclusion.** At ε = 0.1
the noise inflates thresholds so the detector flags almost everything: recall
*rises* to 1.000 while FPR hits 0.602. Balanced accuracy — (TPR+TNR)/2, which is
independent of class balance — is monotone in ε and is what we report. (F1 is not
a safe substitute: its precision term depends on the attack fraction of the
capture, so on an all-attack slice a blanket detector still scores F1 ≈ 1.0.)
**Practical operating point: ε ≈ 5 costs nothing measurable.**

**E7 — time to fleet immunity (quorum = 3, p_encounter = 0.1/round/vehicle, 500 trials):**

| N | rounds to quorum (p50) | T_immunity mean | p50 | p95 |
|---|---|---|---|---|
| 5 | 7 | 11.90 | 11 | 20 |
| 10 | 3 | 7.63 | 7 | 11 |
| 20 | 2 | 5.99 | 6 | 7 |
| 50 | 1 | 5.09 | 5 | 6 |
| 100 | 1 | **5.00** | 5 | 5 |

T_immunity = rounds to quorum + 4 fixed backend rounds (quorum → validation →
signing → OTA). It falls ~2.4× from N = 5 to N = 100 and then **floors at the
irreducible backend cost of 5 rounds** — herd immunity is a function of exposure
count, so more vehicles find the attack sooner, but no fleet size can beat the
validate-and-sign pipeline. Backend wall time is sub-millisecond (0.05–0.45 ms),
so the round count, not compute, is the bottleneck. This uses the real
`AttestationGate` / `tally_candidates` / `ValidationLab` / HSM signing /
`verify_and_apply` path; only per-round exposure is stochastic.

## 7. Bugs the real data exposed that synthetic data hid

Every one of these passed the synthetic test suite:

1. **Gaussian z-score could never fire.** Real CAN inter-arrivals are
   heavy-tailed (p50 = 10 ms, max = 209 ms for the same id). The outliers
   inflate σ so much that `mean − 4σ` goes **negative** — no gap could ever
   trigger it. Replaced with a robust **median-ratio** rule
   (`nominal / observed ≥ 2`): gear/RPM spoofing went **0.00 → 0.97**.
2. **Concatenating captures invents phantom gaps.** Fitting on benign files from
   different days produced a 19-day inter-arrival at the seam, inflating silence
   budgets to ~3.3 M s so the stretched-gap check never fired. `fit_sessions()`
   now tracks continuity per capture: suspension detection **0.003 → 0.99**.
   ROAD is worse in a subtler way — it re-bases every capture to the *same*
   start timestamp, so concatenation collapses 11 sessions onto one time axis
   and inflated a calibrated rate threshold to 300 033.
3. **0/0 reported as 0.0000.** Two CAN-MIRGU fuzzing captures inject only after
   ~150 k frames; a head-slice evaluation contained *no attack frames at all*,
   so "0 % detection" actually meant "nothing to detect". Now reported as `n/a`,
   and those captures are evaluated whole-file: **0.93–0.99**.
4. **Adapter written against an assumed format.** CAN-MIRGU ships candump logs
   with a trailing per-frame label, not labelled CSV.
5. **Crash on a placeholder id.** ROAD's fuzzing captures use `injection_id:
   "XXX"` because the injected ids are random. Those captures now degrade to
   "no id-based ground truth" and are excluded from scoring rather than
   mislabelled.

**Conclusion:** synthetic data validated the plumbing, not the statistics. Three
of these five were *silent* — wrong numbers, no error.

## 8. Ground truth is not uniformly available

- **OTIDS** labels by campaign, not per frame. Our DoS labels are a heuristic
  (injected id `0x000`); Fuzzy and Impersonation ship no per-frame labels and
  are reported `n/a` rather than guessed.
- **ROAD** fuzzing captures cannot be labelled by id (random ids).
- **CAN-MIRGU** suspension captures label *all traffic in the attack window*,
  including other ids' frames — which is why an absence-detector scores high
  precision there: the window itself is the ground truth.

Any cross-paper comparison has to state which labelling convention was used.

---

## Open items

- Decoded-signal bounds (DBC) to replace the raw byte envelope (§4).
- Cross-vehicle corroboration for V2X constant-offset attacks (§5).
- Voltage fingerprinting on the real bench for masquerade (§3) — the simulated
  HAL exercises the logic, silicon is still required.
- Only 19 of 225 VeReMi scenarios and one density/attacker-fraction setting were
  evaluated; the full sweep is a paper-scale run.
