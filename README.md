# VIS — Vehicular Immune System

A bio-inspired, multi-layered, self-evolving intrusion-defense system for
connected and autonomous vehicles: a fast **reflex** layer, a learning
**adaptive** layer with deception, and a federated **fleet** layer that shares
validated "antibodies" to give the whole fleet herd immunity.

> New here (or using Claude Code)? Read **CLAUDE.md** first — it is the design
> contract: architecture, data schemas, the non-negotiable rules, and the build
> phases.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                              # needs Python >= 3.10

pytest -q                  # run the tests (should pass out of the box)
python eval/harness.py      # smoke run: detectors vs. synthetic/simulated traffic
python eval/experiments.py  # full E1-E7 evaluation + Section 8 adversarial claims
python eval/run_datasets.py # detectors vs. the real datasets (needs downloads)
python eval/run_fleet.py    # fleet layer E5/E6/E7 over an emulated fleet
```

Results on the real datasets: [docs/RESULTS.md](docs/RESULTS.md) and
[docs/RESULTS_FLEET.md](docs/RESULTS_FLEET.md) (raw tables), and
[docs/FINDINGS.md](docs/FINDINGS.md) (what they mean, including the limitations).

Or use the Makefile: `make install`, `make test`, `make lint`, `make smoke`,
`make all`. Everything runs on a laptop against synthetic/simulated data — no
datasets or hardware required. The core install is dependency-light; heavier
components have extras: `pip install -e ".[ml]"` (richer anomaly models),
`pip install -e ".[fl]"` (federated-learning backends).

## Layout

```
src/vis/shared   contracts (Message/Event/Antibody), traffic, state, keystore/HSM, datasets
src/vis/reflex   authentication, fingerprinting (clock-skew), physics, traffic monitor,
                 response engine, hal (simulated CAN-FD bench + voltage sampler)
src/vis/adaptive anomaly, decoys, correlation, antibody pipeline (+ negative selection,
                 shadow runner), V2X misbehavior, LLM (advisory)
src/vis/fleet    attestation gate, federated learning (FedAvg/Krum), secure agg + DP,
                 validation lab, signed OTA, revocation
eval             metrics, evaluation harness, and the experiment suite
tests            unit + integration tests
datasets         (git-ignored) put CAN/V2X datasets here — see datasets/README.md
```


