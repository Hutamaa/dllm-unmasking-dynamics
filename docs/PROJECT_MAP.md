# Project map

Install this checkout once with `python -m pip install --no-deps -e .` after
installing the chosen CPU/GPU requirements and modern packaging tools. Commands
below use importable modules. [The layout guide](STRUCTURE.md) lists every move.

| Area | Location and entry point |
| --- | --- |
| Decoding and evaluation | `src/dllm_unmasking/`: original samplers, schedule, extractor and data helpers |
| Offline analyses | `src/dllm_unmasking/analyze_*.py`; run with `python -m dllm_unmasking.analyze_phase7`, etc. |
| Inference runners | `experiments/phaseN/run.py`; run with `python -m experiments.phaseN.run` |
| Experiment settings | `configs/phase1.yaml` through `configs/phase6.yaml`, moved byte-for-byte |
| HPC templates / methodology | `experiments/phaseN/run.pbs` and `README.md` |
| CPU tests | `tests/`; `python scripts/check_cpu.py`, or `--with-artifacts` for all integrations |
| Canonical results | [Phase 7 report](../reports/phase7/PHASE7_REPORT.md), tables and PNGs alongside it |
| Public historical aggregates | `research/`, including the text-free benchmark selection index |
| Private/generated artifacts | Ignored `outputs/` and `results/`, unchanged |

## Import and baseline dependencies

Phase 2 supplies predictions, selection, journaling, cache checks and CSV helpers.
Phases 3–6 import these and the Phase 1.5/2.5 statistical/diagnostic helpers. Phase 4
requires completed Phase 3 results; Phases 3, 5 and 6 require completed Phase 2.
Phase 6 reuses the original Phase 1 sampler for P1. NumPy is therefore a runner
dependency, not only an offline plotting dependency. Torch is required for sampler
tests; model loading remains deferred and requires the pinned offline snapshot.

Phase 1 uses `records.jsonl`; Phases 2–6 use started/completed journal events with
completed records under `result`. These schemas are not interchangeable. Baseline
validation checks matching question IDs/settings and historical source identity,
not merely row counts. See [structural provenance compatibility](STRUCTURE.md#provenance-and-resume).

## Working directories

CLI entry points discover the editable checkout and resolve paths from its root.
PBS must still be submitted from that root, for example
`qsub experiments/phase6/run.pbs` after reviewing its placeholder allocation.
PBS scripts import the installed modules; do not transfer a single runner file.
Deploy the whole reviewed public tree into a new destination, install it there,
and restore exact baseline files separately. No packages are installed in a job.
