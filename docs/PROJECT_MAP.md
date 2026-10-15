# Project map

Run Python commands from the repository root. The root layout is intentional:
runners record source hashes, and later phases import earlier helpers directly.
Moving these files would change provenance and resume behavior.

| Phase | Scientific implementation / runner | Offline analysis / documentation | Generated location |
| --- | --- | --- | --- |
| 1 | `samplers.py`, `run_experiment.py`, `config.yaml` | `analyze_results.py`, [historical guide](PHASE1.md) | `outputs/phase1/` |
| 1.5 | No inference | `analyze_phase15.py`, [public report](../research/phase15/report.md) | `results/phase15/` |
| 2 | `phase2_samplers.py`, `phase2_data.py`, `run_phase2.py` | [design](../PHASE2.md) | `outputs/phase2/` |
| 2.5 | No inference | `analyze_phase25.py`, `prepare_ltr_audit.py` | `results/phase25/` |
| 3 | `phase3_schedule.py`, `phase3_sampler.py`, `run_phase3.py` | [design](../PHASE3.md), runner `--plan` / `--summarize` | `outputs/phase3/` |
| 4 | `phase4_sampler.py`, `run_phase4.py` | [design](../PHASE4.md), runner `--summarize` | `outputs/phase4/` |
| 5 | `phase5_sampler.py`, `run_phase5.py` | [design](../PHASE5.md), runner `--summarize` | `outputs/phase5/` |
| 6 | Original `samplers.py` through `run_phase6.py` | `analyze_phase6.py`, [design](../PHASE6.md) | `outputs/phase6/` |
| 7 | No inference | `analyze_phase7.py`, [canonical report](../reports/phase7/PHASE7_REPORT.md) | `results/phase7_reproduced/` |

Each inference phase has a matching config, PBS template, and (Phases 2–6)
`test_phaseN.py`. `evaluation.py` is the shared, unchanged numerical extractor.

## Import and input dependencies

- Phase 2 uses `phase2_data` for selection, offline cache verification, durable
  write-ahead journaling and duplicate detection; `run_phase2` provides cache and
  CSV helpers consumed by later runners.
- Phase 3 imports the Phase 2 prediction helper, exact Phase 2 selection and
  completed 600-evaluation baseline. Phase 4 additionally requires completed Phase 3.
- Phase 5 and Phase 6 require completed Phase 2. Later baseline validation checks
  original source hashes, question metadata and configuration, not just CSV counts.
- Phases 3–6 import the offline analysis helpers `analyze_phase25` and
  `analyze_phase15`. NumPy is therefore an inference-runner dependency too.
- Phase 6 imports its analyzer at run time; its analyzer imports runner validation
  inside the CLI. Keep both files present. Historical scripts are not standalone
  single-file transfers.
- Torch is needed for samplers and fixture tests. Transformers/Hugging Face imports
  are deferred until cache checks or model loading. PyArrow reads the pinned Parquet.
- Raw journal records carry responses and trajectories. Started/completed events
  nest completed records under `result` in Phases 2–6. Phase 1 uses a different
  `records.jsonl` schema. Do not interchange these formats.

`research/` contains curated Phase 1/1.5 outputs and a text-free benchmark selection
index. `reports/phase7/` contains the final report, lightweight figures, and public
per-question numeric measurements. Generated raw outputs, environment metadata,
manual response audits, dataset caches and weights stay outside Git. Historical
phase guides retain their preparation-time status statements; completed outcomes
are consolidated in the canonical report.
