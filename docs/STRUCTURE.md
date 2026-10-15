# Repository layout and structural verification

```text
dllm-unmasking-dynamics/
├── README.md
├── pyproject.toml / setup.py
├── requirements.txt / requirements-analysis.txt
├── src/dllm_unmasking/       # samplers, evaluator, data and offline analysis
├── experiments/phase1..6/    # run.py, run.pbs, methodology README.md
├── configs/phase1..6.yaml    # original settings, unchanged bytes
├── tests/                   # fixture, baseline and structural verification
├── scripts/check_cpu.py     # verification utility
├── docs/                    # methodology, reproduction, audits, layout
├── reports/phase7/          # canonical report, tables and figures
├── research/                # public aggregates and selection index
├── outputs/                 # ignored raw outputs, not moved
└── results/                 # ignored generated analyses, not moved
```

## Commands

After dependency installation, use modern pip/setuptools and install the checkout:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps --no-build-isolation -e .
OMP_NUM_THREADS=1 python scripts/check_cpu.py
python -m dllm_unmasking.analyze_phase7 --output-dir results/phase7_reproduced
python -m experiments.phase6.run --help
```

Only the help/analysis commands above are CPU-only; actual runner execution
requires its original baseline files, cache and PBS GPU allocation. Tests with
`--with-artifacts` additionally need the original baseline results as documented
in [the reproduction guide](REPRODUCIBILITY.md). No top-level compatibility
scripts or `PYTHONPATH` edits are necessary. Root contains packaging, requirements
and overview files; scientific modules are importable under `dllm_unmasking`.

## Provenance and resume

Baseline: commit `fa8971a`, backed up before moving. The machine-readable
[source map](../src/dllm_unmasking/source_layout.json) records old names, new paths,
original SHA256 hashes, and every literal import/path substitution. For each old
module, reversing these finite substitutions must reproduce the **exact original
bytes**. An unrecorded computation change fails that hash check; a CPU regression
test deliberately changes a seed to verify rejection. Samplers with no internal
imports, the evaluator, and all six configs are byte-identical even without
normalization. Remaining modules preserve their original computation bodies.

Historical baseline manifests may contain old root-level names. The layout helper
resolves those names to relocated files and accepts only a direct physical hash
match or verified exact reconstruction matching the expected historical hash.
It never rewrites an artifact or returns a historical hash for an unverified edit.
Current runs hash **actual current source bytes**. Their different physical hashes
remain intentionally incompatible with interrupted pre-move run manifests: resume
those with the original checkout rather than bypassing checks. Existing original
runs and result archives are untouched.

Phase 7's canonical generator digest remains the verified pre-move generator
hash; normalization is used only for that compatibility field. This preserves the
existing published provenance and all deterministic numerical/figure outputs.
The layout helper and its tests are new structural code, not decoder changes.

## Complete movement inventory

| Before | After |
| --- | --- |
| `samplers.py` | `src/dllm_unmasking/samplers.py` |
| `evaluation.py` | `src/dllm_unmasking/evaluation.py` |
| `phase2_data.py` | `src/dllm_unmasking/phase2_data.py` |
| `phase2_samplers.py` | `src/dllm_unmasking/phase2_samplers.py` |
| `phase3_schedule.py` | `src/dllm_unmasking/phase3_schedule.py` |
| `phase3_sampler.py` | `src/dllm_unmasking/phase3_sampler.py` |
| `phase4_sampler.py` | `src/dllm_unmasking/phase4_sampler.py` |
| `phase5_sampler.py` | `src/dllm_unmasking/phase5_sampler.py` |
| `analyze_results.py` | `src/dllm_unmasking/analyze_results.py` |
| `analyze_phase15.py` | `src/dllm_unmasking/analyze_phase15.py` |
| `analyze_phase25.py` | `src/dllm_unmasking/analyze_phase25.py` |
| `analyze_phase6.py` | `src/dllm_unmasking/analyze_phase6.py` |
| `analyze_phase7.py` | `src/dllm_unmasking/analyze_phase7.py` |
| `prepare_ltr_audit.py` | `src/dllm_unmasking/prepare_ltr_audit.py` |
| `run_experiment.py` | `experiments/phase1/run.py` |
| `config.yaml` | `configs/phase1.yaml` |
| `run.pbs` | `experiments/phase1/run.pbs` |
| `run_phase2.py` | `experiments/phase2/run.py` |
| `phase2_config.yaml` | `configs/phase2.yaml` |
| `run_phase2.pbs` | `experiments/phase2/run.pbs` |
| `PHASE2.md` | `experiments/phase2/README.md` |
| `run_phase3.py` | `experiments/phase3/run.py` |
| `phase3_config.yaml` | `configs/phase3.yaml` |
| `run_phase3.pbs` | `experiments/phase3/run.pbs` |
| `PHASE3.md` | `experiments/phase3/README.md` |
| `run_phase4.py` | `experiments/phase4/run.py` |
| `phase4_config.yaml` | `configs/phase4.yaml` |
| `run_phase4.pbs` | `experiments/phase4/run.pbs` |
| `PHASE4.md` | `experiments/phase4/README.md` |
| `run_phase5.py` | `experiments/phase5/run.py` |
| `phase5_config.yaml` | `configs/phase5.yaml` |
| `run_phase5.pbs` | `experiments/phase5/run.pbs` |
| `PHASE5.md` | `experiments/phase5/README.md` |
| `run_phase6.py` | `experiments/phase6/run.py` |
| `phase6_config.yaml` | `configs/phase6.yaml` |
| `run_phase6.pbs` | `experiments/phase6/run.pbs` |
| `PHASE6.md` | `experiments/phase6/README.md` |
| `test_phase2.py` | `tests/test_phase2.py` |
| `test_phase3.py` | `tests/test_phase3.py` |
| `test_phase4.py` | `tests/test_phase4.py` |
| `test_phase5.py` | `tests/test_phase5.py` |
| `test_phase6.py` | `tests/test_phase6.py` |
| `test_phase7.py` | `tests/test_phase7.py` |
| `tools/check_cpu.py` | `scripts/check_cpu.py` |

New files: package markers, `src/dllm_unmasking/layout.py`,
`src/dllm_unmasking/source_layout.json`, `tests/test_layout.py`, `pyproject.toml`,
`setup.py`, `experiments/phase1/README.md`, and layout/planning documentation.
README, reproduction/navigation/audit guides, phase methodology links, PBS
invocations, tests and imports were updated to the new paths. Scientific result
CSV/JSON/PNG files and private raw artifacts were not changed.
