# Reproduction guide

Commands below run from the repository root after editable installation.
Use pip ≥21.3 and setuptools ≥64 for editable builds. Installation is a one-time
preparation step, never performed by a PBS job. The package is intended for an
editable research checkout, not an isolated wheel without its configs/data.
CLI entry points also discover this checkout when invoked from another directory;
explicit relative CLI input/output paths are repository-relative. There are three levels: public-table
analysis, CPU sampler tests, and full GPU reproduction. Only the last needs model
weights. Raw journals are not bundled, so an empty clone cannot rebuild response
case studies or resume the original jobs. No public raw-data download is currently
provided; do not imply otherwise.

## Environment and dependencies

`requirements-analysis.txt` pins the exercised local analysis packages: NumPy
2.0.2, Matplotlib 3.9.4, PyYAML 6.0.2 and PyArrow 19.0.1. PyArrow is needed only for
selection reconstruction. The local verification interpreter was Python 3.9 with
Torch 2.7.1; use Python 3.11 for the documented GPU setup. These are tested package
sets, not a complete cross-platform lockfile.

`requirements.txt` adds Transformers 4.38.2, datasets 3.6.0, huggingface-hub 0.30.2
and Accelerate 1.6.0. PyTorch must be installed separately for the target device.
The recorded Phase 2 environment used Torch 2.11.0+cu128, Transformers 4.38.2 and
huggingface-hub 0.30.2. The old Phase 1 guide suggested Torch 2.7.1+cu128; neither
pin guarantees identical results across machines. Accelerate's existing pin is
retained for model loading and was not exercised by CPU fixtures.

CPU setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1
python -m pip install -r requirements-analysis.txt
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

For GPU reproduction, use a compatible CUDA-enabled PyTorch installation in
`dllm-unmasking`, install `requirements.txt` and the editable package (`python -m pip install --no-deps -e .`),
and record the actual versions and
GPU. Do not copy a macOS environment to a GPU node. Full inference requires BF16
support and enough memory for the 8B model; recorded jobs requested one GPU, four
CPUs and 64 GB host memory. Published timings used an RTX PRO 6000 Blackwell
Server Edition. No model downloads are part of tests or analysis.

## Public Phase 7 figures and statistics

```bash
python -m dllm_unmasking.analyze_phase7 --output-dir results/phase7_reproduced
```

Inputs are the tracked `reports/phase7/unified_per_question.csv`,
`method_summary.csv`, `selected_paired_tests.csv` and
`research/benchmark_selection.json`. The script checks 1,020 unique evaluations,
17 matching sets of 60, strata, observed timings, published summary values,
frontier membership, discordant counts and exact McNemar tests. It regenerates
three PNGs, `method_summary.csv`, `paired_reanalysis.csv`, a Markdown table, and input/script-hash metadata.
Plots convey the same measured data but are not pixel-identical recreations of
the original figures. Defaults use 10,000 within-stratum paired bootstrap samples,
seed 42 with NumPy PCG64, linear percentile quantiles, and Holm correction across
the ten displayed contrasts. The report specifies the exact RNG consumption order.

The original Phase 7 plotting/analysis generator was not included in the imported
report package. Its bootstrap seed, repetition count and resampling scheme are
not independently documented. Original reported intervals remain untouched; the
new, explicitly specified intervals are the canonical current reanalysis, not a
claim of historical reproduction. The report and `reports/phase7/paired_reanalysis.csv`
use these new values; `bootstrap_provenance.json` records the generating script and
input hashes. To verify their exact reproduction:

```bash
python -m dllm_unmasking.analyze_phase7 --output-dir results/phase7_verified --seed 42 --bootstrap-repetitions 10000
cmp reports/phase7/paired_reanalysis.csv results/phase7_verified/paired_reanalysis.csv
cmp reports/phase7/bootstrap_provenance.json results/phase7_verified/reproduction_metadata.json
```

Use the pinned analysis dependencies for byte comparisons. Response degeneration and revision-change claims require
private raw journals; the public numeric table alone cannot regenerate them.

## CPU tests

```bash
OMP_NUM_THREADS=1 python scripts/check_cpu.py
```

This runs the historical CPU fixture tests plus public-table tests, with seven
explicit skips for tests that consume excluded artifacts. It does not invoke
pretrained models or GPUs. Skips are named individually; other failures remain
failures. Fixture logits test simultaneous commitment, tie-breaking, fresh revision
predictions, revision/new-position exclusion, exact coverage, P1 equivalence,
journal recovery and duplicate rejection.

For the full suite, provide original Phase 2/3 directories and exact selection:

```bash
export PHASE2_RESULTS="$RAW_RESULTS_DIR/phase2"
export PHASE3_RESULTS="$RAW_RESULTS_DIR/phase3"
# outputs/phase2/selection.json must also exist (see next section).
OMP_NUM_THREADS=1 python scripts/check_cpu.py --with-artifacts
# Equivalent historical discovery command, also requiring those artifacts:
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v -p 'test_*.py'
```

The original tests are preserved unchanged. Some tests expect actual baseline
counts and matching source hashes, so arbitrary synthetic baseline CSVs are not
a substitute. Raw input hashes should be checked before and after any new analysis.

## Pinned data, selection and seeds

- Model/tokenizer: `GSAI-ML/LLaDA-8B-Instruct` at
  `852152c60bdfed4632635bfcf3e83408ad3f65bf`.
- Dataset: `openai/gsm8k`, `main`, `test`, revision
  `740312add88f781978c0658806c59bc2815b9866`.
- Test Parquet SHA256:
  `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
- Selection: exclude the 150 Phase 1 row indices; sample 20 in each calculation
  stratum using Python RNG seed 20261014 and sorted input pools; preserve sampled
  stratum order and sort IDs within each stratum. Row IDs refer to the pinned split.
- Inference/global seed 42; private random-position RNG uses `42 + question_id`.
  Remasking RNG uses `314159 + question_id`, independent of evaluation order.
- The public selection index includes all 60 IDs, strata, calculation counts,
  exclusions and the original full selection SHA256. It contains no question text.

To reconstruct the original full selection from an **existing** pinned Parquet:

```bash
mkdir -p outputs/phase2
python - <<'PY'
import json
from pathlib import Path
public = json.loads(Path('research/benchmark_selection.json').read_text())
path = Path('outputs/phase2/phase1_exclusions.json')
value = {'question_ids': public['excluded_phase1_ids']}
if path.exists():
    assert json.loads(path.read_text())['question_ids'] == value['question_ids']
else:
    path.write_text(json.dumps(value, indent=2) + '\n')
PY
python -m experiments.phase2.run --prepare-selection --dataset-parquet "$GSM8K_TEST_PARQUET"
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('outputs/phase2/selection.json')
expected = json.loads(Path('research/benchmark_selection.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['selection_sha256']
print('Exact selection bytes verified')
PY
```

This restores selection, not the original exclusions-file metadata/hash or raw
results. To resume old runs, restore their exact original files rather than these
reconstructed exclusions. New-run manifests record their own input hashes. Model
weights and dataset text must be obtained separately under their applicable terms;
this repository does not redistribute either. The reconstruction command never
accesses weights or downloads data.

## Model cache and HPC execution

The pinned model includes custom Python code and loads with
`trust_remote_code=True`; review the pinned implementation. Provision the model
snapshot and pinned dataset once using your institution's approved process.
Phases 2–6 require existing local files and fail if unavailable; do not turn off
offline flags to bypass a missing-revision error.

```bash
export HF_HOME="$SCRATCH_DIR/huggingface"
# If your established installation uses a separate HF_HUB_CACHE, retain it.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m experiments.phase2.run --check-cache --dataset-parquet "$GSM8K_TEST_PARQUET"
```

PBS templates assume queue `gpu_bwp6000`, module `anaconda2025/2025-gpu` for later
phases, and Conda environment `dllm-unmasking`. Allocation identifiers are
`YOUR_PROJECT_CODE` placeholders. Site resource names, queue, walltime, project
flag (`-A` in earlier templates, `-P` in later templates), shell/module initialization
and actual allocation must be reviewed locally. Phase 2 still has generic Phase 1
Conda variables; configure `CONDA_BASE`, optional `CONDA_MODULE`, and `LLADA_ENV`.
These templates are historical examples, not portable scheduler configuration.

Copy the template to an ignored operational file before editing it:

```bash
cp experiments/phase6/run.pbs run_phase6.local.pbs
# Edit allocation/site settings in run_phase6.local.pbs.
# Deploy this whole repository, excluding .git, outputs, results and caches,
# into a NEW destination; do not overwrite a live historical experiment tree.
# After review and a future commit, transfer only tracked public files:
git ls-files -z | rsync -av --from0 --files-from=- ./ "$HPC_HOST:$PROJECT_ROOT/"
# Transfer your reviewed run_phase6.local.pbs separately; it is intentionally ignored.
```

Restore exact baseline files separately into configured paths. Later runners
verify source hashes and reject mismatches. Never edit a saved manifest to bypass
validation. All new runs should use fresh output directories; resume only exact
matching code/configuration. PBS commands below are **manual examples**, not run
by analysis:

```bash
# On HPC, from the deployed repository root, with baseline and cache present:
python -m experiments.phase6.run --check-cache
export PHASE6_POLICY=left_to_right
qsub -v HF_HOME,PHASE6_POLICY run_phase6.local.pbs
# After that job has stopped, confidence can resume into the same run directory:
export PHASE6_POLICY=confidence
qsub -v HF_HOME,PHASE6_POLICY run_phase6.local.pbs
```

Pass `HF_HUB_CACHE` too if your installation defines it. Never submit concurrent
writers to the same output directory. Phase 6 requests 8 hours for 30,720 forwards;
this is overhead allowance, not a runtime guarantee.

## Historical phase reproduction and outputs

| Phase | Runner and scope | Required baseline / analysis |
| --- | --- | --- |
| 1 | `experiments/phase1/run.py --config PATH`; 150 × five policies | Copy `configs/phase1.yaml` to an ignored config, replace dataset `main` with pinned SHA, set fresh output; [historical instructions](PHASE1.md) |
| 2 | `experiments/phase2/run.py`; 600 evaluations, P4/P8 only | Exact selection/exclusions and pinned caches |
| 3 | `experiments/phase3/run.py`; 60 spaced P8 | Original completed Phase 2; `--plan` performs CPU schedule analysis |
| 4 | `experiments/phase4/run.py`; 120 revision evaluations | Original completed Phase 3; `--limit 20`, `--smoke-test` have separate documented scopes |
| 5 | `experiments/phase5/run.py`; 120 confidence revision evaluations | Original completed Phase 2 |
| 6 | `experiments/phase6/run.py --policy both`; 120 P1 evaluations | Original completed Phase 2; independent policies supported |

All full-run commands above perform inference and require a PBS GPU allocation.
Phase 1 resolves symbolic revisions and can download; pin its dataset SHA and
set offline flags before reproduction. The later runners never download. Phase 1
also runs one warm-up decode, excluded from its timing; Phases 2–6 do not.

Shared controls: BF16, 256 response positions, block length 256, deterministic
argmax, MASK ID 126336 suppressed, zero guidance, no early stopping. A forward
predicts the entire response; policy selects positions. Confidence is FP32
softmax probability of the top token after MASK suppression, stable descending
sort with lower position winning ties. Text is decoded before first EOS with
special tokens skipped. The extractor is unchanged and compares exact rationals.
The chat prompt suffix is exactly:

```text
\nSolve this problem step by step. End with '#### <final numerical answer>'.
```

Phase 1 writes `records.jsonl`, `questions.json`, CSVs, trajectories, manifest and
environment logs. Phases 2–6 write fsynced started/completed `journal.jsonl` events
with `result` records, manifests, atomic derived CSVs and private environment
metadata. Resume accepts completed records only after phase-specific validation;
partial final appends are repaired only by the owning runner. Do not use runners
as read-only auditors of original archived journals.

Offline commands with original raw inputs:

```bash
python -m dllm_unmasking.analyze_phase15 --input-dir "$RAW_RESULTS_DIR/phase1" --output-dir results/phase15_reproduced
python -m dllm_unmasking.analyze_phase25 --input-dir "$RAW_RESULTS_DIR/phase2" --output-dir results/phase25_reproduced
# Phase 6 analyzer requires an independent copy under outputs/phase6 and the
# original config/source hashes. It rewrites derived summaries, not its journal:
python -m dllm_unmasking.analyze_phase6 --run-dir outputs/phase6/retrieved/run --phase2-results "$RAW_RESULTS_DIR/phase2"
```

Later runners' `--summarize` modes rebuild only their own derived outputs; work on
copies with exact manifests. Phase 1.5 and 2.5 have different journal schemas and
must receive the corresponding phase. Consult the phase guides for detailed
revision logs, response diagnostics and matched-case files.

## Timing and uncertainty

`gpu_seconds` is a synchronized CUDA-event interval for the whole sampler,
including selection and host launch gaps; it is not summed kernel time.
`wall_seconds` is sampler wall time, excluding final trajectory transfer/decoding
and loading. Phase 6 also records forward-only stream intervals, prompt-to-extraction
wall time, and main-thread nonforward CPU overhead. Older phases lack those scopes;
leave them missing in cross-phase tables. Jobs ran at different times, so a common
field name alone does not establish a controlled latency benchmark.

Paired tests use matched question IDs; no independent-proportions tests are used
for policy differences. The 20/20/20 stratified set is not representative of the
natural GSM8K difficulty distribution. Numerical correctness is distinct from
arithmetic validity, coherent text and faithful reasoning. See the canonical
report for exploratory multiplicity and parser limitations.

## Structural compatibility and historical resumes

See [the layout guide](STRUCTURE.md). Source names in historical manifests resolve
to relocated files. A baseline is accepted only if the physical SHA256 matches, or
if reversing the recorded import/path substitutions reconstructs the exact
historical bytes and expected SHA256. Current run manifests hash actual physical
files; they do not pretend the moved code has the old byte hash. New source hashes
therefore require fresh inference output directories. Resume interrupted runs from
the original `fa8971a` checkout; never edit historical manifests to bypass a mismatch.
The configs are moved byte-for-byte, so settings, output paths and configuration
hashes are unchanged. No historical artifact is relocated or rewritten.

Phase 7's existing `script_sha256` identifies the pre-structural generator bytes.
The generator verifies their exact reconstruction before reporting that digest.
The [source-layout record](../src/dllm_unmasking/source_layout.json) documents all
substitutions, so metadata and numerical tables reproduce without changing the
canonical historical provenance file.
