> Commands assume an editable installation of the full repository. See the [reproduction guide](../../docs/REPRODUCIBILITY.md).

# Phase 3: spaced left-prioritized parallel unmasking

Only `ltr_spaced` is implemented and run by the new runner. All earlier code and
artifacts are preserved. No remasking, baseline rerun, model pilot or warm-up.
No local model inference, HPC access, submission, commit or push has occurred.

## Exact schedule and verification

Response positions are zero-based. For step `t=2*b+r`, select

```text
B_t = [16*b + r + 2*j for j=0,...,7]
b=0,...,15; r=0,1 (even offsets, then odd offsets)
```

First six batches:

```text
0: [0, 2, 4, 6, 8, 10, 12, 14]
1: [1, 3, 5, 7, 9, 11, 13, 15]
2: [16, 18, 20, 22, 24, 26, 28, 30]
3: [17, 19, 21, 23, 25, 27, 29, 31]
4: [32, 34, 36, 38, 40, 42, 44, 46]
5: [33, 35, 37, 39, 41, 43, 45, 47]
30: [240, 242, 244, 246, 248, 250, 252, 254]
31: [241, 243, 245, 247, 249, 251, 253, 255]
```

Each position p uniquely determines `b=p//16`, `r=p%2`, and
`j=(p%16)//2`. Hence **256 unique positions exactly once, 32 batches × 8**.
Each batch has sorted gap **2**, spread **14**, adjacent pairs **0**, longest
contiguous run **1**. One forward produces all baseline predictions; one
indexed assignment simultaneously commits all eight. No forward or state
update occurs between winners. The schedule does not use prediction scores.

This is **left-prioritized interleaving inside 16-position tiles**, not strict
monotone LTR. Average position is `16*b+7` then `16*b+8`; contiguous LTR uses
`16*b+3.5` then `16*b+11.5`. Even positions in the latter half of a tile move
one step earlier, odd positions in its first half move one step later.
Spacing therefore also changes context timing/directionality. A result would
not isolate adjacency from these factors or establish token-dependency strength.

Pre-execution analysis is already saved in `outputs/phase3/plan/`:
`schedule.csv`, `geometry_comparison.csv`, `position_dynamics.csv`, one PNG,
and `report.md`. It compares the planned deterministic trajectory with the
60 recorded trajectories each for LTR P8, RTL P8 and confidence P8.
LTR/RTL have 7 adjacent pairs and spread 7; confidence averages 3.745 adjacent
pairs and spread 92.114. There are no Phase 3 accuracy results yet.

## Inputs and outputs

The same `outputs/phase2/selection.json` is reused without copying/reselecting:
60 IDs, 20/20/20 strata. Its SHA256 must match the completed Phase 2 manifest.
Every question's text, gold, stratum, calculation count and inference seed must
match all ten stored Phase 2 conditions. Missing or mismatched IDs stop execution
instead of running a reduced or substituted selection.

Pinned model: `852152c60bdfed4632635bfcf3e83408ad3f65bf`.
Pinned GSM8K test: `740312add88f781978c0658806c59bc2815b9866`.
Prompt, BF16, MASK ID, selection seed 20261014 and inference seed 42 match
Phase 2. Position seeds `42 + question_id` are retained for comparison; this
deterministic policy consumes no position RNG. The unchanged primary extractor
and EOS-prefix decoding are reused. Generated token count remains 256 even
if an EOS is generated early. No tokenizer or dataset/model downloads occur.
The existing selected texts are used directly; no dataset reloading is needed.

New files: `src/dllm_unmasking/phase3_schedule.py`, `src/dllm_unmasking/phase3_sampler.py`, `experiments/phase3/run.py`,
`configs/phase3.yaml`, `experiments/phase3/run.pbs`, `tests/test_phase3.py`, this document.
Read-only helpers from Phase 2/2.5 are reused; no earlier helper is invoked to
run another policy. Provenance hashes include all reused scientific helpers.

The default independent run directory is `outputs/phase3/run/`:

- `journal.jsonl`: fsynced write-ahead starts and complete records, including
  full responses/token IDs/trajectories/confidence/timings. Only an incomplete
  trailing append is repaired on resume; duplicate completions fail.
- `manifest.json`: exact config, schedule, selection and Phase 2 input hashes,
  source hashes, 60-evaluation budget.
- `results.csv`, `summary.csv`, `response_diagnostics.csv`: incrementally
  rebuilt atomically from durable completions; repetition, formatting and EOS
  diagnostics reuse the existing offline definitions.
- `paired_phase2_comparisons.csv`: matched accuracy differences, discordant
  counts, exact McNemar tests with Holm adjustment across ten comparisons,
  question-level bootstrap intervals preserving observed strata. P8 comparisons
  share step budgets; P4 comparisons have different budgets and are reference
  contrasts, not spacing-only effects. Partial runs report matched denominators.
- `environment-JOBID.json`, `logs/job-JOBID.log`: private runtime/progress logs.

Resume locks prevent concurrent writers. Changed source/config/input hashes
require a fresh run directory. Only incomplete new evaluations restart; all
60 completed Phase 3 records exit without loading a model. Raw Phase 2 data
and completed old evaluations are never repaired or rewritten.

**Budget: 60 new evaluations × 32 forwards = 1,920 forwards.** No extra warm-up
passes. Timing includes first-evaluation initialization effects. The PBS request
is 1 GPU, 4 CPU cores, 64 GB RAM and 4 hours; runtime is not guaranteed, and
interrupted jobs can resume. Existing raw Phase 2 results, selection, and cache
must be present; the script fails closed on missing pinned cache files.

## Local CPU verification and planning

Use a local Python with Torch (CPU), NumPy, Matplotlib and PyYAML installed:

```bash
export PHASE2_RESULTS="$HOME/Documents/dllm-unmasking-results/phase2"
python -m unittest -v tests.test_phase3
python -m py_compile src/dllm_unmasking/phase3_schedule.py src/dllm_unmasking/phase3_sampler.py experiments/phase3/run.py tests/test_phase3.py
bash -n experiments/phase3/run.pbs
python -m experiments.phase3.run --plan --phase2-results "$PHASE2_RESULTS"
```

Tests verify exact coverage/spacing/determinism, prompt preservation, irreversible
reveals, 32 toy forwards, common-state simultaneous predictions, selection
matching/rejection, and crash/resume/duplicate handling. Toy logits change if
any within-batch update occurs, so greedy recomputation would fail the test.
The planning command performs no model loading or cache access.

## Manual deployment and submission (not executed)

Follow the [whole-tree deployment and editable-install instructions](../../docs/REPRODUCIBILITY.md#model-cache-and-hpc-execution).
Do not copy individual files into a flat directory. Restore the exact selection
and required completed baselines separately. All commands run from the repository
root after dependency/package installation. Use a fresh output directory for this
source layout; historical interrupted runs require their original checkout.

```bash
# On HPC after deployment and restoration of baseline files/cache:
python -m experiments.phase3.run --check-cache
cp experiments/phase3/run.pbs run_phase3.local.pbs
# Edit the local copy's allocation and site settings before submission.
PBS_VARS='HF_HOME'
if printenv HF_HUB_CACHE >/dev/null; then PBS_VARS="$PBS_VARS,HF_HUB_CACHE"; fi
JOB_ID=$(qsub -v "$PBS_VARS" run_phase3.local.pbs)
qstat -f "$JOB_ID"
```

The local copy is ignored by Git. Do not submit concurrent writers. For Phase 2,
retain the established `CONDA_BASE`/`LLADA_ENV`/optional module and explicit cached
Parquet settings if used; pass those variables to PBS too. For Phase 4,
`PHASE4_MODE=smoke`, `pilot`, and `full` retain their original documented scopes.
Phase 6 also supports independent `left_to_right` and `confidence` policies.
These are manual commands only; no job is submitted by CPU tests or analysis.

Retrieve results into a separate local directory after the job finishes:

```bash
mkdir -p outputs/phase3/retrieved
scp -r "$HPC_HOST:$PROJECT_ROOT/outputs/phase3/run" "$HPC_HOST:$PROJECT_ROOT/outputs/phase3/logs" outputs/phase3/retrieved/
```
