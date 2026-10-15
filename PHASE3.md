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

New files: `phase3_schedule.py`, `phase3_sampler.py`, `run_phase3.py`,
`phase3_config.yaml`, `run_phase3.pbs`, `test_phase3.py`, this document.
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
python -m unittest -v test_phase3
python -m py_compile phase3_schedule.py phase3_sampler.py run_phase3.py test_phase3.py
bash -n run_phase3.pbs
python run_phase3.py --plan --phase2-results "$PHASE2_RESULTS"
```

Tests verify exact coverage/spacing/determinism, prompt preservation, irreversible
reveals, 32 toy forwards, common-state simultaneous predictions, selection
matching/rejection, and crash/resume/duplicate handling. Toy logits change if
any within-batch update occurs, so greedy recomputation would fail the test.
The planning command performs no model loading or cache access.

## Manual deployment and submission (not executed)

On your Mac, from the repository, replace only the host/login/repository/cache
placeholders with the existing Phase 2 values:

```bash
export HPC_TARGET='YOUR_LOGIN@YOUR_HPC_HOST'
export HPC_REPO='/path/to/existing/dllm-unmasking-dynamics'
scp phase3_schedule.py phase3_sampler.py run_phase3.py phase3_config.yaml run_phase3.pbs test_phase3.py PHASE3.md "$HPC_TARGET:$HPC_REPO/"
# Stage read-only dependencies without overwriting older operational files:
ssh "$HPC_TARGET" "mkdir -p '$HPC_REPO/outputs/phase3/dependencies'"
scp analyze_phase25.py analyze_phase15.py "$HPC_TARGET:$HPC_REPO/outputs/phase3/dependencies/"
ssh "$HPC_TARGET"
```

Earlier Phase 2 implementation files, `evaluation.py`, completed outputs at
`outputs/phase2/run`, and `outputs/phase2/selection.json` must already exist.
The staged diagnosis helpers are installed below only if missing. An existing
helper must match the staged copy or deployment stops; it is never overwritten.
The cache and old results do not need upload or downloading.

Inside your HPC session:

```bash
cd /path/to/existing/dllm-unmasking-dynamics
for helper in analyze_phase25.py analyze_phase15.py; do
    if [[ -e "$helper" ]]; then
        cmp "$helper" "outputs/phase3/dependencies/$helper" || { echo "Dependency differs; stop without overwriting"; exit 1; }
    else
        cp -n "outputs/phase3/dependencies/$helper" "$helper"
    fi
done
module load anaconda2025/2025-gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dllm-unmasking
export HF_HOME='/path/to/existing/offline/phase2/huggingface-cache'
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PHASE2_RESULTS='outputs/phase2/run'
python -m unittest -v test_phase3
python run_phase3.py --plan
python run_phase3.py --check-cache
# If Phase 2 used separate HF_HUB_CACHE, export and pass it as well.
PBS_VARS='HF_HOME'
if printenv HF_HUB_CACHE >/dev/null; then PBS_VARS="$PBS_VARS,HF_HUB_CACHE"; fi
JOB_ID=$(qsub -v "$PBS_VARS" run_phase3.pbs)
printf '%s\n' "$JOB_ID"
qstat -f "$JOB_ID"
# Once running:
tail -f "outputs/phase3/logs/job-${JOB_ID}.log"
```

The script uses project **gs_ccds_taisp**, queue **gpu_bwp6000**, module
**anaconda2025/2025-gpu**, and environment **dllm-unmasking**. `--check-cache`
inspects existing pinned files only; it does not load weights or run inference.
Do not enable downloads if preflight fails. Submit once, and resubmit the same
command after a previous job stops if resume is needed.

After completion, on your Mac:

```bash
mkdir -p outputs/phase3/retrieved
scp -r "$HPC_TARGET:$HPC_REPO/outputs/phase3/run" "$HPC_TARGET:$HPC_REPO/outputs/phase3/logs" outputs/phase3/retrieved/
```

On HPC, summary-only rebuild is `python run_phase3.py --summarize` and never
loads a model. No scientific conclusion about spacing is justified before the
new run and paired comparisons exist; even then, timing/direction confounding
must be considered.
