# Phase 5: confidence-based parallel decoding with remasking

Phase 5 adds only `confidence_random_remask` and
`confidence_low_confidence_remask`. Earlier implementations/results are unchanged.
The baseline is the saved **Phase 2 confidence P8, 26/60**, never rerun.

## Fixed controls and state semantics

Reuse exactly `outputs/phase2/selection.json` (60 questions, 20/20/20 strata),
model revision `852152c60bdfed4632635bfcf3e83408ad3f65bf`, GSM8K main/test revision
`740312add88f781978c0658806c59bc2815b9866`, original prompt/EOS-prefix extraction,
BF16, 256-token full block, inference seed 42 and **32 forwards per evaluation**.
The runner verifies question content/IDs, selection hash, recorded seeds, baseline
source hashes, trajectories and baseline numerical extraction. No new dataset,
selection, tokenizer version, confidence score, baseline run, or model pilot is used.

At each step:

1. Eligible revision positions are `committed − already_revised`, sorted by position.
   No prompt/future position is eligible. Step zero has none; later steps have two.
2. Random selection uses a private `random.Random(314159 + question_id)`, exactly
   Phase 4's convention. Low-confidence selection uses the lowest **original
   commitment probabilities**, with lower-position ties. Original probabilities
   are immutable. Every revised position is permanently ineligible for further revision.
3. Mask only those revision positions before **one** model forward.
4. Rank positions that have **never** been committed by the Phase 2 score:
   argmax-token softmax probability, computed in FP32 after MASK suppression using
   log-sum-exp. Set every previously committed position's new-selection score to
   negative infinity. Select top eight with stable descending ranking; equal
   scores favor lower positions, exactly as Phase 2 confidence.
5. Commit these eight new predictions and the fresh replacement predictions together.
   All use the same pre-forward masked state. Other committed tokens are untouched.

A MASK test alone would be wrong: temporarily remasked tokens are **not new**.
The explicit `ever_committed` mask excludes them even if their confidence is highest.
Confidence ranking can change after remasking; no extra forward computes a pre-remask
ranking. Unlike Phase 4 there is no fixed spaced schedule. Scores come directly from
its reused Phase 2 predictor helper, not a new uncertainty metric. No stochastic
sampling is used; revision RNG never consumes global model/Python RNG state.

Invariants: eight distinct initial commits per step; 256 distinct initial positions;
net MASK count decreases by eight each step; all positions filled after 32 steps;
no repeated revision; 62 attempts per evaluation (30 first-half, 32 second-half).
Both policies have identical budgets and eligibility rules. Replacement may equal
original; the changed flag records the actual equality test. Nonfinite selected or
new-position probabilities stop execution. EOS/special tokens may be revised if
eligible; the sampler never early-stops.

**Full budget: 120 evaluations × 32 forwards = 3,840 forwards.** There are 7,440
revision attempts, but they add no forward calls. No warm-up/pilot/smoke runs are
automatically performed. First-evaluation initialization and remasking/assertion/
selection overhead remain in measured sampler GPU/wall time. Four-hour PBS walltime
is a request, not a guaranteed runtime estimate.

## Journaling, resume and outputs

All outputs stay under `outputs/phase5/`; the default run is `outputs/phase5/run`.
A lock excludes concurrent writers. The manifest fixes config, all 60 IDs, source
hashes, baseline/selection hashes, dynamic policy semantics and pass budget.
A fsynced write-ahead start precedes each evaluation; full completion is fsynced
before summaries. Resume skips only validated completions. Replay checks initial
eligibility, confidence order/ties, RNG/low-confidence revision choices, original
values/probabilities, replacement logs, uniqueness, net MASK counts, final IDs,
seeds and primary extraction. Invalid records fail instead of being overwritten.
Only an incomplete final append in the **new** Phase 5 journal can be repaired.
Previous scientific artifacts are never repaired or changed.

Saved scores cannot certify globally highest-confidence choices without unselected
logits; source inspection and deterministic fixture tests address the actual selection
implementation. A changed source/config/baseline requires a fresh Phase 5 directory.

Files:

- `journal.jsonl`, `manifest.json`, `environment-JOBID.json`: full immutable-input
  provenance, per-result token IDs/responses/trajectories, private runtime metadata.
- `results.csv`, `summary.csv`: correctness by policy/stratum and overall, extraction
  failures, fallback/marker counts, word/token repetition, changed counts/fractions,
  forward passes and GPU/wall timing. Partial denominators remain explicit.
- `revisions.csv`: original/replacement IDs and confidence, position, zero-based
  revision step, decoding half and token-change flag.
- `revision_confidence.csv`: old/replacement means and quartiles by policy and half,
  with event denominators; these events are not independent accuracy observations.
- `response_diagnostics.csv`: reused text/token/EOS diagnostics; final-EOS timing is
  revision-aware last commitment, not its original initial reveal.
- `paired_phase2.csv`: matched recoveries/regressions against confidence P8,
  exact two-sided McNemar tests, Holm adjustment over the two policies, question-level
  bootstrap intervals (10,000 draws, seed 42) preserving observed stratum counts.
- `matched_cases.csv`, `matched_report.md`: all numerical improvements/regressions
  and their full visible outputs, including Phase 5 raw text for human interpretation.

Numerical correctness is not proof of faithful mathematical reasoning. Changed
outputs can reflect formatting, termination or repetition changes, not necessarily
arithmetic/reasoning correction. Full-token repetition includes special/post-EOS
positions; visible-word diagnostics use the stored decoded prefix. No thresholds
are presented as ground-truth degeneration labels.

## CPU verification

From the repository root, using a CPU-capable Torch/NumPy/PyYAML environment:

```bash
export PHASE2_RESULTS="$HOME/Documents/dllm-unmasking-results/phase2"
python -m unittest -v test_phase5
python -m py_compile phase5_sampler.py run_phase5.py test_phase5.py
bash -n run_phase5.pbs
python -c 'import phase5_sampler, run_phase5, phase2_samplers, phase2_data, run_phase2, evaluation, analyze_phase25, analyze_phase15'
```

Tests use fixture logits only: changing confidence ranks, adversarial highest-score
revised positions excluded from initial commits, common-state input masks/fresh
replacements, preserved prompt/other committed tokens, stored-confidence selection,
deterministic ties and independent RNG, 256/62 coverage and mask invariants,
nonfinite rejection, saved-record replay/corruption rejection, journal recovery,
output schemas and the actual 60-question/26-correct baseline. No GPU is used locally.

## Manual transfer and submission (not executed)

On the Mac, from the repository root:

```bash
export HPC_TARGET='YOUR_LOGIN@YOUR_HPC_HOST'
export HPC_REPO='/path/to/existing/dllm-unmasking-dynamics'
scp PHASE5.md phase5_config.yaml phase5_sampler.py run_phase5.py run_phase5.pbs test_phase5.py "$HPC_TARGET:$HPC_REPO/"
ssh "$HPC_TARGET"
```

The completed Phase 2 files at `outputs/phase2/run`, exact selection, and read-only
helpers listed by the import check must already exist. Missing/different dependencies
must be investigated; do not overwrite earlier operational code to force validation.

Inside your HPC session:

```bash
cd /path/to/existing/dllm-unmasking-dynamics
module load anaconda2025/2025-gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dllm-unmasking
export HF_HOME="$SCRATCH_DIR/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PHASE2_RESULTS='outputs/phase2/run'
python -c 'import phase5_sampler, run_phase5, phase2_samplers, phase2_data, run_phase2, evaluation, analyze_phase25, analyze_phase15'
python -m unittest -v test_phase5
python run_phase5.py --check-cache
export PHASE5_MODE=full
PBS_VARS='HF_HOME,PHASE5_MODE'
if printenv HF_HUB_CACHE >/dev/null; then PBS_VARS="$PBS_VARS,HF_HUB_CACHE"; fi
JOB_ID=$(qsub -v "$PBS_VARS" run_phase5.pbs)
printf '%s\n' "$JOB_ID"
qstat -f "$JOB_ID"
# Once running:
tail -f "outputs/phase5/logs/job-${JOB_ID}.log"
```

PBS uses **`#PBS -P YOUR_PROJECT_CODE`**, `gpu_bwp6000`, one GPU, four CPU cores,
64 GB RAM, `dllm-unmasking`, and **`set -eo pipefail`**. The script changes to
submission root, checks imports, then runs offline file-only cache preflight and
full Phase 5. It supports only `PHASE5_MODE=full`. Cached weights/tokenizer must be
available at the pinned revision; no downloads or baseline inference are allowed.
For resume, submit the same command after the previous job has stopped.

On HPC, summary-only rebuild is `python run_phase5.py --summarize`, with no model
loading. On your Mac, retrieve separately:

```bash
mkdir -p outputs/phase5/retrieved
scp -r "$HPC_TARGET:$HPC_REPO/outputs/phase5/run" "$HPC_TARGET:$HPC_REPO/outputs/phase5/logs" outputs/phase5/retrieved/
```

No Phase 5 effect or accuracy has been measured locally. No jobs, commits or pushes
are performed by implementation or verification.
