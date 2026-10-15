> Commands assume an editable installation of the full repository. See the [reproduction guide](../../docs/REPRODUCIBILITY.md).

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
python -m unittest -v tests.test_phase5
python -m py_compile src/dllm_unmasking/phase5_sampler.py experiments/phase5/run.py tests/test_phase5.py
bash -n experiments/phase5/run.pbs
python -c 'import dllm_unmasking.phase5_sampler, experiments.phase5.run, dllm_unmasking.phase2_samplers, dllm_unmasking.phase2_data, experiments.phase2.run, dllm_unmasking.evaluation, dllm_unmasking.analyze_phase25, dllm_unmasking.analyze_phase15'
```

Tests use fixture logits only: changing confidence ranks, adversarial highest-score
revised positions excluded from initial commits, common-state input masks/fresh
replacements, preserved prompt/other committed tokens, stored-confidence selection,
deterministic ties and independent RNG, 256/62 coverage and mask invariants,
nonfinite rejection, saved-record replay/corruption rejection, journal recovery,
output schemas and the actual 60-question/26-correct baseline. No GPU is used locally.

## Manual deployment and submission (not executed)

Follow the [whole-tree deployment and editable-install instructions](../../docs/REPRODUCIBILITY.md#model-cache-and-hpc-execution).
Do not copy individual files into a flat directory. Restore the exact selection
and required completed baselines separately. All commands run from the repository
root after dependency/package installation. Use a fresh output directory for this
source layout; historical interrupted runs require their original checkout.

```bash
# On HPC after deployment and restoration of baseline files/cache:
python -m experiments.phase5.run --check-cache
cp experiments/phase5/run.pbs run_phase5.local.pbs
# Edit the local copy's allocation and site settings before submission.
export PHASE5_MODE=full
PBS_VARS='HF_HOME,PHASE5_MODE'
if printenv HF_HUB_CACHE >/dev/null; then PBS_VARS="$PBS_VARS,HF_HUB_CACHE"; fi
JOB_ID=$(qsub -v "$PBS_VARS" run_phase5.local.pbs)
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
mkdir -p outputs/phase5/retrieved
scp -r "$HPC_HOST:$PROJECT_ROOT/outputs/phase5/run" "$HPC_HOST:$PROJECT_ROOT/outputs/phase5/logs" outputs/phase5/retrieved/
```
