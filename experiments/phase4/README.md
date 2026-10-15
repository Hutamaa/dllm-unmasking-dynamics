> Commands assume an editable installation of the full repository. See the [reproduction guide](../../docs/REPRODUCIBILITY.md).

# Phase 4: online remasking under the Phase 3 forward budget

Only `random_remask` and `low_confidence_remask` are new. All earlier implementations,
selections and artifacts remain unchanged. No GPU inference, cache downloads,
HPC submission, commit or push is performed locally.

## Design

The exact Phase 3 schedule is reused: for `t=2b+r`, reveal
`[16b+r+2j for j=0,...,7]`, `b=0,...,15`, `r=0,1`. There are 32 steps, eight new
positions per step, and 256 new positions exactly once. At each step after zero,
revision candidates are only previously committed generation positions that have
never been revised. Select two candidates (or fewer if not enough eligible):

- Random: uniform sampling without replacement from sorted eligible positions,
  using private `random.Random(314159 + question_id)`.
- Low-confidence: ascending probability stored at each position's original
  commitment, ties broken by lower position. These are stored prediction probabilities,
  **not recomputed current-state uncertainty**. Revised tokens are ineligible thereafter.

Replace those positions with MASK **before the single forward**. The model predicts
both the eight new positions and revised positions from this same masked input;
commit all ten together. Replacements are fresh predictions from that forward,
not cached original tokens. Prompt positions are never candidates. There is no
remasking or revision on step zero. MASK is excluded from output prediction.

For both policies, eligible count before step t≥1 is `8t−2(t−1)=6t+2`, so two
revisions are always possible. Exact budget: **62 distinct revisions** per evaluation
(30 in steps 0–15, 32 in steps 16–31). Some replacements may equal their originals.
Temporary remasking increases MASK count by two, then committing new+revised
positions decreases it by ten: the **net decrease is exactly eight** every step.
After step 31 all 256 generation positions are filled. Exactly 32 model forwards;
no counterfactual probes, additional uncertainty passes, warm-up or baseline reruns.

Softmax probabilities use the Phase 2 FP32 log-sum-exp implementation with MASK
suppression. Selected probabilities must be finite and in [0,1]; otherwise stop.
Argmax token ties and revision-confidence ties are deterministic. The model inference
seed remains 42; argmax prediction uses no token-sampling RNG. Revision randomness
uses its separate private stream and does not change global Torch/Python RNG state.

This design **does not isolate reasoning correction from formatting or repetition
correction**. Correct extracted answers and changed tokens do not establish sound
reasoning. Those are separate downstream measurements and human-review questions.

## Exact inputs and offline controls

Reuse `outputs/phase2/selection.json` and completed `outputs/phase3/run` as read-only
inputs. The same 60 question IDs, texts, gold solutions, 20/20/20 strata, prompt,
selection seed, inference seed, EOS-prefix decoding and answer extractor are required.
The pinned model revision is `852152c60bdfed4632635bfcf3e83408ad3f65bf`; GSM8K main/test
revision is `740312add88f781978c0658806c59bc2815b9866`. Source hashes and selection hash
must match the completed Phase 3 manifest. Missing baseline questions or mismatches
stop the run; no substitute questions or baseline inference are allowed.

Model/tokenizer loading uses the existing scratch Hugging Face cache with local-files-only
and `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. There is no
dataset reload: the unchanged saved selection already contains the exact questions.
`--check-cache` validates baseline/selection and inspects pinned cached files only,
without model loading or inference. The existing Conda environment supplies Torch,
Transformers, NumPy, PyYAML and analysis dependencies; PBS does not install packages.

## Modes and forward budget

| Mode | Questions | New evaluations | Forward passes | Directory |
| --- | ---: | ---: | ---: | --- |
| `--smoke-test` | 2 | 4 | 128 | `outputs/phase4/smoke` |
| `--limit 20` | 20 | 40 | 1,280 | `outputs/phase4/run` |
| Full | 60 | 120 | 3,840 | `outputs/phase4/run` |

Question ordering interleaves the three unchanged strata deterministically. The
20-question pilot uses 7/7/6 questions; smoke uses one each from the first two
strata. Original question IDs/texts never change. Policy execution order alternates
across questions. The pilot is the first 20 questions of the same 60-question
production experiment: a later full run resumes that directory and skips the
40 completed pilot evaluations. No pilot evaluations are duplicated. The manifest
always fixes all 60 production IDs; `--limit` is only a work limit per invocation.
Smoke has a separate directory and does not contribute to production results.
Changing smoke/production mode, source or configuration requires a fresh directory.
`--output-dir` may specify another directory **under outputs/phase4**. No smoke
or pilot is automatically run before a full run; choose the PBS mode to submit.
Timings include any first-evaluation initialization effects, and 4-hour walltime is
an allocation request rather than a guaranteed runtime estimate.

## Outputs and diagnostics

Every directory has a lock and exact provenance manifest. A fsynced `started` event
precedes inference; a fsynced complete result follows. Resume skips durable completions
and restarts incomplete evaluations only. Only a partial final append to the new
Phase 4 journal may be repaired. Duplicate/unknown completed keys fail. Derived CSVs
are atomically rebuilt from the journal. Earlier journals are read without recovery.

- `journal.jsonl`: full question/result, final token IDs, responses and step logs.
  Each step preserves eight **new** `positions`, their original commit predictions,
  plus `revisions`, eligibility/budget, MASK counts and pass counts. A revised position's
  first-commit token may differ from its final ID; replay revisions to audit final tokens.
- `revisions.csv`: position, original/replacement token, original/replacement confidence,
  revision step (zero-based), decoding half and `changed` flag. No token appears twice
  among a question-policy's revisions.
- `results.csv`, `summary.csv`: overall/stratum accuracy, extraction failures, fallback,
  marker presence, maximum identical-word runs, repeated-token fractions, revised/changed
  counts and changed fractions, forward passes, GPU and wall timing.
- `revision_confidence.csv`: original and replacement confidence quantiles and means,
  by policy and decoding half, with revision denominators. These event distributions are
  descriptive; revisions are not independent accuracy observations.
- `response_diagnostics.csv`: unchanged offline output features. Token repetition covers
  all 256 IDs including special/post-EOS tokens; visible-word measures use stored decoded
  responses. EOS timing is revision-aware **last commitment of the final first-EOS position**,
  replacing the irreversible-trajectory helper's first-reveal timing.
- `paired_phase3.csv`: exactly matched questions, Phase 3/4 correctness transitions,
  exact two-sided McNemar p-values and Holm correction across the two strategies;
  question-level bootstrap intervals preserve observed stratum counts.
- `matched_cases.csv`, `matched_report.md`: all observed improvements/regressions with
  full visible responses and Phase 4 raw text; no automatic reasoning-error attribution.
- `environment-JOBID.json`, logs: private runtime/progress information.

Summaries from partial/pilot runs are exploratory with explicit denominators. Neither
parsing success nor raw post-EOS text is automatically counted as reasoning success.

## CPU-only verification

From the repository root in a CPU-capable Python environment:

```bash
export PHASE3_RESULTS="$HOME/Documents/dllm-unmasking-results/phase3"
python -m unittest -v tests.test_phase4
python -m py_compile src/dllm_unmasking/phase4_sampler.py experiments/phase4/run.py tests/test_phase4.py
bash -n experiments/phase4/run.pbs
```

Tests verify common-state input masks, fresh replacements, unchanged prompt, exact
spaced new-position schedule, no duplicate revisions, matching budgets/eligibility,
net MASK-count decrement, final coverage, 32 forwards, deterministic random replay,
confidence ties, RNG isolation, nonfinite probability rejection, actual baseline
validation, journal recovery/duplicate rejection and summary schemas with temporary
fixture results. They load no pretrained models. Local smoke-test execution is not
permitted: actual inference requires a PBS GPU allocation.

## Manual deployment and submission (not executed)

Follow the [whole-tree deployment and editable-install instructions](../../docs/REPRODUCIBILITY.md#model-cache-and-hpc-execution).
Do not copy individual files into a flat directory. Restore the exact selection
and required completed baselines separately. All commands run from the repository
root after dependency/package installation. Use a fresh output directory for this
source layout; historical interrupted runs require their original checkout.

```bash
# On HPC after deployment and restoration of baseline files/cache:
python -m experiments.phase4.run --check-cache
cp experiments/phase4/run.pbs run_phase4.local.pbs
# Edit the local copy's allocation and site settings before submission.
export PHASE4_MODE=full
PBS_VARS='HF_HOME,PHASE4_MODE'
if printenv HF_HUB_CACHE >/dev/null; then PBS_VARS="$PBS_VARS,HF_HUB_CACHE"; fi
JOB_ID=$(qsub -v "$PBS_VARS" run_phase4.local.pbs)
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
mkdir -p outputs/phase4/retrieved
scp -r "$HPC_HOST:$PROJECT_ROOT/outputs/phase4/run" "$HPC_HOST:$PROJECT_ROOT/outputs/phase4/logs" outputs/phase4/retrieved/
```
