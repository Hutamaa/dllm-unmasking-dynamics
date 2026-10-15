# Phase 6: matched sequential token-selection benchmark

Only new P1 `left_to_right` and `confidence` evaluations are performed. Existing
P4/P8 results and all earlier source/config/artifacts are preserved. No remasking,
revision, accuracy optimization, baseline rerun, model download, GPU pilot, warm-up,
local GPU inference, HPC submission, commit or push.

## Exact semantics and mathematical consistency

The unchanged **Phase 1 `samplers.generate`** is reused, not copied. Phase 2's
production guard intentionally only accepts P4/P8; it is left unchanged. Phase 1
already accepts the new requested P1 setting. In a full 256-token block, its
linear schedule is `256 // 256 = 1` commitment per step, no remainder.

Phase 1 and Phase 2 both use direct LLaDA logits, deterministic argmax with lower
vocabulary-ID ties, FP32 selected-token softmax probability after MASK suppression,
irreversible reveals, and stable descending confidence ranking with lower-position
ties. For a full block their formulas are mathematically identical: gathering
`logit(token) − logsumexp(logits)` then exponentiating versus gathering from the
same log-probability vector. LTR selects the leftmost remaining position; confidence
selects the maximal probability among uncommitted positions. P1 is exactly one-token
application of those rules, not standard autoregressive `model.generate()`.
The architecture and attention behavior remain unchanged. EOS never terminates
the sampling loop; all 256 positions are initially masked and filled exactly once.
The decoder/evaluator uses the prefix before first EOS and the same answer extractor.

CPU fixtures compare every model input, selected position, token and probability
against a one-token mathematical extrapolation of Phase 2's existing predictor/ranking.
This reference loop exists only in tests. Nonuniform/tied scores, dynamic confidence
ranking, initial masks, unique coverage, and 256-call counts are checked.

## Inputs, pinned settings and reproducibility

`outputs/phase2/selection.json` is reused without reselection: 60 questions, 20 each
in calculation-count strata 0–2, 3–4, 5+. Selection SHA256 must equal the completed
Phase 2 manifest. Prompts/question text/gold/strata/seeds are matched to the saved
P4/P8 evaluations. Pinned revisions:

- LLaDA-8B-Instruct: `852152c60bdfed4632635bfcf3e83408ad3f65bf`.
- GSM8K main/test: `740312add88f781978c0658806c59bc2815b9866`.
- Selection seed: 20261014; inference seed: 42; position seed: `42 + question_id`.
- BF16, length/block 256, P1, 256 steps. Temperature zero/argmax, no guidance.

The cached model/tokenizer and saved selection are used offline. All earlier
policies/evaluations are read-only. No dataset reload or newer revision is substituted.
Missing pinned cache or mismatched source/question/selection fails closed.

## Budget, policies and restart

**120 evaluations × 256 calls = 30,720 forward passes**. Each policy alone uses
60 × 256 = 15,360. The same source/config manifest always describes both policies:

```bash
python run_phase6.py --policy left_to_right
python run_phase6.py --policy confidence
python run_phase6.py --policy both
```

These inference commands require a PBS GPU allocation. You can finish and inspect
LTR first, then resume with confidence; old completions are validated and skipped.
Selecting both after either policy completes runs only missing evaluations.
Never submit two concurrent writers to the same run directory; an exclusive lock
rejects them. Completion is **both policies × all 60 IDs**, not the current CLI scope.

The default is `outputs/phase6/run`. A write-ahead start is fsynced before each
evaluation; completion is fsynced before summaries. Manifest locks configuration,
selection/baseline/source hashes and policy set. Every record includes config hash.
Resume validates IDs/policies/seeds, no repeated reveals/revisions, 256 calls,
final token IDs, extraction and timing fields. Partial final journal appends are
repaired only by the runner; completed corrupt/duplicate records fail. Offline
analysis refuses a truncated journal instead of repairing it. Changed config/code/
baseline requires a fresh output directory. No earlier journal is rewritten.

PBS requests one GPU, four CPUs, 64 GB and **8 hours**. Observed Phase 2 decode
seconds/forward provide only a rough scale for 30,720 calls: different call overhead,
GPU load, P1 dispatch, event instrumentation, summaries and environment can change
runtime. Eight hours is a conservative allocation with restart support, not a
promised runtime. No unnecessary confidence/candidate forward calls occur.

## Timing definitions (do not conflate them)

`ForwardMeter` is a transparent callable wrapper around the loaded model. It forwards
the identical inputs/kwargs once and records events/host clocks. It does not create
a new sampler, prediction, synchronization inside a step, or candidate-scoring call.

- **`gpu_seconds`**: unchanged sampler CUDA-event elapsed decode-stream interval.
  Includes GPU selection/score operations and host launch gaps; not pure kernel time.
- **`model_forward_gpu_stream_seconds`**: sum of 256 per-forward CUDA-stream intervals.
  Excludes selection operations outside those intervals, but can still include CPU
  launch gaps within a forward. Not a profiler's summed kernel execution time.
- **`wall_seconds`**: unchanged synchronized decode wall interval, excluding prompt
  tokenization, final trajectory CPU serialization, text decode and persistence.
- **`evaluation_wall_seconds`**: synchronized prompt preparation through generation,
  trajectory serialization, decoding, visible-response retokenization and numerical
  extraction. Excludes model loading, journal/fsync, summaries and record audits.
- **`sampler_nonforward_main_thread_cpu_seconds`**: sampler main-thread CPU time minus
  delegated model-call thread CPU time. Includes host allocation, selection dispatch,
  timing instrumentation and trace serialization. Excludes GPU wait and worker CPU
  threads. An observable host-overhead measure, not isolated pure-selection cost.
- **`model_forward_host_enqueue_wall_seconds`**: accumulated host model-call wall time;
  may include any internal blocking, so it is not labeled CPU compute time.

No warm-up adds hidden forward calls. First-evaluation initialization effects are
included. Per-forward event instrumentation has overhead and differs from legacy
runs, so exact latency equivalence is not claimed. P4/P8's existing `gpu_seconds`
and `wall_seconds` have compatible decode scopes; their E2E, forward-only GPU and
main-thread CPU timing are **unavailable and left blank**, not reconstructed by
subtracting overlapping clocks. Check hardware/software metadata for cross-run limits.

`tokens_initially_committed=256` counts initial reveals. `tokens_before_first_eos`
is the prefix length in generated IDs. `tokens_final_decoded_response` counts the
actual visible text re-encoded with the pinned tokenizer and add_special_tokens=False;
it can differ from the prefix count because special tokens are skipped. Old runs
lack that re-encoding measurement and it is left blank.

## Output and statistical comparison

- `journal.jsonl`, `manifest.json`, environment metadata and logs: provenance and
  full per-question responses/token IDs/confidence trajectories/timings.
- `results.csv`, `response_diagnostics.csv`: only observed new P1 records, including
  marker/fallback/failure/repetition metrics and all timing/token-count fields.
- `summary.csv`: overall and stratum accuracy/counts for P1/P4/P8, with timing
  availability denominators and no imputed missing P1 accuracy.
- `accuracy_latency_table.csv`: analysis-ready matched observations across the same
  IDs, policies and P1/P4/P8. Old 240 records are observed; new 120 rows appear only
  as completed. Observed latency columns are separate from theoretical 1×/4×/8×
  forward-call reduction; that reduction is **not observed runtime speedup**.
- `paired_comparisons.csv`: within-P1 policies and each P1-versus-observed-P4/P8
  condition, exact two-sided McNemar tests and discordances, question-level paired
  bootstrap intervals (10,000, seed 42, observed stratum counts preserved), Holm
  correction over currently available rows (nine when complete).
- `completion.json`, `report.md`: total/per-policy completion and timing/statistical
  limitations. Only 60+60 completions mark the experiment complete.

Partial-run/subgroup comparisons are exploratory. Numerical correctness is not
proof of faithful mathematical reasoning. Phase 1's different 150 questions are
not treated as a matched sequential accuracy baseline.

CPU-only analysis after/while the GPU job is stopped:

```bash
python analyze_phase6.py
python run_phase6.py --summarize
```

Both analysis commands reuse old results without inference. The standalone analyzer
validates journals and never repairs them; original input/journal hashes are checked.
Use the original config; the standalone analyzer accepts --run-dir for a copied result directory.

## CPU verification

From repository root, with local CPU Torch/NumPy/PyYAML:

```bash
export PHASE2_RESULTS="$HOME/Documents/dllm-unmasking-results/phase2"
python -m unittest -v test_phase6
python -m py_compile run_phase6.py analyze_phase6.py test_phase6.py
bash -n run_phase6.pbs
python -c 'import run_phase6, analyze_phase6, samplers, phase2_samplers, phase2_data, run_phase2, evaluation, analyze_phase25, analyze_phase15'
```

Tests cover deterministic/dynamic/tied selection, exact Phase 2-rule equivalence,
256 unique irreversible reveals and 256 calls, prompt/input equality, timing scopes,
config/ID/final-state corruption rejection, pending journal and duplicate handling,
independent-policy resume, exact 60/20/20/20 matching, table scopes and missing latency.
Temporary fixtures are not experimental results. Earlier artifact hashes are checked
separately. No pretrained model or GPU is used by these tests.

## Exact manual transfer, preflight, submission and retrieval (not executed)

On your Mac, from repository root:

```bash
export HPC_TARGET='YOUR_LOGIN@YOUR_HPC_HOST'
export HPC_REPO='/path/to/existing/dllm-unmasking-dynamics'
scp PHASE6.md phase6_config.yaml run_phase6.py run_phase6.pbs test_phase6.py analyze_phase6.py "$HPC_TARGET:$HPC_REPO/"
ssh "$HPC_TARGET"
```

All existing read-only dependencies in the import command, Phase 2 completed data,
selection and pinned scratch cache must already be present. Do not overwrite prior
sources to force hash matching.

Inside HPC SSH:

```bash
cd /path/to/existing/dllm-unmasking-dynamics
module load anaconda2025/2025-gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dllm-unmasking
export HF_HOME='/path/to/established/scratch/huggingface-cache'
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PHASE2_RESULTS='outputs/phase2/run'
python -c 'import run_phase6, analyze_phase6, samplers, phase2_samplers, phase2_data, run_phase2, evaluation, analyze_phase25, analyze_phase15'
python -m unittest -v test_phase6
python run_phase6.py --check-cache
# Choose LTR first, or set both to run the entire matched benchmark:
export PHASE6_POLICY=left_to_right
PBS_VARS='HF_HOME,PHASE6_POLICY'
if printenv HF_HUB_CACHE >/dev/null; then PBS_VARS="$PBS_VARS,HF_HUB_CACHE"; fi
JOB_ID=$(qsub -v "$PBS_VARS" run_phase6.pbs)
printf '%s\n' "$JOB_ID"
qstat -f "$JOB_ID"
# Once started:
tail -f "outputs/phase6/logs/job-${JOB_ID}.log"
```

After LTR job completion and inspection, submit confidence into the same directory:

```bash
export PHASE6_POLICY=confidence
JOB_ID=$(qsub -v "$PBS_VARS" run_phase6.pbs)
```

Alternatively set `PHASE6_POLICY=both` before one submission. Resubmission skips all
validated completions. Do not run concurrent writers. PBS uses project gs_ccds_taisp,
queue gpu_bwp6000, dllm-unmasking, and **set -eo pipefail**. No downloads or baseline
runs are launched. `--check-cache` inspects pinned files without loading/inference.

After completion, on Mac:

```bash
mkdir -p outputs/phase6/retrieved
scp -r "$HPC_TARGET:$HPC_REPO/outputs/phase6/run" "$HPC_TARGET:$HPC_REPO/outputs/phase6/logs" outputs/phase6/retrieved/
python analyze_phase6.py --run-dir outputs/phase6/retrieved/run --phase2-results "$HOME/Documents/dllm-unmasking-results/phase2"
```

No P1 accuracy, GPU latency or accuracy–latency conclusion is measured locally.
