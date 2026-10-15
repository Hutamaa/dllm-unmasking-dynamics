# Phase 2: parallel unmasking

Phase 1 and 1.5 code/results remain unchanged. All new implementation is in
`phase2_samplers.py`, `phase2_data.py`, and `run_phase2.py`; configuration,
PBS template, and CPU fixture tests are separate. No HPC access, deployment,
submission, model download, real model inference, or GPU pilot is performed locally.

## Design and prepared selection

- LLaDA model revision: `852152c60bdfed4632635bfcf3e83408ad3f65bf`.
- GSM8K main/test revision: `740312add88f781978c0658806c59bc2815b9866`.
- Pinned test Parquet SHA-256:
  `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
- Selection seed **20261014**: visit all test indices in order, exclude exactly
  the 150 Phase 1 IDs, build calculation-count pools, and use a private Python
  RNG to sample 20 from each pool in order 0–2, 3–4, 5+. Sort selected IDs inside
  each pool. No model answers or Phase 1 correctness are used for selection.
- Actual selected questions are saved in `outputs/phase2/selection.json`:
  **20 / 20 / 20**, **60 distinct new IDs**, zero Phase 1 overlap. It includes
  full question texts, gold solutions, calculation counts, strata, seed,
  revision, and excluded IDs. `phase1_exclusions.json` also records the original
  Phase 1 questions-file hash. Both files were saved before inference.
- Selection was prepared locally from only the pinned public test Parquet,
  without downloading any model weights. This local dataset retrieval is not
  part of HPC deployment; the HPC runner must reuse existing cache files.
- Inference seed **42**, with position RNG seed `42 + question_id`. Random uses
  its own RNG; other strategies do not consume it. No temperature or guidance.
- Full block and response length **256**, BF16. Parallel-4 has **64** steps;
  parallel-8 has **32** steps. No other settings, especially 256 steps, are accepted.
- Five strategies at both settings: **600 evaluations**. All ten condition
  orders rotate across questions to reduce systematic timing-order bias.

Each step obtains baseline argmax tokens and confidence from one shared state.
It selects exactly 4 or 8 distinct masked response positions and writes all their
baseline tokens in one indexed assignment. MASK is suppressed as a predicted
output; EOS does not shorten the schedule. Tokens already revealed remain fixed.
LTR/RTL, random selection, confidence, and stable tie semantics match Phase 1.

`batched_influence` takes the confidence top-16 eligible positions from that
same state (fewer near completion). Each serial, memory-bounded probe starts
from a fresh clone, reveals only its candidate's baseline token, and scores
`sum(H_baseline[j] - H_probe[j])` over all other masked positions. Entropy is
FP32, MASK-suppressed, in nats. Already revealed and noncandidate positions
have negative-infinity selection scores. Ties favor lower positions. The top
4/8 scores win **simultaneously**, using baseline tokens. Candidate positions,
scores, and baseline confidence are saved. This policy differs from Phase 1's
sequential greedy influence: there is no recomputation or permanent update
between winners, and individual counterfactual effects need not add up for a
joint reveal. Top-16 filtering is explicitly an approximation.

## Forward-pass budget (not a runtime guarantee)

| Setting | Each non-influence strategy / question | Batched influence / question | Total for 60 questions, all strategies |
| --- | ---: | ---: | ---: |
| Parallel-4 | 64 | 1,064 (64 baseline + 1,000 probes) | 79,200 |
| Parallel-8 | 32 | 536 (32 baseline + 504 probes) | 39,840 |
| Both | | | **119,040** |

There are no extra model pilot/warm-up passes. The first evaluation can include
initialization overhead. CUDA-event timings include all probes and host launch/
selection gaps, exclude loading/tokenization/final serialization, and accompany
synchronized wall timings. Phase 1 timings suggest an order of hours, but GPU
load, caches, software, and entropy/probe overhead prevent a guaranteed estimate.
The template requests the same queue/GPU/CPU/RAM as Phase 1 and **24 hours**,
with incremental results and resume if more time is needed.

## Offline caches and account

The public Phase 1 script contains a funding placeholder, not the actual account.
Use **your actual Phase 1 account** via `qsub -A`; it cannot be recovered from the
saved public artifacts. Use your existing Phase 1 Conda environment and HF cache.
No packages are installed by the job. Existing Phase 1 dependencies include
PyArrow (via datasets), PyYAML, Torch, Transformers, and huggingface-hub.

Model preflight uses the pinned HF snapshot with `local_files_only=True` and
checks configuration, tokenizer, remote-code files, and weight shards.
Dataset preflight looks for the pinned hub Parquet or the datasets downloads
cache; any file used must match the exact SHA-256 above. Alternatively point
`GSM8K_TEST_PARQUET` to the **existing** Phase 1 cached Parquet. Arrow-only caches
without the raw pinned Parquet are not accepted because their version cannot be
verified through the required file hash. If either pinned cache is missing,
**stop**; do not switch revisions or enable downloads. File preflight never
instantiates the model or runs forward passes.

## Local CPU verification

Nine deterministic toy-data tests passed locally. To repeat them in a local
Python environment with Torch and PyYAML installed, from the repository:

```bash
python -m unittest -v test_phase2
python -m py_compile phase2_samplers.py phase2_data.py run_phase2.py test_phase2.py
bash -n run_phase2.pbs
```

No pretrained model is loaded by these tests. The saved 60-question selection
is required by the actual-selection test. Regenerating/validating the selection
also requires PyArrow and an existing pinned test Parquet:

```bash
python run_phase2.py --prepare-selection --dataset-parquet /path/to/pinned-test.parquet
```

This command refuses to replace a different saved selection. The prepared
selection SHA-256 is
`5eb7681ca76a0ed61d7fe55296e9cf9a8aacfb8f802971abd9b5894e9992a45c`.
All 26 tracked Phase 1/1.5 checkpoint files were compared against Git HEAD and
are byte-identical. No new experimental responses or metrics were generated.

## Manual deployment commands

These are instructions for you; no SSH/SCP/qsub commands have been executed.
Replace the three placeholders with your existing Phase 1 hostname, login,
and repository directory. Keep the shell variables in the same session.

On your Mac, from this repository:

```bash
export HPC_TARGET='YOUR_LOGIN@YOUR_HPC_HOST'
export HPC_REPO='/path/to/existing/dllm-unmasking-dynamics'
# Upload only new Phase 2 files; existing Phase 1 files remain untouched.
scp phase2_samplers.py phase2_data.py run_phase2.py phase2_config.yaml run_phase2.pbs test_phase2.py PHASE2.md "$HPC_TARGET:$HPC_REPO/"
ssh "$HPC_TARGET" "mkdir -p '$HPC_REPO/outputs/phase2'"
scp outputs/phase2/phase1_exclusions.json outputs/phase2/selection.json "$HPC_TARGET:$HPC_REPO/outputs/phase2/"
ssh "$HPC_TARGET"
```

Inside your HPC SSH session:

```bash
cd /path/to/existing/dllm-unmasking-dynamics
export PHASE1_ACCOUNT='YOUR_ACTUAL_PHASE1_PROJECT_CODE'
export CONDA_BASE='/path/to/existing/conda'
export LLADA_ENV='llada-phase1'
export HF_HOME='/path/to/existing/phase1/huggingface-cache'
# If Phase 1 used a module, use that exact site module:
# export CONDA_MODULE=YOUR_PHASE1_CONDA_MODULE
# module load "$CONDA_MODULE"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$LLADA_ENV"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Optional if the automatic pinned dataset cache lookup fails:
# export GSM8K_TEST_PARQUET='/path/to/existing/cached/test-parquet'
DATA_ARGS=()
if [[ -n "${GSM8K_TEST_PARQUET:-}" ]]; then DATA_ARGS+=(--dataset-parquet "$GSM8K_TEST_PARQUET"); fi
python -m unittest -v test_phase2
python run_phase2.py --config phase2_config.yaml --check-cache "${DATA_ARGS[@]}"
# Only after CPU tests and file-only offline preflight pass:
PBS_VARS='HF_HOME,CONDA_BASE,LLADA_ENV,HF_HUB_OFFLINE,HF_DATASETS_OFFLINE,TRANSFORMERS_OFFLINE'
for var in GSM8K_TEST_PARQUET CONDA_MODULE HF_DATASETS_CACHE HF_HUB_CACHE; do
    if printenv "$var" >/dev/null; then PBS_VARS="$PBS_VARS,$var"; fi
done
JOB_ID=$(qsub -A "$PHASE1_ACCOUNT" -v "$PBS_VARS" run_phase2.pbs)
printf '%s\n' "$JOB_ID"
qstat -f "$JOB_ID"
# Submit once; optional exported cache/module overrides were included above.
```

Monitor after the job starts:

```bash
qstat -f "$JOB_ID"
tail -f "outputs/phase2/logs/job-${JOB_ID}.log"
# You can stop tail with Ctrl-C; it does not cancel the job.
cat outputs/phase2/run/summary.csv
```

Restart an interrupted job with the same qsub command after the previous job
has stopped. A lock prevents concurrent writers; manifest/source/selection
mismatches stop resume. Changing source/config requires a fresh `output_dir`.
No completed evaluation is rerun; only a started but uncompleted evaluation
restarts. Model initialization on a restart is required but is not an evaluation.

Back on your Mac, retrieve results without overwriting Phase 1:

```bash
mkdir -p outputs/phase2/retrieved
scp -r "$HPC_TARGET:$HPC_REPO/outputs/phase2/run" "$HPC_TARGET:$HPC_REPO/outputs/phase2/logs" outputs/phase2/retrieved/
```

## Output schema and integrity

- `selection.json`, `phase1_exclusions.json`: pre-inference selection/exclusions.
- `run/manifest.json`: complete config, source/selection/exclusion hashes,
  revisions, seed, and policy; compared exactly on resume.
- `run/journal.jsonl`: authoritative write-ahead log. A `started` record is
  fsynced before each evaluation; a full `completed` result is fsynced after.
  Includes response/raw text, token IDs, confidences, all trajectories,
  counterfactual candidate distributions, actual passes and measured timings.
- `run/results.csv`: per-evaluation results, atomically rebuilt from completions.
- `run/summary.csv`: accuracy and denominators by strategy × parallelism ×
  stratum, plus all-stratum rows; expected counts flag partial runs.
- `run/environments.jsonl` and `logs/job-JOBID.log`: private runtime/progress logs.

Only an incomplete final journal append is truncated on restart. Corrupt full
records, unexpected keys, missing starts, or duplicate completions fail. Full
results persist before derived summaries, so interrupted summary writing is
repaired from the journal. To rebuild summaries locally without model access:

```bash
python run_phase2.py --config phase2_config.yaml --summarize
```

CPU tests cover both full schedules/all policies, baseline-token simultaneous
commitment, independent probe states, self-entropy exclusion, confidence top-16
filtering, revealed-position exclusion, influence ties, RNG isolation, fixed
pass counts, deterministic strata, actual no-overlap selection, and journal
crash/resume/duplicate handling. No pretrained model outputs are simulated or
reported as experimental results. Phase 2 results will exist only after your
manual HPC run. No Phase 2 files are committed or pushed.
