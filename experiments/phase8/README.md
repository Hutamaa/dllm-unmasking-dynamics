# Phase 8: matched autoregressive baselines

**Default: Qwen2.5-7B-Instruct. Optional: Llama 3.1 8B Instruct, pending gated access.**
Both use the same unchanged 60-question GSM8K benchmark and AR generation procedure.
Completed Qwen-256 and Qwen-512 results are in the
[canonical Phase 8 report](../../reports/phase8/PHASE8_REPORT.md). Llama remains
unevaluated. Comparison with LLaDA is system-level: architecture, training, tokenizer and instruction tuning
all differ. Differences cannot be attributed causally to diffusion versus AR.

## Models, revisions and isolated outputs

| Selection | Configuration | Pinned model/tokenizer commit | Outputs / pin |
| --- | --- | --- | --- |
| `--model qwen` (default) | [phase8_qwen.yaml](../../configs/phase8_qwen.yaml) | `a09a35458c702b33eeacc393d103063234e8bc28` | `outputs/phase8/qwen/run`, `outputs/phase8/qwen/model_pin.json` |
| `--model llama` | [phase8_llama.yaml](../../configs/phase8_llama.yaml) | `0e9e39f249a16976918f6564b8830bc894c89659` | Original `outputs/phase8/run`, `outputs/phase8/model_pin.json` |

[phase8.yaml](../../configs/phase8.yaml) is an identical Qwen default config.
`--config PATH` selects a custom config. If supplied together, `--model` and
`--config` must identify the same model. Unsupported models are rejected.
Qwen's commit and ungated status were verified from the
[official metadata API](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct)
on 15 October 2026. The Llama commit was previously verified through its
[official metadata](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct).
No revision was guessed; the explicit pinning operation rechecks the exact commit
before execution. Pins cannot be overwritten silently.

Both models may share the separate AR Hub cache root, `PHASE8_HUB_CACHE`, because
Hub snapshots are isolated as `models--Qwen--Qwen2.5-7B-Instruct/snapshots/COMMIT`
and `models--meta-llama--Llama-3.1-8B-Instruct/snapshots/COMMIT`. The AR root must be
outside the checkout and separate from LLaDA's existing Hub cache. Model/revision,
policy, configuration and run hashes protect journals against mixed observations.
The Llama config retains its original paths and controls. Changed source hashes
still require a fresh directory for any pre-extension unfinished run; never edit
saved manifests to bypass validation. The combined analyzer reads the original
Llama result schema via its manifest, without modifying or relabeling that journal.

## Unchanged matched protocol and native formatting

Restore `outputs/phase2/selection.json` byte-for-byte, using the
[selection reproduction instructions](../../docs/REPRODUCIBILITY.md#pinned-data-selection-and-seeds).
Its original hash, public index and Phase 7 data hashes are checked. There are
60 questions, 20 per 0–2 / 3–4 / 5+ gold calculation-count stratum; question wording,
gold answers and IDs are unchanged. Gold is used only for post-generation exact
rational evaluation, never as model input. The suffix remains exactly:

```text
\nSolve this problem step by step. End with '#### <final numerical answer>'.
```

For both models: native `AutoTokenizer` chat template with a generation prompt,
then `add_special_tokens=False`; greedy `AutoModelForCausalLM.generate`,
`do_sample=False`, one beam, batch one, at most 256 new tokens, KV cache enabled,
BF16 and SDPA. Reset seed 42 per question. No temperature sampling, prompt tuning,
repetition penalty, ngram blocking, tools or Qwen3 thinking options are introduced.
The unchanged native-generate function is shared by both models.

Qwen2.5's native template uses ChatML-style role delimiters and inserts its default
helpful-assistant system message when no system message is supplied. Llama's
native template instead includes its role delimiters and system/date metadata;
LLaDA has its own template. These differences are retained and recorded, not
normalized away. No custom system instruction is added for either AR model.
The native generation configuration supplies EOS IDs; the native tokenizer supplies
PAD. No Qwen EOS/PAD IDs are hardcoded. Batch-one generation stops at EOS or the
length cap. Only IDs after the prompt are decoded; the final EOS is excluded from
visible text but retained in generated-token counts. Correct numerical extraction
is not proof of faithful mathematical reasoning.

The [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#requirements)
identifies Transformers support from 4.37.0. The existing separate AR environment's
**Transformers 4.45.2** supports both Qwen2 and Llama 3.1; no version change is needed
when that Phase 8 environment is already installed. Keep the historical diffusion
environment unchanged. Real model loading and GPU kernel compatibility have not
been tested locally; CPU fixtures establish API/selection/IO contracts only.

## Exact Qwen deployment and environment commands

Set `HPC_HOST` and `PROJECT_ROOT` privately, with a fresh destination using the
current repository layout. From the local repository root, transfer the reviewed
tracked tree, then include the uncommitted Phase 8 files explicitly:

```bash
git ls-files -z | rsync -av --from0 --files-from=- ./ "$HPC_HOST:$PROJECT_ROOT/"
rsync -avR --exclude='__pycache__/' --exclude='*.pyc' \
  configs/phase8.yaml configs/phase8_qwen.yaml configs/phase8_llama.yaml \
  experiments/phase8/ src/dllm_unmasking/ar_baseline.py \
  src/dllm_unmasking/ar_analysis.py src/dllm_unmasking/ar_compare.py \
  tests/test_phase8.py tests/test_phase8_models.py setup.py "$HPC_HOST:$PROJECT_ROOT/"
rsync -avR outputs/phase2/selection.json "$HPC_HOST:$PROJECT_ROOT/"
```

On HPC, from the deployed repository root (manual commands, not executed here):

```bash
module load anaconda2025/2025-gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
# Only if the separate AR environment does not already exist:
conda create -n dllm-ar-baseline --clone dllm-unmasking
conda activate dllm-ar-baseline
python -m pip install -r experiments/phase8/requirements.txt
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps --no-build-isolation -e .
export HF_HOME="$SCRATCH_DIR/huggingface"
export PHASE8_HUB_CACHE="$HF_HOME/phase8-ar/hub"
python -m unittest -v tests.test_phase8 tests.test_phase8_models
python -m experiments.phase8.run --model qwen --check-cache
```

The local CPU-only preflight is the same last command after package installation,
selection restoration and setting an external AR cache path. It reads files only,
including safetensors headers/payload bounds; it loads no tensors/tokenizer/model,
creates no cache files and never downloads. It reports missing/incomplete cache
without claiming that network access has been tested.

### Explicit metadata pin and optional download

**Qwen requires no gated-access approval or login.** Its Hub metadata, HEAD checks
and explicit download use `token=False`; existing Llama credentials are not needed
or sent for Qwen. On a network-permitted host, run these yourself:

```bash
# Metadata/HEAD requests only; no weights downloaded:
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
  python -m experiments.phase8.run --model qwen --check-access
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
  python -m experiments.phase8.run --model qwen --resolve-revision
```

If the exact snapshot is already complete, an offline alternative to the metadata
pinning route is:

```bash
python -m experiments.phase8.run --model qwen --pin-cached-revision a09a35458c702b33eeacc393d103063234e8bc28
```

Choose one pinning route. If missing, the following is an **explicit user-authorized
download step**; preflight, tests and PBS never invoke it automatically:

```bash
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
  python -m experiments.phase8.run --model qwen --download-model
python -m experiments.phase8.run --model qwen --check-cache
```

Only the pinned configuration/tokenizer/safetensors files are retrieved, including
Qwen's BPE merges/vocabulary assets. Header bounds and available file sizes are
checked; this is not a full cryptographic audit of tensor contents. No alternative
model is substituted after a failed check.

### Full Qwen submission, resume and retrieval

The template retains queue `gpu_bwp6000`, one GPU, four CPUs, 64 GB RAM, four hours,
Conda module `anaconda2025/2025-gpu`, and `set -eo pipefail`. Walltime is an allowance,
not a guarantee. Supply the authorized allocation privately, never in tracked code:

```bash
export PHASE8_ENV=dllm-ar-baseline
export PHASE8_MODEL=qwen
# Set PBS_PROJECT privately to the same authorized allocation used previously.
JOB_ID=$(qsub -P "$PBS_PROJECT" -v HF_HOME,PHASE8_HUB_CACHE,PHASE8_ENV,PHASE8_MODEL experiments/phase8/run.pbs)
qstat -f "$JOB_ID"
```

PBS sets offline flags and executes only the selected model. It optionally accepts
`PHASE8_CONFIG` for a custom config; pass that variable through `qsub -v` too.
No command here has been submitted. There are 60 generations, at most 15,360 new
IDs/native cached model forwards; AR tokens/cached calls are not equivalent to
LLaDA full-sequence diffusion passes.

Resubmit the same command after the previous writer stops. Fsynced start/completion
events, configuration/model/revision hashes, per-result validation and a single-writer
lock prevent duplicated or mixed evaluations. Unfinished questions restart; only
an incomplete final append is repaired. Complete corruption is rejected. Completion
requires all 60 distinct IDs. Use the original checkout to resume a pre-extension
inference run whose source hashes differ; existing data remains readable offline.

```bash
# CPU rebuild of the owning Qwen run's derived tables (exact cache/config required):
python -m experiments.phase8.run --model qwen --summarize
# Retrieve into a separate local location after completion:
mkdir -p outputs/phase8/retrieved/qwen
scp -r "$HPC_HOST:$PROJECT_ROOT/outputs/phase8/qwen/run" \
  "$HPC_HOST:$PROJECT_ROOT/outputs/phase8/qwen/model_pin.json" \
  "$HPC_HOST:$PROJECT_ROOT/outputs/phase8/logs" outputs/phase8/retrieved/qwen/
```

## Optional Llama route

Use `--model llama` or `--config configs/phase8_llama.yaml`. The generation,
cache-validation, timing and extraction implementation is the same as before.
Llama remains gated: request access through the official model page and authenticate
interactively with `huggingface-cli login` in your own terminal. Credentials stay
outside the repository; never put tokens into commands, config, logs or PBS files.
Unlike Qwen, the Llama operations may use the Hub library's implicit local login.

Run the same `--check-access`, `--resolve-revision`, optional `--download-model` and
`--check-cache` operations with `--model llama`. Its existing immutable cache can
instead be pinned with `--pin-cached-revision 0e9e39f249a16976918f6564b8830bc894c89659`.
For a user-approved job set `PHASE8_MODEL=llama` and use the same submission command.
Missing access is reported; it does not cause a fallback to Qwen or another model.

## Analysis and combined figure

Per-model outputs remain separate: `results.csv`, `summary.csv`, `journal.jsonl`,
`manifest.json`, `environment.json`, `paired_diffusion.csv`, `latency_comparison.csv`
and `completion.json`. Each model has its own four paired contrasts versus
confidence P8, confidence P8 + random revision, LTR P4, and LTR P1. No observations
are pooled or relabeled across models, revisions or run hashes. The same Phase 7
bootstrap method is retained: 10,000 within-stratum paired resamples, NumPy PCG64
seed 42, linear percentile quantiles, ascending question IDs, fixed stratum order,
and a shared stream across the four displayed contrasts. Exact McNemar and Holm
p-values remain distinct from pointwise bootstrap intervals. Partial denominators
are explicit. Numerical correctness does not establish reasoning validity.

Build a combined plot on a CPU without weights, a model pin, network access or
inference (the exact question selection is still required):

```bash
# Auto-detect existing configured Qwen/Llama runs. With neither, plot only the
# observed LLaDA frontier and label the absence of completed AR observations.
python -m dllm_unmasking.ar_compare --output-dir results/phase8_comparison
# Or select copies retrieved from HPC explicitly:
python -m dllm_unmasking.ar_compare \
  --run-dir outputs/phase8/retrieved/qwen/run \
  --output-dir results/phase8_comparison
# Repeat --run-dir with a Llama run to overlay both.
```

The analyzer checks manifest/run hashes, input identity, unchanged evaluator and
journal integrity, reading journals without repair. Original Llama records without
per-record model fields inherit identity only from their validated manifest. One
run per model/revision is permitted; replicates are never pooled. Archived source
hashes are retained in input manifests, not asserted to match current code.
Derived per-run summaries/paired contrasts go into separate `runs/` subdirectories.

`accuracy_latency.png` overlays **completed, matched 60-question** AR runs on the
original four LLaDA frontier points. `accuracy_latency.csv` marks descriptive
cross-scope empirical dominance; incomplete AR runs are listed but omitted from
the plot and ranking. The original LLaDA frontier stays visible even if an AR point
descriptively dominates it. Figure captions explicitly state timing-scope,
cross-job, model/tokenizer and uncertainty caveats. These point estimates do not
establish a controlled frontier or diffusion-versus-AR causal effect. No Phase 7
result files are updated. With no completed AR data, no AR measurements are invented.

## Timing scopes and verification limits

The shared generator's timing procedure is unchanged. CUDA synchronization/events
measure total native generation including prefill/scheduling (`gpu_prefill_decode_seconds`),
first-forward prefill, and summed subsequent forward intervals (`gpu_decode_forward_seconds`).
The latter excludes between-forward scheduling and is not the entire post-prefill
wall interval. Host generation wall time excludes loading/tokenization and later
CPU decoding, extraction and IO. No warm-up is run; cold-start effects remain.
Hooks add overhead but no per-token synchronization. Peak allocated memory includes
resident weights, not reserved memory. Tokens/second uses returned new IDs including
EOS; `response_tokens` excludes EOS. Real library/GPU versions are saved.

LLaDA's recorded sampler intervals and AR's prefill+generation intervals have
different scopes and were collected in different jobs. They are shown separately
with `directly_comparable=False`; cached AR forward counts are never equated to
diffusion forward passes. CPU tests cover native-template call contracts and
synthetic ChatML/special-token fixtures; they do not certify a downloaded tokenizer
or real CUDA model load. Those checks remain user-run after explicit download.

A separate [Qwen 512-token sensitivity experiment](QWEN_512.md) preserves the
completed matched-256 run and documents explicit PBS configuration selection.


## Completed Phase 8 analysis

The [canonical report](../../reports/phase8/PHASE8_REPORT.md) incorporates the
completed Qwen-256 and Qwen-512 runs. Raw journals remain isolated and private.
To verify public statistics without raw data:

```bash
python -m dllm_unmasking.finalize_phase8 --verify-public
```

To repeat raw verification, restore the original selection and Qwen model pin,
keep downloaded Qwen runs in their configured directories, and point to the
parent of the downloaded Phase 2, 5 and 6 directories:

```bash
python -m dllm_unmasking.finalize_phase8 --diffusion-root "$DIFFUSION_RESULTS_ROOT" --output-dir results/phase8_recheck
```

This imports byte-verified snapshots without modifying the original runs and
regenerates summaries, paired tests, figures and report in a separate directory.
Historical per-run bootstrap tables remain preserved; the canonical report uses
a versioned analysis with an independent seed-42 bootstrap stream per contrast.
