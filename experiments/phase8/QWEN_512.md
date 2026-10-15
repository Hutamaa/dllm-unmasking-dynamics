# Qwen 512-token budget sensitivity

This is a separate extended-budget analysis, not a replacement for the matched
256-token benchmark. `configs/phase8_qwen_512.yaml` materializes the Qwen config;
only `max_new_tokens` (512) and `output_dir` differ. It shares the exact pinned
model/tokenizer revision, Phase 2 selection, prompts, seed, evaluator, BF16,
native chat template, greedy KV-cached generation and timing implementation.
EOS stopping remains enabled: 512 is a maximum, not a required response length.
No new model download or revision resolution is needed.

All new artifacts, including `manifest.json`, `journal.jsonl`, `results.csv`,
`summary.csv` and environment metadata, belong to `outputs/phase8/qwen_512/run/`.
The existing `outputs/phase8/qwen/model_pin.json` is read during preflight and
execution. Do not repin it, copy the 256 journal into the new directory, or run
`--summarize` against the completed 256 directory. Resume with the identical
512 command; manifests/config hashes and explicit journal budgets must match.
Legacy journals without a budget field are accepted only as 256-token runs.
The default Qwen and optional Llama configurations retain their 256-token caps.

## Deployment and submission

From the local checkout, set `HPC_HOST` and `PROJECT_ROOT` to your existing
SSH destination and absolute remote checkout path. Transfer source only:

```bash
rsync -avR README.md experiments/phase8/README.md configs/phase8_qwen_512.yaml experiments/phase8/run.py experiments/phase8/run.pbs experiments/phase8/QWEN_512.md src/dllm_unmasking/ar_baseline.py src/dllm_unmasking/ar_analysis.py src/dllm_unmasking/ar_compare.py tests/test_phase8_budget.py "$HPC_HOST:$PROJECT_ROOT/"
```

On HPC, from that checkout, reuse the existing Phase 8 Conda environment and
scratch cache variables (`HF_HOME`, `PHASE8_HUB_CACHE`). Keep `PBS_PROJECT`
private. These commands do not download weights:

```bash
cd "$PROJECT_ROOT"
module load anaconda2025/2025-gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
export PHASE8_ENV="${PHASE8_ENV:-dllm-ar-baseline}"
conda activate "$PHASE8_ENV"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PHASE8_MODEL=qwen
export PHASE8_CONFIG=configs/phase8_qwen_512.yaml
: "${HF_HOME:?Reuse existing scratch HF_HOME}"
: "${PHASE8_HUB_CACHE:?Reuse existing AR cache}"
: "${PBS_PROJECT:?Set your allocation privately}"
python -m experiments.phase8.run --model qwen --config "$PHASE8_CONFIG" --check-cache
qsub -P "$PBS_PROJECT" -v HF_HOME,PHASE8_HUB_CACHE,PHASE8_ENV,PHASE8_MODEL,PHASE8_CONFIG experiments/phase8/run.pbs
```

No dependency changes are needed relative to the working Qwen-256 environment.
The existing four-hour PBS request is an allowance, not a runtime guarantee.
Maximum computation is 60 × 512 = 30,720 model forwards, including one prefill
per question; actual token counts and forwards depend on EOS termination.

## Offline comparison after completion

```bash
python -m dllm_unmasking.ar_compare --run-dir outputs/phase8/qwen/run --run-dir outputs/phase8/qwen_512/run --output-dir results/phase8_budget_comparison
```

This reads original journals without repairing or overwriting them. Tables and
labels identify model, pinned revision and token budget; each run gets a separate
summary. `comparison_kind` distinguishes `matched_256` from
`extended_budget_512`. Matched-256 dominance excludes extended-budget points;
a separate descriptive cross-budget flag is provided. The original LLaDA
frontier remains unchanged. Unequal generation caps, tokenizers, timing scopes
and jobs limit interpretation; numerical accuracy does not establish faithful
reasoning. Partial runs are not ranked. No 512 observations are inferred from
256 results.
