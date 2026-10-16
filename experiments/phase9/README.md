# Phase 9 — Concise Reasoning Intervention

**Completed run; the preselected prospective protocol below is preserved.**
See the [canonical Phase 9 report](../../reports/phase9/PHASE9_REPORT.md):
concise-256 reaches 22/60 versus original-256 28/60, with paired uncertainty
that includes zero. Blinded human reasoning-quality review remains outstanding.
This is one preselected within-Qwen prompt intervention motivated by
[Phase 8.5](../../reports/phase8_5/PHASE8_5_REPORT.md). It is not an architecture
comparison, prompt search, or new decoding algorithm. Historical reference journals
verify Qwen-256 **28/60**, Qwen-512 **50/60** and LLaDA LTR P4 **37/60** (the LLaDA
reference is the unchanged, raw-validated Phase 7 table).

## Locked intervention and controls

Original suffix (preceded by the same single newline after the question):

> `Solve this problem step by step. End with '#### <final numerical answer>'.`

The only new instruction, preselected before this experiment:

> `Solve this problem using only the essential calculation steps. Be concise, avoid unnecessary explanations, and end with '#### <final numerical answer>'.`

No wording variants, few-shot examples, roles, or other prompt changes are allowed.
Each rendered intervention prompt must equal the historical native rendered prompt
with exactly this instruction substituted in its original position. Prompt token
counts may change because instruction length changes.

[Configuration](../../configs/phase9_qwen_concise_256.yaml): exact same 60 GSM8K
questions and order, gold solutions and strata; model/tokenizer
`Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`; native chat template, BF16, SDPA,
batch one, greedy `do_sample=False`, KV cache, at most 256 new tokens, native EOS,
inference seed 42, unchanged numerical evaluator. Dataset revision:
`740312add88f781978c0658806c59bc2815b9866`; selection seed 20261014.

The Phase 8 runner has an opt-in protocol adapter. Its default path and
`generate_one`, loading sequence, CUDA event scope, tokenizer, evaluator and journal
utilities are reused. The Phase 9 adapter restricts output paths and instruction,
checks immutable historical reference hashes and cached native metadata, and
requires the original recorded GPU/software environment (including the NVIDIA
RTX PRO 6000 Blackwell Server Edition class). No access/download/repinning mode
is permitted through Phase 9. The original inference manifests/source hashes
remain historical, not silently updated. Resuming older interrupted Phase 8 runs
still requires their original source checkout; completed baselines are read-only.

Independent outputs: `outputs/phase9/qwen_concise_256/run/` contains immutable
manifest, configuration hash, run hash, condition/instruction hash, environment,
fsynced write-ahead journal, per-evaluation results, stratum summaries and completion
status. A single-writer lock and strict run/config/code/input/environment hashes
prevent mixed runs. Original journals never count as Phase 9 completions.
The owning runner may repair an incomplete final journal append, validates all
complete records, resumes incomplete questions and skips completed ones. Duplicate
or corrupt complete records fail. Configuration/output changes cannot bypass the
historical input checks. Reuse the same command to resume.

## Local CPU checks (no model inference)

```bash
python scripts/check_cpu.py
python -m experiments.phase9.run --help
bash -n experiments/phase9/run.pbs
```

Use the established editable Phase 8 checkout; its existing experiments package
resolves the added Phase 9 subpackage without reinstallation. Fresh clones use
the repository editable-install instructions and a current PEP 660-capable pip.
No change to the working Qwen GPU dependencies is required.

The full artifact integration suite additionally requires the documented local
Phase 2/3 artifacts and `--with-artifacts`. Cache preflight below is read-only;
missing weights or tokenizer bytes fail clearly with no network download.

## Mac-to-HPC transfer

Set `HPC_HOST` and `PROJECT_ROOT` privately to the existing SSH destination and
absolute remote checkout. Run from this local repository root. Transfer source
and protocol only, without model files, original raw results or environments:

```bash
rsync -avR --exclude='__pycache__/' --exclude='*.pyc' setup.py experiments/phase8/run.py experiments/phase9/ configs/phase9_qwen_concise_256.yaml src/dllm_unmasking/ tests/test_phase9.py "$HPC_HOST:$PROJECT_ROOT/"
rsync -avR --ignore-existing reports/phase8/provenance.json "$HPC_HOST:$PROJECT_ROOT/"
```

The established HPC checkout must retain its exact original
`outputs/phase2/selection.json`, `outputs/phase8/qwen/model_pin.json`, both original
Qwen run directories and public Phase 7 tables/provenance. The config names these
existing baseline directories. Do not repin, reformat or overwrite those files.
Checks reject any mismatch. The source transfer does not transfer `outputs/`,
`results/`, tokenizer assets, caches or weights.

## HPC environment, cache preflight and manual submission

Use the same established Qwen Conda environment and scratch cache variables. No
new package upgrade or model download is needed. `PBS_PROJECT` stays private:

```bash
cd "$PROJECT_ROOT"
module load anaconda2025/2025-gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
export PHASE9_ENV="${PHASE8_ENV:-dllm-ar-baseline}"
conda activate "$PHASE9_ENV"
export PHASE9_CONFIG=configs/phase9_qwen_concise_256.yaml
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${HF_HOME:?Use the original scratch cache setting}"
: "${PHASE8_HUB_CACHE:?Use the original AR Hub cache}"
: "${PBS_PROJECT:?Set your allocation privately}"
python -m experiments.phase9.run --config "$PHASE9_CONFIG" --check-cache
qsub -P "$PBS_PROJECT" -v HF_HOME,PHASE8_HUB_CACHE,PHASE9_ENV,PHASE9_CONFIG experiments/phase9/run.pbs
```

These are instructions for manual execution, not actions taken by implementation.
The PBS request reuses queue `gpu_bwp6000`, one GPU, four CPUs, 64 GB RAM, and the
original four-hour allowance. Runtime is not guaranteed. Maximum native model
forwards: **60 × 256 = 15,360**, including one prefill per question; EOS can reduce
actual forwards. There are no confidence probes or extra scoring passes.

## Retrieve results and analyze after inference

Back on the Mac, copy into the new Phase 9 directory only:

```bash
mkdir -p outputs/phase9/qwen_concise_256/run
rsync -av "$HPC_HOST:$PROJECT_ROOT/outputs/phase9/qwen_concise_256/run/" outputs/phase9/qwen_concise_256/run/
python -m dllm_unmasking.analyze_phase9 --input-dir outputs/phase9/qwen_concise_256/run --tokenizer-dir local_tokenizers/qwen --output-dir results/phase9/analysis
```

Analysis requires all 60 valid completed records and the pinned local Qwen tokenizer
JSON/config used in Phase 8.5; it does not load weights. It fails rather than
approximating token offsets. Default analysis writes only ignored private staging,
and refuses to overwrite an existing analysis or human ratings; pick a fresh
`results/phase9/` subdirectory for reanalysis.

Outputs include a standalone `PHASE9_REPORT.md`, per-question measurements,
recoveries/regressions, per-stratum and overall summaries, paired correctness and
secondary contrasts, three figures, and an analysis manifest. No observations are
fabricated in advance. Primary comparison: original versus concise Qwen-256,
paired numerical correctness, 10,000 within-stratum paired bootstrap samples
(20 per stratum), NumPy PCG64 seed 42 reset per contrast, linear percentile 95%
intervals, exact two-sided McNemar. Secondary EOS, markers, lengths and latency
intervals are exploratory and pointwise. Latencies include mean/median/p90.

First complete explicit answer uses the unchanged parser on saved native prefixes,
excluding last-number fallback and partial digits within a full-output number.
Positions are zero-based/exclusive ends. Full-output completeness uses hindsight,
so it is not a stopping rule. Native token counts, whitespace words and Unicode
characters remain distinct. Gold labels support post-generation evaluation only.
GPU interval is the original synchronized prefill+native cached decode scope;
generation wall time excludes loading/tokenization/CPU decode. Different jobs and
lack of timing repetitions limit latency interpretation. LLaDA decode-only timings
are not treated as harmonized with AR. No architecture claim is tested.

## Blinded reasoning-quality audit and publication

Share only `results/phase9/analysis/review_packet/`. The comparison contains anonymous
case IDs and A/B outputs, the question and reference final numerical answer, and
blank independent ratings. Condition labels, original row IDs, automated correctness,
timing/length summaries and keys are excluded. Question ordering uses Python
`random.Random(20261015)`; A/B allocation uses an independent seed-20261016 stream.
Keep `unblinding/key.json` separate until ratings are locked. Reviewers may infer
conditions from style; concealment is not proof of effective blinding.

Criteria explicitly cover arithmetic correctness, essential intermediate-step
completeness, coherence, unsupported jumps and final-answer correctness. Concise
valid derivations need not spell out trivial steps. Wrong extracted numbers are
not automatically arithmetic errors; shorter text is not automatically worse.
Use evidence and unclear options, preferably two reviewers and documented
reconciliation. No reasoning-quality preservation claim is permitted without
actual review evidence; the generated report says quality is unassessed.

Only after scientific and privacy review should selected derived report/CSV/PNG
files be copied into `reports/phase9/`. Never copy raw journals, full-response audit
packets, ratings, keys, runtime logs, environments, or tokenizer assets there.
Future prompt variants must have separate exploratory labels and outputs; this
implementation rejects changed wording. No commit, push or history rewrite is
part of this phase's preparation.


## Completed-run finalization

The immutable downloaded concise journal is now finalized in the
[Phase 9 report](../../reports/phase9/PHASE9_REPORT.md). Reproduce validation and
derived publication files locally, without inference or changing old artifacts:

```bash
python -m dllm_unmasking.finalize_phase9 --phase2-results "$PHASE2_RESULTS"
```

The full fixed-seed blinded packet is generated privately at
`results/phase9/finalization/review_packet/`; keep its sibling `unblinding/` hidden.
The finalizer preserves existing review forms. A separate
`--output-dir results/phase9/recheck_public --private-dir results/phase9/recheck_private`
regenerates independent derived copies. Error-mechanism notes in the report are
unblinded assistant-assisted evidence audits, not completed human quality ratings.
See [final validation](../../docs/PHASE9_FINAL_VALIDATION.md) for checks and limitations.
