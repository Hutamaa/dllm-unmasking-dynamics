# Phase 1: Unmasking strategies in LLaDA

Compare random, left-to-right, right-to-left, highest-confidence, and influence-based reveal orders
on a fixed seeded sample of 150 GSM8K test questions. Load LLaDA-8B-Instruct once;
no training, architecture changes, gold-answer prompting, or autoregressive
`generate()` calls.

The model-loading and logits interface follow the
[official LLaDA inference code](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py)
and its [recommended Transformers version](https://github.com/ML-GSAI/LLaDA#inference).
The model is pinned to
[852152c60bdfed4632635bfcf3e83408ad3f65bf](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/852152c60bdfed4632635bfcf3e83408ad3f65bf).
It requires `trust_remote_code=True` to load its published model implementation.

## Cluster setup

Use Python 3.11 on **Linux WildFly**, with sufficient Hugging Face cache storage
for the approximately 16 GB model and downloads. First inspect `module avail`
and `nvidia-smi`; the site's module names and driver have not been verified here.
If using a site Conda module, load its actual name before these commands; otherwise
use your installed Miniconda. Create the environment once:

```bash
conda create -n llada-phase1 python=3.11 -y
conda activate llada-phase1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

The CUDA 12.8 PyTorch wheel supports Blackwell, as documented in
[PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/). The GPU node needs a compatible
NVIDIA driver. Do not use an older CUDA wheel or copy a macOS environment.
No separate FlashAttention installation is needed. BF16 is the default; the run
fails if unsupported rather than silently changing precision.

If compute nodes lack internet, populate a shared cache on a node permitted to
download (downloads alone do not run inference):

```bash
export HF_HOME=/path/to/shared/huggingface-cache
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('GSAI-ML/LLaDA-8B-Instruct', revision='852152c60bdfed4632635bfcf3e83408ad3f65bf')
from datasets import load_dataset
load_dataset('openai/gsm8k', 'main', split='test')
PY
```

For a fully offline first run, replace dataset `revision: main` in `config.yaml`
with the downloaded dataset's commit SHA, and set `HF_HUB_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1`. Full commit SHAs avoid online revision-resolution calls.
Download the dataset with that same SHA (pass `revision=...` to `load_dataset`)
to ensure the matching dataset cache is populated before submitting offline.

## Submission

Edit `run.pbs`: replace `YOUR_PROJECT_CODE`; set `CONDA_BASE` to the actual Conda
installation and optionally `CONDA_MODULE` to an available site module. Its
allocation requests queue `gpu_bwp6000`, one GPU, four CPUs, 64 GB RAM, and 24 hours.
Confirm the site's GPU resource name (`ngpus`) and permitted walltime if PBS
rejects the request. No cluster job has been submitted by this implementation.

From this repository on WildFly:

```bash
qsub run.pbs
```

Set cache/module/environment overrides through PBS if needed:

```bash
qsub -v CONDA_BASE=/path/to/conda,LLADA_ENV=llada-phase1,HF_HOME=/path/to/cache run.pbs
```

Within a PBS GPU allocation the underlying command is
`python -u run_experiment.py --config config.yaml`. The runner requires
`PBS_JOBID` and a CUDA GPU to help prevent login-node inference. For an initial
GPU smoke run, copy the config, set `num_questions: 2` and a different
`output_dir`, then submit with `qsub -v EXPERIMENT_CONFIG=smoke.yaml run.pbs`.
Run the full 150-question experiment after reviewing its real outputs.

## Controlled decoding

Default: batch size one, 256 response tokens, 256 diffusion steps, one 256-token
block, argmax predictions (temperature zero), no classifier-free guidance and
no early stopping. All five strategies have the same prompt, update counts,
length, precision, and seed settings. One full block makes directionality global;
smaller blocks always advance left-to-right and compare orders *inside* blocks.
The schedule distributes each block's token count as evenly as possible across
its steps, assigning the remainder to the earliest steps, following the official
linear schedule. Every selected position is permanently revealed. MASK is
suppressed as a predicted output token for every strategy, guaranteeing that a
reveal cannot leave another mask. EOS remains eligible and never shortens the
sampler. Evaluation decodes only the prefix before the first EOS; full raw text
and all token IDs remain available.

Confidence is the chosen argmax token's softmax probability in FP32 after MASK
suppression. Confidence ties favor lower positions through a stable sort;
argmax token ties favor lower token IDs. Random position selection uses a private
Python RNG seeded by the experiment seed plus dataset question ID, independent
of all global RNGs. CUDA deterministic algorithms are required; unsupported
operations raise an error. Reproducibility is intended within the same recorded
hardware/software environment, not guaranteed across GPU types or versions.
Strategy execution order rotates between questions to reduce timing-order bias.

Timing uses CUDA events with synchronization after a full decode. `gpu_seconds`
is elapsed GPU-stream time for the complete sampler, including any launch gaps;
it is not a profiler's sum of kernel times. `wall_seconds` is synchronized sampler
wall time. Both exclude loading, prompt tokenization, final trajectory serialization,
text decoding, and file I/O. Influence candidate selection requires CPU/GPU
synchronization inside decoding; its measured runtime includes those costs and
all counterfactual passes. A full warm-up decode is excluded. The original four
strategies compute confidence and capture trajectories as before.

## Influence-based unmasking

`influence` uses real counterfactual model forwards, without attention proxies or
reference answers. At each selection it computes top-1 tokens, confidence, and
predictive entropy from FP32 logits, using the same MASK-suppressed vocabulary
as the other strategies. For the currently masked response positions M,
its score for candidate i is

```text
influence(i) = sum over j in M, j != i of [H_j(x) - H_j(x with i revealed)]
H_j = -sum over tokens v of p_j(v) * log(p_j(v))
```

Entropy uses natural logs (nats). Every probe starts from a fresh clone of the
same baseline state; only candidate i's current top-1 token changes. A positive
score means entropy fell elsewhere; negative scores are allowed and the largest
score still wins. Already revealed positions and candidates outside the pool
have selection scores of negative infinity. Influence ties favor lower positions.
Candidates must be masked in the active block; the entropy sum includes **all
other masked response positions**, including future blocks, and excludes the
prompt, revealed positions, and the candidate's own entropy reduction.

For multiple reveals in one diffusion step, selection uses a **sequential greedy
approximation**: score candidates, permanently reveal the winner, rerun the
baseline, and repeat until the scheduled number of positions has been revealed.
Predictions and the candidate pool are recomputed after each winner. This is
not joint optimization over sets of reveals. The logged `positions`, `token_ids`,
`confidences`, `influence_scores`, and `greedy_rounds` are aligned in selection
order within the diffusion step. Each greedy round records its candidate pool
and all counterfactual scores, enabling inspection of the selection decision.

`decoding.influence_top_k: 8` is the default **confidence-prefilter approximation**:
only the top eight eligible positions by confidence are evaluated, with lower
position breaking confidence ties. Confidence restricts the pool; the winner
is chosen by counterfactual entropy reduction. Set this option to `null` for
exhaustive candidate evaluation. Both the setting and the approximation label
are recorded in results and trajectories. The setting has no effect on the
original four strategies.

Forward-pass cost per response (excluding warm-up): confidence uses `steps`
passes. Influence uses one baseline pass per permanently revealed token plus
one probe per evaluated candidate. With one 256-token block and 256 steps:

- Confidence: **256** passes.
- Influence, top-8: **2,276** passes (2,020 additional, about **8.9x** total).
- Influence, exhaustive: **33,152** passes (32,896 additional, **129.5x** total).

For a block of length L, the probe count is `sum(min(K, m), m=1..L)`;
exhaustive evaluation uses `L*(L+1)/2`. Sum over blocks and add `gen_length`
baseline passes. If multiple positions are revealed per step, greedy baseline
recomputation also adds `gen_length - steps` passes versus confidence.
These are pass-count estimates, not measured runtime ratios: entropy processing
and candidate synchronization add overhead. All five strategies keep the same
per-step permanent reveal counts, final length, and diffusion step budget;
influence deliberately performs additional model calls inside that budget.
Review the GPU smoke run before deciding whether the PBS walltime suffices.

## Evaluation and outputs

Numerical answers use exact rational comparison, including commas, signed
decimals, scientific notation, and fractions. Extraction prioritizes `####`,
then LaTeX boxed answers, then explicit answer phrases. A final-number fallback
is recorded separately and can misidentify a number in an unfinished rationale;
inspect those rows during analysis. A missing numeric answer is `None`/blank,
with `extraction_failed=true`; it counts as incorrect, never as numeric zero.
A 256-token budget may truncate reasoning, which is a Phase 1 limitation shared
by all strategies. This zero-shot subset is not a reproduction of published
few-shot benchmark scores.

`outputs/phase1/` contains:

- `results.csv`: questions, gold/predicted numeric answers, extraction status,
  correctness, timings, forward-pass counts, approximation settings, confidence,
  and actual responses.
- `summary.csv`: per-strategy accuracy (failures included), total GPU time, mean timings, total/mean forward
  passes, approximation settings, extraction counts, and accuracy restricted to fully completed five-strategy
  question groups (`paired_accuracy`) for partial-run comparisons.
- `trajectories.jsonl`: response-relative positions, selected token IDs, and
  confidences at each zero-based diffusion step. Influence trajectories also
  include selected influence scores, evaluated candidates and their scores for
  each greedy round, approximation settings, and step forward-pass counts.
- `accuracy.png`, `runtime.png`: final comparison plots.
- `manifest.json`, `questions.json`, `environments.jsonl`: exact configuration,
  resolved model/dataset revisions, source hashes, selected IDs, and software/GPU
  information for each invocation.
- `records.jsonl`: authoritative journal, flushed and fsynced after each completed
  evaluation. Derived tables are atomically replaced after each evaluation.

Resubmit the same command to resume. An interrupted final journal append is
removed; unfinished evaluations restart. Concurrent writers to one output directory
are rejected. Changed configuration/code requires a new output directory;
use a new directory for this five-strategy version rather than mixing it with
previous four-strategy results. Old journals can still be analyzed, with their
forward counts inferred as one per logged step; five-way paired accuracy is
blank until all five strategies are present. An
interrupted question may leave different strategy counts; use paired accuracy
until the run completes. Rebuild tables/plots on a CPU without loading the model:

```bash
python analyze_results.py outputs/phase1
```

## Completed Phase 1 and Phase 1.5 checkpoint

Phase 1 completed on the GPU cluster: **150 GSM8K test questions × five strategies
= 750 evaluations**. Phase 1.5 was run locally on a CPU against those saved outputs.
Public aggregate summaries and lightweight plots are under `research/`; raw
responses, trajectories, datasets, caches, and environment logs are excluded.

| Strategy | Correct / 150 | Accuracy | Mean GPU seconds / response | Forwards / response |
| --- | --- | --- | --- | --- |
| Random | 83 | 55.3% | 10.092 | 256 |
| Left-to-right | 121 | 80.7% | 10.090 | 256 |
| Right-to-left | 58 | 38.7% | 10.091 | 256 |
| Confidence | 77 | 51.3% | 10.090 | 256 |
| Influence (top-8 approximation) | 97 | 64.7% | 91.327 | 2,276 |

The LTR−RTL paired accuracy gap is 4.1 percentage points for 0–2 annotated
calculations (n=49), 59.5 for 3–4 (n=79), and 63.6 for 5+ (n=22). Influence−random
gaps also widen across these groups, but the adjusted trend evidence is weak.
These are exploratory associations with complexity **proxies**, not evidence
that complexity or early reveal position causes accuracy differences. Only 8 of
314 incorrect evaluations are extraction failures; parsed wrong answers do not
automatically identify reasoning or arithmetic errors.

See [the Phase 1.5 report](research/phase15/report.md) for paired tests,
question-level bootstrap intervals, multiple-testing corrections, trajectory
and influence diagnostics, limitations, and three proposed Phase 2 hypotheses.
Phase 2 has not been implemented.

### Reproduce Phase 1 raw results

Use the Linux/PBS setup above. `run.pbs` is a reviewed public template with a
funding-code placeholder, optional site module, and generic Conda variables;
no account credentials or fixed personal paths are included. Copy it to a local
operational file if needed and fill in your environment settings before submission.
No submission is performed by preparing this checkpoint.

The completed run used model revision
`852152c60bdfed4632635bfcf3e83408ad3f65bf` and GSM8K dataset revision
`740312add88f781978c0658806c59bc2815b9866`, subset `main`, split `test`.
Copy `config.yaml` to an ignored location such as `outputs/reproduce.yaml`, set
`dataset.revision` to that dataset SHA, and use a fresh `output_dir`.
Keep seed **42**, 150 questions, BF16, 256 response tokens, 256 steps, one
256-token block, MASK ID 126336, and influence top-K **8**. All strategies use
the same prompt suffix, deterministic argmax predictions, and irreversible
reveals. Random position selection uses seed 42 plus the dataset question ID.

```bash
qsub -v EXPERIMENT_CONFIG=outputs/reproduce.yaml run.pbs
```

The runner generates `records.jsonl`, `questions.json`, `results.csv`,
`trajectories.jsonl`, summaries, plots, and a revision/configuration manifest.
Raw files are required for offline analysis but are not included in Git.
Within a compatible recorded environment, seed/settings reproduce the procedure;
bitwise agreement across GPU/software versions is not guaranteed.

### Reproduce Phase 1.5 on a CPU

Only NumPy and Matplotlib are needed; neither Torch nor model downloads are used:

```bash
python -m pip install numpy==2.0.2 matplotlib==3.9.4
python analyze_phase15.py --input-dir /path/to/private/phase1 --output-dir results/phase15 --seed 42 --bootstrap-repetitions 10000
```

Those versions match the completed offline analysis. The portable default input
is `outputs/phase1`. Required inputs are `records.jsonl`, `questions.json`, and
`results.csv`; include `manifest.json` to report revisions and settings. The
script verifies all 750 unique evaluations, checks schemas/CSV agreement, and
reads inputs without repairing or rewriting them. Outputs are written separately
to `results/phase15`, which is ignored by Git. EOS token offsets require an
explicit confirmed `--special-token-id eos=ID`; otherwise only observed raw
special-marker character offsets are reported.

[Public provenance](research/provenance.json) records revisions, settings,
seeds, original source/input hashes, and analysis package versions without local
paths or environment logs. Public report copies replace private paths; original
artifacts remain unchanged. New run manifests include source hashes, so changed
code/configuration requires a fresh output directory rather than resuming an
older run. The checkpoint's analysis default input-path change does not change
results when `--input-dir` is supplied explicitly.
