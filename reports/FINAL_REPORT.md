# Unmasking Dynamics and Reasoning Budgets — v1.0 Technical Report

[Project overview](../README.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Release validation](../docs/V1_VALIDATION.md)

## Abstract

This exploratory GSM8K study began by asking how a masked diffusion language
model should choose which tokens to reveal. It developed into a matched benchmark
of position selection, parallelism, spacing and revision in LLaDA-8B-Instruct,
then an autoregressive comparison with Qwen2.5-7B-Instruct and a prospective concise
prompt intervention. The initial 150-question cohort and subsequent 60-question
cohort are distinct. On the latter, LLaDA left-to-right (LTR) achieves 43/60 at
parallelism one (P1), 37/60 at P4 and 5/60 at P8; confidence achieves 26/60, 27/60
and 26/60 respectively. Qwen reaches 28/60 under a 256-token cap and 50/60 under
512. A fixed concise instruction produces much shorter, faster outputs but scores
22/60, with 11 recoveries and 17 regressions relative to original Qwen-256.
Formatting and extraction effects materially complicate interpretation.

These are numerical-answer observations on repeatedly inspected questions, not
validated measures of reasoning quality or evidence that architecture causes the
cross-model differences. Recorded timing boundaries differ between model families.
The v1.0 release consolidates completed evidence; no further inference is performed.

## 1. Research progression and experiment populations

The original questions were whether reveal order affects correctness, whether
counterfactual influence offers useful selection signals, and how those relationships
vary with question properties. Parallel decoding then exposed an LTR-specific
failure pattern, motivating spacing and revision controls. The sequential matched
benchmark supplied an accuracy–latency reference. Qwen comparisons shifted attention
to actual answer availability within output budgets. Phase 8.5 analyzed existing
continuations, motivating Phase 9's single preselected concise instruction.

| Phase | Population and inference count | Purpose and canonical reference |
| --- | --- | --- |
| 1 | 150 questions × five policies = 750 evaluations | [Sequential unmasking methodology](../docs/PHASE1.md) |
| 1.5 | Same 750 outputs; no inference | [Complexity, trajectories, influence diagnosis](../research/phase15/report.md) |
| 2 | 60 new questions × five policies × P4/P8 = 600 | [Parallel benchmark](../experiments/phase2/README.md) |
| 2.5 | Existing Phase 2 outputs; no inference | [Offline diagnosis and manual-audit tooling](../src/dllm_unmasking/analyze_phase25.py), [audit preparation](../src/dllm_unmasking/prepare_ltr_audit.py) |
| 3 | Same 60 × spaced LTR P8 = 60 | [Spaced schedule](../experiments/phase3/README.md) |
| 4 | Same 60 × two spaced-LTR revision policies = 120 | [Remasking controls](../experiments/phase4/README.md) |
| 5 | Same 60 × two confidence revision policies = 120 | [Dynamic confidence plus revision](../experiments/phase5/README.md) |
| 6 | Same 60 × LTR/confidence P1 = 120 | [Matched sequential benchmark](../experiments/phase6/README.md) |
| 7 | Phases 2–6: 1,020 outputs, 17 configurations; no inference | [LLaDA synthesis](phase7/PHASE7_REPORT.md) |
| 8 | Same 60 × original Qwen-256/512 = 120 | [AR baseline and budget sensitivity](phase8/PHASE8_REPORT.md) |
| 8.5 | Existing LLaDA P4 and Qwen-256/512; no inference | [Reasoning-budget utilization](phase8_5/PHASE8_5_REPORT.md) |
| 9 | Same 60 × concise Qwen-256 = 60 | [Prospective prompt intervention](phase9/PHASE9_REPORT.md) |

The matched set excludes all 150 Phase 1 IDs and contains 20 questions in each gold
calculation-count stratum (0–2, 3–4, 5+). Counts of `<<...>>` annotations are
complexity **proxies**, not validated reasoning depth. The balanced set is not
representative of the full GSM8K difficulty distribution. Phase 1's accuracy cannot
be treated as a baseline for a paired cross-phase improvement or decline.
Llama remains an optional implementation, not an evaluated baseline.

## 2. Shared protocol, provenance and measurement

Dataset: `openai/gsm8k`, `main/test`, revision
`740312add88f781978c0658806c59bc2815b9866`. LLaDA revision:
`852152c60bdfed4632635bfcf3e83408ad3f65bf`. Qwen revision:
`a09a35458c702b33eeacc393d103063234e8bc28`. The matched selection seed is
20261014; inference seed is 42; position RNG uses 42 + question ID and revision
RNG uses 314159 + question ID. Phase-specific settings remain in their configs
and manifests; see the [selection index](../research/benchmark_selection.json).

LLaDA commits 256 positions in one block, BF16, deterministic MASK-suppressed
argmax, with no early EOS stopping. P1/P4/P8 mean 1/4/8 new positions per step,
requiring 256/64/32 steps. Simultaneous batches use predictions from a common
pre-commit state. Counterfactual influence adds scoring forward passes. Revision
re-predicts eligible committed tokens in the same pass as new tokens, at most once
per token; random and low-confidence controls share the revision budget.

Qwen uses its native tokenizer/chat template, greedy decoding, KV cache, BF16,
batch size one, and EOS-aware stopping. Original Qwen question wording and suffix
are shared across its two caps. Native templates differ across model families;
Phase 2 did not log rendered LLaDA prompts or prompt-token IDs, so identical full
templates cannot be established. Qwen tokenizer hashes are authenticated by saved
manifests. Local LLaDA assets decode every observed response exactly, but an
unrecorded historical tokenizer checksum cannot be reconstructed.

The unchanged evaluator compares extracted numerical answers as exact rationals.
It prioritizes the last `####` marker, followed by boxed/answer phrases and a logged
last-number fallback. Extraction failure counts as incorrect. A trailing empty
marker can reject an otherwise readable value. Correct numerical extraction does
not certify arithmetic, logical coherence or faithful reasoning. Evaluation labels
are never silently replaced by sensitivity diagnostics.

**Budget and timing boundaries:** a nominal cap of 256 native Qwen tokens matches
256 LLaDA positions only nominally. Tokenization, actual visible length, stopping,
compute and architecture differ. LLaDA times cover decode-only sampler intervals;
Qwen times cover prefill plus cached decode. CUDA-event intervals can include host
launch gaps, exclude model loading and queue time, and are neither pure kernel
execution time nor production throughput. Separate jobs and no repeated timing
trials limit precision. Cross-model dominance is not established until timing
boundaries and run conditions are harmonized.

## 3. Initial cohort: reveal order and influence

| Phase 1 policy | Correct / 150 | Accuracy |
| --- | ---: | ---: |
| Random | 83/150 | 55.3% |
| LTR | 121/150 | 80.7% |
| RTL | 58/150 | 38.7% |
| Confidence | 77/150 | 51.3% |
| Influence | 97/150 | 64.7% |

This is sequential 256-step generation on the original cohort. Influence scores
measure counterfactual reduction in predictive entropy at other masked positions;
the implemented confidence top-8 candidate pool is an approximation, not attention
or confidence substituted for influence. Multi-reveal influence uses sequential
greedy updates in Phase 1; Phase 2 batched influence instead scores a top-16 pool
from the common pre-commit state and commits together. These are different policies.

Phase 1.5 found a larger observed LTR–RTL gap in higher calculation-count groups,
and earlier leftward confidence/influence reveals among correct outputs. These are
exploratory associations, not causal effects of complexity or position. Matched
question comparisons, candidate distributions, denominators, uncertainty and cost
are retained in the [Phase 1.5 report](../research/phase15/report.md).

## 4. Matched diffusion benchmark: parallelism, spacing and revision

All cells below have the same 60 question IDs; untested cells are explicit.

| Policy | P1 | P4 | P8 |
| --- | ---: | ---: | ---: |
| LTR | 43/60 (71.7%) | 37/60 (61.7%) | 5/60 (8.3%) |
| Confidence | 26/60 (43.3%) | 27/60 (45.0%) | 26/60 (43.3%) |
| Random | Not evaluated | 26/60 (43.3%) | 21/60 (35.0%) |
| RTL | Not evaluated | 17/60 (28.3%) | 14/60 (23.3%) |
| Batched influence | Not evaluated | 23/60 (38.3%) | 19/60 (31.7%) |

LTR P4→P8 has 33 losses and one recovery, whereas confidence is comparatively
stable. RTL P8 also uses adjacent batches yet reaches 14/60, so adjacency alone
cannot explain the direction-specific pattern. Trajectory geometry measures
selection patterns, not actual semantic dependency strength. Source inspection and
CPU toy tests complement stored trajectories; logs alone cannot prove model-state
semantics.

Contiguous LTR P8 has mean longest repeated-word run 27.22 and final markers in
19/60, versus 2.75 and 51/60 at P4. This supports observable degeneration beyond
extraction alone. Spaced LTR P8 reaches 13/60, but spacing changes reveal timing and
directionality as well as adjacency. It does not isolate local dependency interference.

| P8 configuration | Correct / 60 | Passes per question |
| --- | ---: | ---: |
| Spaced LTR | 13/60 | 32 |
| Spaced LTR + random revision | 13/60 | 32 |
| Spaced LTR + low-confidence revision | 19/60 | 32 |
| Confidence + random revision | 27/60 | 32 |
| Confidence + low-confidence revision | 25/60 | 32 |

Revision uses 62 attempts per evaluation. Changed tokens and reduced repetition
demonstrate revisability, not successful correction of reasoning. The small paired
gains do not establish a reliable improvement over confidence P8. The
[Phase 7 paired reanalysis](phase7/paired_reanalysis.csv) supplies exact tests and
uncertainty rather than inferring significance from accuracy rankings.

## 5. Empirical LLaDA accuracy–latency Pareto frontier

Among the 17 observed LLaDA configurations, these four are nondominated by mean
recorded GPU decode time and numerical accuracy on this sample:

| Configuration | Correct / 60 | Mean decode seconds | Passes per question |
| --- | ---: | ---: | ---: |
| Confidence P8 | 26/60 | 1.2837 | 32 |
| Confidence P8 + random revision | 27/60 | 1.3178 | 32 |
| LTR P4 | 37/60 | 2.5726 | 64 |
| LTR P1 | 43/60 | 10.2506 | 256 |

![Observed LLaDA frontier, decode-only timing](phase7/accuracy_latency_pareto.png)

This is an empirical sample frontier, not a confidence-bounded operational optimum.
The revision point adds one correct answer and has paired exact p=1 versus its
baseline. Batched influence's extra scoring makes it expensive in this implementation,
not inherently inefficient across all possible implementations. Pass counts are
algorithmic budgets, not measured energy or throughput.

## 6. Autoregressive comparison and reasoning-budget utilization

| Configuration | Correct / 60 | Accuracy | Mean GPU seconds | Timing scope | Length-limited / 60 |
| --- | ---: | ---: | ---: | --- | ---: |
| LLaDA LTR P4 | 37/60 | 61.7% | 2.5726 | Decode only | Not an AR stopping rule |
| LLaDA LTR P1 | 43/60 | 71.7% | 10.2506 | Decode only | Not an AR stopping rule |
| Qwen original-256 | 28/60 | 46.7% | 5.3244 | Prefill + decode | 32/60 |
| Qwen original-512 | 50/60 | 83.3% | 6.2347 | Prefill + decode | 0/60 |
| Qwen concise-256 | 22/60 | 36.7% | 1.4791 | Prefill + decode | 0/60 |

![Original Qwen budgets and LLaDA empirical frontier; unequal timing scopes](phase8/accuracy_latency.png)

Qwen-512 improves on original-256 by 23 recoveries and one regression: +36.7
percentage points, paired 95% interval [25.0, 48.3], exact McNemar p=2.98023e-6.
This does not establish superiority over LLaDA as an architecture. For example,
Qwen-512 versus LTR P1 remains statistically uncertain on these questions;
all planned contrasts and multiplicity families are in the
[Phase 8 report](phase8/PHASE8_REPORT.md).

Phase 8.5 verifies exact saved-token and visible-character prefix relations for
all 60 Qwen-256/512 pairs. All 23 recoveries first acquire a complete explicit-cue
parser-correct answer beyond token 256: native prefix lengths 258–507, median 345.
Of 32 incorrect Qwen-256 outputs, 31 are cap-limited; this is an association, not
proof that each text is semantically unfinished. Among 16 cases where LLaDA P4 is
correct and Qwen-256 wrong, all Qwen outputs reach their cap and 14 recover at 512.

LLaDA P4 averages 686.5 visible characters and 133.5 words; Qwen-256 averages 757.0
and 139.2; Qwen-512 averages 884.5 and 164.0. Native visible token means are 240.4,
234.7 and 281.9 respectively, and must not be treated as common units of reasoning.
Censored short text is not necessarily a compact complete solution. Gold-assisted
answer-prefix analysis is hindsight, not an implementable stopping rule. Compactness
has not been established as a causal mediator of the model gap. The
[Phase 8.5 report](phase8_5/PHASE8_5_REPORT.md) links the full local manual audit
and documents offsets, unavailable provenance and overlapping error flags.

## 7. Prospective concise intervention: Phase 9

The single preselected instruction, in the same native-template user role as the
original suffix, was:

> Solve this problem using only the essential calculation steps. Be concise, avoid unnecessary explanations, and end with '#### &lt;final numerical answer&gt;'.

The original requested step-by-step solving and the same final-answer format.
No wording was tuned to the completed intervention results. Questions, revision,
cap, greedy settings, tokenizer, evaluator and timing methodology were held fixed
within Qwen; the sample and its baseline outcomes were already known.

| Paired outcome: original-256 → concise-256 | Questions / 60 |
| --- | ---: |
| Both correct | 11/60 |
| Both incorrect | 21/60 |
| Recovery | 11/60 |
| Regression | 17/60 |

Concise minus original accuracy is **−10.0 percentage points**, with paired
95% bootstrap interval **[−26.6667, +6.6667] percentage points** and exact two-sided
McNemar **p=0.34492845088243484**. The point estimate does not support the intended
accuracy improvement; uncertainty does not establish either population-level harm
or equivalent accuracy. Secondary comparisons against Qwen-512 and LLaDA P4 remain
separately labeled exploratory in the [Phase 9 tables](phase9/paired_correctness.csv).

Mean generated tokens decrease from 235.13 to 65.47, and mean recorded GPU time
from 5.3244 to 1.4791 seconds, about 72.2% reductions. All 60 concise outputs reach
EOS, yet 19 contain empty final-answer placeholders; 13 of those contain the gold
value earlier in the text. Two further cases expose first-number selection within
an equation after the marker. These facts do not revise the locked score of 22/60.
The separately labeled last-numerical-mention diagnostic is posthoc and can select
intermediate quantities; it is not a corrected benchmark evaluator.

![Concise intervention: output length distributions](phase9/output_lengths.png)

The text audit covers all 38 incorrect concise outputs with overlapping, verbatim
evidence tags. Some show explicit false arithmetic or mismatched constraints; bare
wrong answers do not reveal the hidden mechanism. Shorter text alone is not evidence
of worse reasoning. The fixed-seed blinded review packet contains 60 pairs and
120 blank rating rows; **no independent human reasoning-quality outcomes exist**.
Unblinded assistant-assisted textual assessment must not be substituted for that review.

Exploratory strata change from original to concise as follows: 12→9/20 (0–2),
13→6/20 (3–4), and 3→7/20 (5+). The hardest group's improvement coexists with
more original truncation exposure, but none of the three exact stratum tests is
significant at 0.05 even before Holm adjustment. This is not evidence that concision
helps difficult reasoning while harming easy reasoning as a general law.

## 8. Statistical interpretation and evidence boundaries

Questions are the pairing and resampling unit. Multiple evaluations of a question
are not independent experimental subjects. Phase 1.5 uses whole-question bootstrap
on its own 150 examples; the matched analyses preserve the three 20-question strata.
Current Phase 7 intervals are a labeled deterministic reanalysis: the original
bootstrap generator could not be recovered, and historical intervals are preserved
separately. Current Phase 7 uses 10,000 draws, PCG64 seed 42 and a shared stream in
contrast order. Phase 9 uses 10,000 within-stratum paired draws, seed 42 reset per
contrast, linear percentile endpoints. These are pointwise approximate intervals,
not simultaneous bounds or estimates of inference-seed variation.

Exact McNemar tests condition on discordant pairs. The Phase 7 ten-contrast family,
Phase 8 nine-contrast family, Phase 8.5 three-contrast family, and Phase 9 primary,
secondary and stratum families have distinct scopes; adjusted p-values must not be
interchanged. Percentile intervals and exact tests can disagree near a threshold.
Repeated inspection and exploratory decisions limit confirmatory interpretation
even after within-table adjustment. See each linked provenance file for exact rules.

**Confirmed observations:** matched artifacts and labels; schedule-dependent LLaDA
accuracy; severe LTR P8 repetition; the empirical within-LLaDA frontier; Qwen answer
availability after the 256 cap in recoveries; concise outputs' lower observed
length/time, official 22/60 score and format failures.

**Plausible mechanisms:** parallel local-dependency interference combined with
reveal direction/timing; model-specific verbosity consuming a finite output budget;
formatting changes contributing to concise-score differences. These explanations
are compatible with parts of the evidence, not isolated causal mechanisms.

**Unanswered questions:** faithful reasoning quality, why specific internal errors
occur, held-out generalization, robustness across seeds/models, whether standardized
latency preserves frontier membership, and whether format-only controls change the
interpretation. Neither diffusion architecture nor brevity is established as the
cause of the observed correctness differences.

## 9. Limitations and conclusions

This release studies one primary diffusion model and one primary AR baseline on
one task family. Only 60 reused, balanced GSM8K questions support the matched
benchmark; model-policy development and case inspection reused those outcomes.
The initial 150-question sample supplies separate exploratory evidence, not an
independent replication of later interventions. Tokenizer and native-template
differences prevent equal native-token caps from being equal reasoning or compute
budgets. Evaluation-format sensitivity, wrong-but-parseable numbers, empty markers,
posthoc diagnostics, incomplete human reasoning-quality review and separate-job
timing all limit interpretation. Private raw journals are not bundled in Git;
public derived data support statistics but not independent full-text audit without
restoring those artifacts. Historical allocation metadata remains in Git history,
not a credential; current-file cleanup does not erase that exposure.

The coherent result is that token-selection schedule, actual answer availability
and output format all matter to this benchmark. The concise instruction traded
substantial observed length and latency reductions for a lower accuracy point
estimate, with wide paired uncertainty and material format effects. There is no
validated general architecture advantage or proven reasoning-quality preservation.
Priority next steps are independent blinded review, prespecified parser/format
controls, harmonized timing and a held-out confirmation set. These are unanswered
research directions, not new experiments or implementations in v1.0.

## 10. Reproducibility and navigation

- [Reproduction and CPU tests](../docs/REPRODUCIBILITY.md), [repository map](../docs/PROJECT_MAP.md), [layout](../docs/STRUCTURE.md).
- [Phase 7 bootstrap provenance](phase7/bootstrap_provenance.json), [Phase 8 provenance](phase8/provenance.json), [Phase 8.5 provenance](phase8_5/provenance.json), [Phase 9 provenance](phase9/provenance.json).
- [Phase 8 validation](../docs/PHASE8_VALIDATION.md), [Phase 8.5 validation](../docs/PHASE8_5_VALIDATION.md), [Phase 9 final validation](../docs/PHASE9_FINAL_VALIDATION.md), [privacy/history audit](../docs/AUDIT.md).
- [v1.0 consolidation provenance](FINAL_PROVENANCE.json) records source/report hashes; [v1.0 validation](../docs/V1_VALIDATION.md) records checks and the complete files changed by consolidation.

The current Phase 7 regeneration script reproduces verified plot data and frontier
membership, but not the archived figure dimensions/style. Its numerical aggregate
tables agree within floating-point serialization tolerance; Phase 8/8.5/9 CSVs and
PNGs regenerate byte-for-byte. Original figures remain unchanged.

Historical reports retain their contemporary wording and future-work status. In
particular, Phase 8.5's proposed concise experiment was subsequently completed as
Phase 9; its historical report is not rewritten to imply foreknowledge. Full local
manual audits remain available at the paths documented by their individual reports.
No scientific implementation, raw artifact, historical report, GPU run or Git
history is changed by this consolidation.
