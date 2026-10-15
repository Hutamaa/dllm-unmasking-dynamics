# Unmasking Dynamics in Diffusion Language Models

[Project overview](../../README.md) · [Reproduction guide](../../docs/REPRODUCIBILITY.md) · [Public per-question data](unified_per_question.csv)

**Status:** Exploratory, single-model matched-question benchmark  
**Date:** 15 October 2026  
**Model:** `GSAI-ML/LLaDA-8B-Instruct` (revision `852152c60bdfed4632635bfcf3e83408ad3f65bf`)  
**Task:** GSM8K numerical-answer extraction, 60 stratified test questions reused across Phases 2–6  
**Decoding:** 256 generated positions in one block; bf16; Blackwell RTX PRO 6000; fixed prompts and evaluation framework.

## Abstract

We evaluated **17 decoding configurations × 60 identical questions = 1,020 generations**. Two striking patterns emerged: (1) left-to-right (LTR) decoding excels when positions are revealed sequentially or in batches of four, but collapses when eight *contiguous* positions are revealed per step; (2) confidence-based selection is comparatively stable across parallelism 1, 4 and 8. Confidence-guided revision can reduce repetitions in a fragile spaced-LTR decoder, but neither random nor low-confidence revision provides a convincing accuracy gain on top of confidence P8.

**Recommended current stopping point:** No further exploratory remasking on these 60 examples. Treat this as a completed within-model *pilot benchmark*, publish reproducible code and transparent findings, then optionally run a held-out confirmation set before asserting generalization or publishing as a research paper.

## Research questions

1. How does increasing the number of newly committed positions per model forward pass affect accuracy for different token-selection policies?
2. Can intra-decoding token revision improve accuracy or generation stability without adding forward passes?
3. Which configurations provide nondominated *observed* accuracy–latency trade-offs on this benchmark?

## Experimental design and data integrity

The Phase 2 question selection is stratified into 20 examples each by the number of annotated gold-solution calculations (`0–2`, `3–4`, `5+`). This number is **a proxy, not an independently validated reasoning-difficulty measure**. Phase 2 generated 600 outputs across five policies × P4/P8, Phase 3 generated 60 spaced-LTR P8 outputs, Phase 4 generated 120 spaced-LTR-plus-revision outputs, Phase 5 generated 120 confidence-plus-revision outputs, and Phase 6 generated 120 P1 outputs. All methods have 60 unique, matching question IDs; observed per-case forward counts are constant within configurations. Older Phase 1 results on a *different* sample are excluded from matched accuracy comparisons.

The journal/manifests record pinned model and dataset revisions, seeds, methods, and the exact decoding configurations. This audit verifies **output-level** completeness and matching, not every internal CUDA operation or hidden implementation detail. Revision uses at most one update per position, choosing up to two previously committed positions per step after the first (62 revision attempts per evaluation). These replace tokens within the normal 32-pass P8 schedule, not with extra model passes.

## Main matched accuracy table

| Policy | P1: 256 passes | P4: 64 passes | P8: 32 passes |
|---|---:|---:|---:|
| Left-to-right | **43/60 = 71.7%** | **37/60 = 61.7%** | 5/60 = 8.3% |
| Confidence | 26/60 = 43.3% | 27/60 = 45.0% | **26/60 = 43.3%** |
| Random | Not evaluated on these 60 | 26/60 = 43.3% | 21/60 = 35.0% |
| Right-to-left | Not evaluated on these 60 | 17/60 = 28.3% | 14/60 = 23.3% |
| Batched influence | Not evaluated on these 60 | 23/60 = 38.3% | 19/60 = 31.7% |

Additional matched P8 runs:

| P8 schedule and revision | Correct / 60 | Accuracy | Mean recorded GPU decode time / question |
|---|---:|---:|---:|
| LTR spaced, no revision | 13 | 21.7% | 1.288 s |
| LTR spaced + random revision | 13 | 21.7% | 1.314 s |
| LTR spaced + low-confidence revision | 19 | 31.7% | 1.309 s |
| Confidence, no revision | 26 | 43.3% | 1.284 s |
| Confidence + random revision | 27 | 45.0% | 1.318 s |
| Confidence + low-confidence revision | 25 | 41.7% | 1.314 s |

![Matched accuracy as a function of parallelism](accuracy_parallelism.png)

### Matched comparisons

Paired comparisons use the **same 60 questions** and exact two-sided McNemar tests. The table below is exploratory; tests across many pairs should not be treated as independent confirmatory claims. The historical table is preserved in [selected_paired_tests.csv](selected_paired_tests.csv). Current reproducible confidence intervals and adjusted p-values are in [paired_reanalysis.csv](paired_reanalysis.csv), with methods below.

| Comparison | Accuracy change (second minus first) | Newly correct / newly incorrect | McNemar exact p |
|---|---:|---:|---:|
| LTR P1 → confidence P1 | −28.3 pp | 3 / 20 | 0.000488 |
| LTR P4 → confidence P4 | −16.7 pp | 8 / 18 | 0.0755 |
| LTR P8 → confidence P8 | +35.0 pp | 22 / 1 | 0.00000572 |
| LTR P4 → LTR P8 | −53.3 pp | 1 / 33 | 0.00000000407 |
| Confidence P1 → confidence P8 | 0.0 pp | 7 / 7 | 1.00 |
| LTR contiguous P8 → LTR spaced P8 | +13.3 pp | 11 / 3 | 0.0574 |
| LTR spaced P8 → + low-conf revision | +10.0 pp | 11 / 5 | 0.2101 |
| Confidence P8 → + random revision | +1.7 pp | 6 / 5 | 1.00 |
| Confidence P8 → + low-conf revision | −1.7 pp | 7 / 8 | 1.00 |

The P1-versus-P8 **relative-policy crossover** is pronounced on the observed sample. At P1 LTR wins (71.7% vs 43.3%); at P8 confidence wins (43.3% vs 8.3%). This does not establish an exact crossover threshold between P4 and P8, nor universal behavior beyond this model and question set.

## Observed accuracy–latency frontier

![Observed accuracy-latency frontier, all methods](accuracy_latency_pareto.png)

Using **mean recorded `gpu_seconds` for decoding** (excluding loading and HPC queue wait), the nondominated empirical configurations among the 17 evaluated are:

| Configuration | Correct / 60 | Mean recorded GPU decode time/question | Initial forward passes/question |
|---|---:|---:|---:|
| Confidence P8 | 26 | 1.284 s | 32 |
| Confidence P8 + random revision | 27 | 1.318 s | 32 |
| LTR P4 | 37 | 2.573 s | 64 |
| LTR P1 | 43 | 10.251 s | 256 |

This is **an observed-sample frontier, not a robust statistically established frontier**. In particular, the 27/60 confidence-random-revision point differs from its 26/60 baseline by one question (paired p = 1). Small timing gaps (e.g. 1.284 vs 1.318 s) can be sensitive to run conditions and timer semantics. Paired accuracy uncertainty can change nondomination. For an operational recommendation, **confidence P8, LTR P4, and LTR P1** are the three clearest qualitatively distinct trade-off options; revision has not shown a compelling additional benefit over confidence P8.

The batched-influence implementation was particularly expensive: roughly 21.8–43.2 s/question for P8/P4 respectively, with 536 or 1064 passes/question (including selection overhead). These are implementation-specific costs, not proof that all influence-based decoding is inefficient.

**Timing caveat:** GPU stream timings and host wall timings were separately exposed in Phase 6. The cross-phase chart uses similarly labeled recorded `gpu_seconds`, but was assembled from jobs executed at different times; it is not a controlled latency microbenchmark. Forward pass count is an algorithmic budget proxy, not an exact measure of energy, throughput, or total deployment latency. Confidence-based revision also incurs CPU selection and bookkeeping.

## Generation degeneration and mechanism hypotheses

Contiguous LTR P8 shows severe degeneration: mean longest repeated-word run **27.22**, final-answer marker present in **19/60**, and parser fallback in **39/60**. LTR P1 has mean longest run **1.30**, markers in **55/60**, and fallback in **4/60**. LTR P4 is intermediate (mean longest run **2.75**, markers in **51/60**). The collapse is therefore more than an answer extraction artifact; outputs themselves frequently degrade.

Spacing eight predictions across nonadjacent positions increases LTR P8 accuracy from 5/60 to 13/60, and reduces repetition, but also changes individual positions' reveal times. It does **not cleanly isolate adjacency**. Low-confidence revision of spaced LTR increases accuracy to 19/60 and reduces average longest repeated-word run to **3.92**. Random revision stays at 13/60 and mean longest run **12.08**. Revision attempts: 3,720 per strategy; token changes: **1,149 (30.9%)** under low-confidence targeting, versus **410 (11.0%)** random. These facts establish revisability, not correctness of individual revised tokens.

On confidence P8, random revision reaches 27/60 and low-confidence revision 25/60 versus 26/60 baseline. The absence of a consistent accuracy gain is compatible with a decoder that already chooses relatively stable positions, but this mechanism remains a hypothesis. Arithmetic error, wording/formats, answer extraction, and repetition should not be conflated.

![Accuracy by gold calculation-count strata](accuracy_by_stratum.png)

## Evaluator and case-study limitations

The unchanged numerical extraction metric is valuable for comparison, but sometimes disagrees with readable outputs. For example, the Phase 5 random-revision response to question **549** visibly contains `400 − 2 = 398` and `#### 398 ####`, yet the matched-case report lists extraction as `None`. Conversely, in question **211** from Phase 4, outputting the gold number does not imply valid arithmetic. Preserve original extraction metrics as the **primary, versioned result**, and separately perform a blind or rule-based parser-sensitivity audit on flagged outputs. Do not silently change labels.

The experimental question set was repeatedly inspected during design; reported comparisons are exploratory and potentially affected by analysis choices. Only one model, one task family, one 256-token budget and one primary GPU type were used. Some recovered outputs retain invalid intermediate math; an extracted correct number is **not evidence of faithful reasoning**. Speed values are per-sequence decode measurements, not an independently measured production throughput result.

## Conclusions and possible confirmation work

The observed benchmark supports substantial schedule sensitivity and a policy crossover on this fixed sample. It does not establish that spacing or revision reliably improves other models or held-out questions. Further heuristic variants evaluated on these same 60 questions risk overfitting analysis choices to the subset.

If the goal is a **portfolio/technical report**, the present evidence is sufficient as an honest, reproducible pilot. Present it with the caveats and plots above. If the goal becomes a research publication or general claim, the next highest-value experiment is a **preregistered held-out confirmation** of a short shortlist: LTR P1, LTR P4, LTR P8, confidence P8, and optionally spaced-LTR P8 + low-confidence revision. Predefine hypotheses and evaluator before looking at new results; choose sample size based on desired precision and HPC budget. An independent parser audit and matched standardized latency benchmark are cheaper priorities than new decoder heuristics.

## Statistical uncertainty

Every accuracy denominator is 60; each calculation-count stratum contains 20.
Contrasts pair correctness by question ID. Exact two-sided McNemar tests use the
discordant pairs, not an independence assumption across strategies. The ten
published contrasts are exploratory and their displayed p-values are unadjusted;
multiple comparisons and repeated inspection limit confirmatory interpretation.
Equal aggregate accuracies can conceal different recovered/regressed cases.

### Deterministic bootstrap reanalysis (v1)

The imported bootstrap generator, seed and resample count cannot be recovered.
Historical values remain unchanged in [selected_paired_tests.csv](selected_paired_tests.csv)
for comparison. **The intervals below are a new analysis**, not exact reproduction
of those historical intervals. Accuracy labels, discordant counts and exact
McNemar p-values have not changed.

Inputs: [unified_per_question.csv](unified_per_question.csv) and the fixed
[selection index](../../research/benchmark_selection.json); contrast definitions
follow the ten rows of the historical table. For each contrast, bootstrap paired
B-minus-A correctness differences within each of the three 20-question strata,
with replacement, preserving 20/20/20 counts. Use **10,000 resamples**, seed **42**,
NumPy **2.0.2**, `Generator(PCG64(42))`, and one shared RNG stream in contrast-row
order. Within each contrast visit strata `0-2`, `3-4`, `5+`, with questions sorted
by ID. Sum sampled differences, divide by 60, and convert to percentage points.
Intervals are the 2.5th/97.5th percentiles with NumPy `method='linear'`.
[bootstrap_provenance.json](bootstrap_provenance.json) records input and script
SHA256 hashes, versions and the complete ordering convention.

<!-- BEGIN REGENERATED PAIRED TABLE -->
| Comparison (A → B) | Change (pp) | New-analysis 95% CI (pp) | Exact p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| left_to_right P1 → confidence P1 | -28.3 | [-41.7, -15.0] | 0.000488281 | 0.00390625 |
| left_to_right P4 → confidence P4 | -16.7 | [-31.7, -1.7] | 0.0755187 | 0.453112 |
| left_to_right P8 → confidence P8 | +35.0 | [23.3, 46.7] | 5.72205e-06 | 5.14984e-05 |
| left_to_right P1 → left_to_right P4 | -10.0 | [-25.0, 3.4] | 0.263176 | 1 |
| left_to_right P4 → left_to_right P8 | -53.3 | [-66.7, -40.0] | 4.07454e-09 | 4.07454e-08 |
| confidence P1 → confidence P8 | +0.0 | [-11.7, 11.7] | 1 | 1 |
| left_to_right P8 → LTR spaced P8 | +13.3 | [1.7, 25.0] | 0.057373 | 0.401611 |
| LTR spaced P8 → LTR spaced + low-conf revision | +10.0 | [-3.3, 23.3] | 0.210114 | 1 |
| confidence P8 → Confidence + random revision | +1.7 | [-8.3, 11.7] | 1 | 1 |
| confidence P8 → Confidence + low-conf revision | -1.7 | [-13.3, 10.0] | 1 | 1 |
<!-- END REGENERATED PAIRED TABLE -->

Every row matches all 60 questions. These are pointwise approximate percentile
intervals, not simultaneous confidence bounds. Holm p-values adjust only the ten
listed exact tests; they do not cover all exploratory choices made during research.
Bootstrap intervals and exact McNemar tests are different procedures and need not
agree at a 0.05 cutoff: the LTR P4 versus confidence P4 interval excludes zero,
while its exact p-value is 0.0755 (Holm 0.453). This is not confirmatory evidence.
Some endpoints differ from the historical intervals because their undocumented
resampling procedure cannot be identified. The regenerated table is now the
canonical interval analysis; the historical table is retained without revision.
The stratified 20/20/20 design also differs from the full GSM8K distribution.

## Reproducibility notes

Model and tokenizer revision: `852152c60bdfed4632635bfcf3e83408ad3f65bf`.
GSM8K `main/test` revision: `740312add88f781978c0658806c59bc2815b9866`.
Selection seed: 20261014; inference seed: 42. Random position selection uses
`42 + question_id`; revision selection uses `314159 + question_id`. See the
[public selection index](../../research/benchmark_selection.json) for IDs,
calculation-count strata, Phase 1 exclusions and full-selection hash.

All runs use one 256-position block, BF16, deterministic argmax predictions,
MASK-suppressed output probabilities, and the pinned tokenizer chat template with
the same question suffix. P1/P4/P8 denote 1/4/8 new tokens per step, requiring
256/64/32 steps. EOS never terminates sampling early; evaluation decodes only the
prefix before first EOS and compares extracted answers as exact rationals. A
trailing second `####` makes the parser inspect the empty suffix, explaining the
question 549 example without changing its saved primary label.

The current cleanup independently matched all 1,020 public rows to the original
completed journals for correctness, extraction failures, strata, forward counts
and timing (within floating-point text roundoff). Source/CPU tests support the
specified implementations but cannot certify every operation of the historical
GPU execution. Degeneration and revision-change claims were checked against raw
records; raw texts and environment logs remain excluded from Git.

Regenerate public summaries and three figures into a separate directory:

```bash
python analyze_phase7.py --output-dir results/phase7_reproduced
```

This uses only the tracked per-question table and selection index. Figure styling
is newly generated, not a byte-for-byte recreation of the imported plots. Response
case studies require original raw journals and cannot be reconstructed from the
public numeric table. Consult the [reproduction guide](../../docs/REPRODUCIBILITY.md)
for dependencies, raw-input requirements, timing scopes and historical PBS setup.
No inference is involved in this command.

## Reproduction artifacts

[method_summary.csv](method_summary.csv), [unified_per_question.csv](unified_per_question.csv), [selected_paired_tests.csv](selected_paired_tests.csv) (historical), [paired_reanalysis.csv](paired_reanalysis.csv) (current intervals), [bootstrap_provenance.json](bootstrap_provenance.json), and [integrity_checks.csv](integrity_checks.csv) accompany this report. The full uploaded source ZIP is not reproduced in this package; preserved original experiments reside in the user's Phase 2–6 data archive. Figures reflect those exact files. No new GPU inference was performed for this report.
