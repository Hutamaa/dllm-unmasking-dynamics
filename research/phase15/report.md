# Phase 1.5: observational analysis of unmasking dynamics

## Input and scope

Validated **750 unique evaluations: 150 matched questions × five strategies**, from `outputs/phase1`.
The journal, selected questions, and CSV agree on question identities, answers, correctness, and responses. Input SHA-256 hashes are stored in `../provenance.json`. No model calls, tokenizer downloads, or Phase 1 writes occur.

Every trajectory reveals all response positions exactly once. This is one sampled benchmark subset and one run per question/strategy, not independent decoding replicates. Recorded correctness is exact agreement of extracted numeric answers with the stored gold answer; it is not a human assessment of reasoning quality.
Recorded decoding settings: `{"gen_length": 256, "steps": 256, "block_length": 256, "mask_id": 126336, "influence_top_k": 8}`. Model revision: `852152c60bdfed4632635bfcf3e83408ad3f65bf`; dataset revision: `740312add88f781978c0658806c59bc2815b9866`.

## Overall observations

| Strategy | Correct / questions | Accuracy |
| --- | --- | --- |
| random | 83/150 | 55.3% |
| left_to_right | 121/150 | 80.7% |
| right_to_left | 58/150 | 38.7% |
| confidence | 77/150 | 51.3% |
| influence | 97/150 | 64.7% |

## Paired comparisons

Each pair uses the same question IDs. Exact two-sided McNemar tests condition on discordant pairs; Holm correction covers all ten overall pairwise tests. Bootstrap intervals resample whole questions (10,000 draws, seed 42) for the mean A−B correctness difference. They quantify question-sampling uncertainty under an IID-question approximation, not variation across model seeds.

| A − B | Matched n | A only / B only | Gap (95% paired bootstrap) | Exact p | Holm p |
| --- | --- | --- | --- | --- | --- |
| random − left_to_right | 150 | 9 / 47 | -25.3 pp [-34.0 pp, -16.7 pp] | 2.572e-07 | 1.8e-06 |
| random − right_to_left | 150 | 34 / 9 | 16.7 pp [8.7 pp, 24.7 pp] | 0.0001702 | 0.001021 |
| random − confidence | 150 | 27 / 21 | 4.0 pp [-4.7 pp, 13.3 pp] | 0.4709 | 0.4709 |
| random − influence | 150 | 15 / 29 | -9.3 pp [-18.0 pp, -1.3 pp] | 0.04877 | 0.09753 |
| left_to_right − right_to_left | 150 | 69 / 6 | 42.0 pp [32.7 pp, 51.3 pp] | 1.164e-14 | 1.164e-13 |
| left_to_right − confidence | 150 | 53 / 9 | 29.3 pp [20.0 pp, 38.7 pp] | 1.051e-08 | 9.457e-08 |
| left_to_right − influence | 150 | 36 / 12 | 16.0 pp [7.3 pp, 24.7 pp] | 0.0007173 | 0.003586 |
| right_to_left − confidence | 150 | 14 / 33 | -12.7 pp [-21.3 pp, -4.0 pp] | 0.007943 | 0.02383 |
| right_to_left − influence | 150 | 9 / 48 | -26.0 pp [-35.3 pp, -17.3 pp] | 1.52e-07 | 1.216e-06 |
| confidence − influence | 150 | 10 / 30 | -13.3 pp [-21.3 pp, -5.3 pp] | 0.002221 | 0.008886 |

## Do strategy gaps vary with complexity proxies?

Calculation count is the number of `<<...>>` annotations in the gold solution; question length is whitespace-separated word count; numerical mentions count digit-written numbers/fractions, not number words. These are **proxies, not validated reasoning depth**. Gold solutions are used only offline. Fixed bins: calculation count 0–2/3–4/5+; words 0–40/41–70/71+; numerical mentions 0–3/4–6/7+. Every bin, including empty bins, is retained in the CSVs.

### calculation_count

| Group | A − B | Matched n | A only / B only | Gap (95% paired bootstrap) |
| --- | --- | --- | --- | --- |
| 0–2 | left_to_right − right_to_left | 49 | 8 / 6 | 4.1 pp [-10.2 pp, 18.4 pp] |
| 0–2 | influence − random | 49 | 5 / 6 | -2.0 pp [-14.3 pp, 10.2 pp] |
| 0–2 | influence − confidence | 49 | 11 / 6 | 10.2 pp [-6.1 pp, 26.5 pp] |
| 3–4 | left_to_right − right_to_left | 79 | 47 / 0 | 59.5 pp [48.1 pp, 69.6 pp] |
| 3–4 | influence − random | 79 | 18 / 8 | 12.7 pp [0.0 pp, 25.3 pp] |
| 3–4 | influence − confidence | 79 | 15 / 3 | 15.2 pp [5.1 pp, 25.3 pp] |
| 5+ | left_to_right − right_to_left | 22 | 14 / 0 | 63.6 pp [45.5 pp, 81.8 pp] |
| 5+ | influence − random | 22 | 6 / 1 | 22.7 pp [0.0 pp, 45.5 pp] |
| 5+ | influence − confidence | 22 | 4 / 1 | 13.6 pp [-4.5 pp, 31.8 pp] |

### question_words

| Group | A − B | Matched n | A only / B only | Gap (95% paired bootstrap) |
| --- | --- | --- | --- | --- |
| 0–40 | left_to_right − right_to_left | 62 | 19 / 3 | 25.8 pp [12.9 pp, 38.7 pp] |
| 0–40 | influence − random | 62 | 9 / 6 | 4.8 pp [-6.5 pp, 17.7 pp] |
| 0–40 | influence − confidence | 62 | 14 / 6 | 12.9 pp [-1.6 pp, 25.8 pp] |
| 41–70 | left_to_right − right_to_left | 77 | 44 / 3 | 53.2 pp [40.3 pp, 66.2 pp] |
| 41–70 | influence − random | 77 | 17 / 7 | 13.0 pp [1.3 pp, 24.7 pp] |
| 41–70 | influence − confidence | 77 | 13 / 4 | 11.7 pp [1.3 pp, 22.1 pp] |
| 71+ | left_to_right − right_to_left | 11 | 6 / 0 | 54.5 pp [27.3 pp, 81.8 pp] |
| 71+ | influence − random | 11 | 3 / 2 | 9.1 pp [-27.3 pp, 45.5 pp] |
| 71+ | influence − confidence | 11 | 3 / 0 | 27.3 pp [0.0 pp, 54.5 pp] |

### numeric_mentions

| Group | A − B | Matched n | A only / B only | Gap (95% paired bootstrap) |
| --- | --- | --- | --- | --- |
| 0–3 | left_to_right − right_to_left | 88 | 34 / 5 | 33.0 pp [20.5 pp, 45.5 pp] |
| 0–3 | influence − random | 88 | 14 / 11 | 3.4 pp [-8.0 pp, 14.8 pp] |
| 0–3 | influence − confidence | 88 | 17 / 8 | 10.2 pp [-1.1 pp, 20.5 pp] |
| 4–6 | left_to_right − right_to_left | 59 | 33 / 1 | 54.2 pp [40.7 pp, 67.8 pp] |
| 4–6 | influence − random | 59 | 14 / 4 | 16.9 pp [3.4 pp, 30.5 pp] |
| 4–6 | influence − confidence | 59 | 13 / 2 | 18.6 pp [6.8 pp, 30.5 pp] |
| 7+ | left_to_right − right_to_left | 3 | 2 / 0 | 66.7 pp [0.0 pp, 100.0 pp] |
| 7+ | influence − random | 3 | 1 / 0 | 33.3 pp [0.0 pp, 100.0 pp] |
| 7+ | influence − confidence | 3 | 0 / 0 | 0.0 pp [0.0 pp, 0.0 pp] |

`complexity_accuracy.csv` includes per-strategy correct/total counts and Wilson intervals; `complexity_gaps.csv` emphasizes matched gaps. Different subgroup p-values are not evidence that gaps differ between groups. To directly describe gap–proxy association, `gap_trends.csv` computes Pearson correlation between the numeric proxy and per-question A−B correctness (−1, 0, +1), with two-sided question-level permutation tests and Holm correction across nine trends. These are exploratory associations and can reflect correlated question properties, truncation, extraction behavior, or other confounders.

| Proxy | A − B | n | Gap–proxy r | Permutation p | Holm p |
| --- | --- | --- | --- | --- | --- |
| calculation_count | left_to_right − right_to_left | 150 | 0.319 | 9.999e-05 | 0.0008999 |
| calculation_count | influence − random | 150 | 0.145 | 0.08209 | 0.5746 |
| calculation_count | influence − confidence | 150 | 0.063 | 0.4876 | 1 |
| question_words | left_to_right − right_to_left | 150 | 0.242 | 0.0039 | 0.0312 |
| question_words | influence − random | 150 | 0.070 | 0.3928 | 1 |
| question_words | influence − confidence | 150 | 0.024 | 0.7779 | 1 |
| numeric_mentions | left_to_right − right_to_left | 150 | 0.141 | 0.09009 | 0.5746 |
| numeric_mentions | influence − random | 150 | 0.085 | 0.299 | 1 |
| numeric_mentions | influence − confidence | 150 | 0.056 | 0.5065 | 1 |

### Main exploratory pattern

**left_to_right − right_to_left:** 0–2 calculations: 4.1 pp gap (n=49); 3–4 calculations: 59.5 pp gap (n=79); 5+ calculations: 63.6 pp gap (n=22). The descriptive gap–calculation-count correlation is 0.319, with Holm-adjusted permutation p=0.0008999.

**influence − random:** 0–2 calculations: -2.0 pp gap (n=49); 3–4 calculations: 12.7 pp gap (n=79); 5+ calculations: 22.7 pp gap (n=22). The descriptive gap–calculation-count correlation is 0.145, with Holm-adjusted permutation p=0.5746.

**influence − confidence:** 0–2 calculations: 10.2 pp gap (n=49); 3–4 calculations: 15.2 pp gap (n=79); 5+ calculations: 13.6 pp gap (n=22). The descriptive gap–calculation-count correlation is 0.063, with Holm-adjusted permutation p=1.

These are observed gap patterns, not an identified mechanism. In particular, weaker evidence of an influence gap trend is not proof that influence performance is invariant to complexity. The numeric-mention 7+ bin contains very few questions; its estimates should not guide firm conclusions.

## Reveal dynamics

Positions are normalized as position/(response token count−1), with 0=left and 1=right. The first f fraction of steps uses ceil(f × total steps), including every reveal in those steps. Each question contributes its own mean; questions, rather than token events, are the unit of aggregation. The curves use the mean newly revealed position at each step, not cumulative position.

| Strategy | Outcome | n | First 10% | First 25% | First 50% |
| --- | --- | --- | --- | --- | --- |
| random | correct | 83 | 0.508 | 0.501 | 0.500 |
| random | incorrect | 67 | 0.501 | 0.502 | 0.503 |
| left_to_right | correct | 121 | 0.049 | 0.124 | 0.249 |
| left_to_right | incorrect | 29 | 0.049 | 0.124 | 0.249 |
| right_to_left | correct | 58 | 0.951 | 0.876 | 0.751 |
| right_to_left | incorrect | 92 | 0.951 | 0.876 | 0.751 |
| confidence | correct | 77 | 0.347 | 0.423 | 0.454 |
| confidence | incorrect | 73 | 0.488 | 0.507 | 0.488 |
| influence | correct | 97 | 0.392 | 0.465 | 0.482 |
| influence | incorrect | 53 | 0.486 | 0.515 | 0.516 |

LTR and RTL are fixed-order reference controls; random is also shown. Correctness-stratified curves describe observed outputs. Conditioning on correctness and observing position differences does **not** establish that early reveal position caused correctness.

For confidence, first-10%-step mean reveal position is 0.347 among correct responses (n=77) and 0.488 among incorrect responses (n=73). This is a correctness-stratified association, not evidence of a causal benefit from moving reveals left.

For influence, first-10%-step mean reveal position is 0.392 among correct responses (n=97) and 0.486 among incorrect responses (n=53). This is a correctness-stratified association, not evidence of a causal benefit from moving reveals left.

## Observable output diagnostics

| Strategy | n | Extraction failures | Last-number fallback | Mean visible words | EoT marker present |
| --- | --- | --- | --- | --- | --- |
| random | 150 | 2 | 8 | 121.6 | 150/150 |
| left_to_right | 150 | 3 | 12 | 142.3 | 150/150 |
| right_to_left | 150 | 0 | 5 | 113.9 | 147/150 |
| confidence | 150 | 2 | 8 | 135.4 | 150/150 |
| influence | 150 | 1 | 11 | 118.7 | 150/150 |

Visible response lengths use stored EOS-truncated, special-token-skipped responses. Generated token count includes the whole fixed budget. Raw special-marker **character offsets** are observed directly; they are not token offsets. Confirmed token offsets are reported only for explicit `--special-token-id NAME=ID` mappings. No tokenizer is loaded and no token identity is guessed. EoT/end-of-text marker presence alone is not evidence that a response is correct or complete.

Extraction failures and fallback use are observable. A parsed but wrong number does not, by itself, identify arithmetic errors, flawed reasoning, truncation, or formatting failure. Full visible and raw responses plus gold solutions appear in the private `representative_responses.md` (not included in this public checkpoint) for human review. The two lowest question IDs are selected per category; these examples are illustrative, not frequency estimates.

| Disagreement / diagnostic category | Questions | Full examples |
| --- | --- | --- |
| left_to_right correct / right_to_left incorrect | 69 | 2 |
| right_to_left correct / left_to_right incorrect | 6 | 2 |
| influence correct / random incorrect | 29 | 2 |
| random correct / influence incorrect | 15 | 2 |
| influence correct / confidence incorrect | 30 | 2 |
| confidence correct / influence incorrect | 10 | 2 |
| Any extraction failure | 8 | 2 |
| All five incorrect | 14 | 2 |

Only 8/314 incorrect evaluations (8/750 total) have extraction failures. Most recorded failures therefore contain a parsed answer that disagrees with gold; this observation does not distinguish reasoning mistakes from other output or benchmark-label problems.

## Influence diagnostics

Only stored `greedy_rounds` are analyzed. Scores are entropy reduction in nats across other masked positions, evaluated inside the recorded confidence top-K pool; they are not estimates of exhaustive global influence. Selected scores are checked against stored candidate maxima. Candidate distributions include min/median/max/std, negative fraction, and winner–runner-up margin (blank for singleton pools). All relationship-to-accuracy summaries first average within question; 38,400 token decisions are not treated as 38,400 independent accuracy observations.

| Question-level metric | Outcome | n | Mean | Descriptive r with correctness |
| --- | --- | --- | --- | --- |
| mean_selected_score_nats | correct | 97 | 3.6211 | -0.175 |
| mean_selected_score_nats | incorrect | 53 | 3.7019 | -0.175 |
| early10_score_nats | correct | 97 | 7.7222 | -0.225 |
| early10_score_nats | incorrect | 53 | 8.8525 | -0.225 |
| early10_normalized_position | correct | 97 | 0.3925 | -0.136 |
| early10_normalized_position | incorrect | 53 | 0.4856 | -0.136 |
| early10_selected_confidence | correct | 97 | 0.9467 | 0.202 |
| early10_selected_confidence | incorrect | 53 | 0.9326 | 0.202 |
| mean_selected_confidence | correct | 97 | 0.9671 | 0.395 |
| mean_selected_confidence | incorrect | 53 | 0.9548 | 0.395 |
| mean_normalized_position | correct | 97 | 0.5000 | NA |
| mean_normalized_position | incorrect | 53 | 0.5000 | NA |
| mean_candidate_std_nats | correct | 97 | 4.3682 | -0.015 |
| mean_candidate_std_nats | incorrect | 53 | 4.3983 | -0.015 |
| mean_candidate_negative_fraction | correct | 97 | 0.7754 | 0.153 |
| mean_candidate_negative_fraction | incorrect | 53 | 0.7639 | 0.153 |
| mean_winner_margin_nats | correct | 97 | 3.6556 | 0.011 |
| mean_winner_margin_nats | incorrect | 53 | 3.6419 | 0.011 |

Raw influence magnitudes depend on how many other positions remain masked. The first-10% score provides a common early window but does not remove all selection/confounding effects. Negative scores mean increased predictive uncertainty elsewhere, not necessarily harmful reveals. Candidate confidence values are not stored for unselected candidates, so their confidence–influence relationship cannot be recovered. The all-reveals mean normalized position is exactly 0.5 by construction for every complete trajectory; its missing correlation is structural, not a finding about influence. Early-window positions carry ordering information. Whole-response means include special tokens and post-termination positions, so selected confidence is not a calibrated probability of answer correctness.

## Limitations and testable Phase 2 hypotheses

This is an exploratory, single-run analysis of 150 benchmark questions and correlated proxies. Subgroup bootstrap intervals are pointwise and can be unstable or degenerate in small/zero-discordance groups; Wilson accuracy intervals likewise do not represent seed variability. The bin analyses are not multiplicity-adjusted confirmatory tests. Overall paired tests and the nine trend tests use separate Holm families. No association here establishes that complexity or reveal position causes an accuracy difference. Accuracy uses the existing extraction and gold labels; reasoning-error taxonomies require annotation.

The following are proposed tests, not conclusions established by this analysis:

1. **Order × calculation-count interaction:** use a fresh held-out, pre-stratified question set and test LTR−RTL paired gap differences across calculation-count groups under identical prompts/budgets. Repeat random decoding seeds and pre-specify the interaction statistic; observational complexity strata alone still do not identify a causal effect of complexity.
2. **Early-position intervention:** randomly assign early left-biased versus right-biased reveal constraints for confidence/influence while keeping per-step counts and later policy fixed. Test paired accuracy differences to separate an early-position intervention from the descriptive correctness-stratified curves.
3. **Influence candidate-pool ablation:** compare top-8 with larger K and exhaustive scoring on a preselected held-out subset, alongside confidence/random. Measure paired accuracy, score decisions, forward passes, and GPU time to test whether the observed influence policy depends on the confidence prefilter.

## Reproduction

`python -m dllm_unmasking.analyze_phase15 --input-dir outputs/phase1 --output-dir results/phase15`

Outputs are separate from Phase 1. `../provenance.json` records source-file hashes, input hashes, analysis settings, package versions, and token-ID mappings. Only the three PNGs are generated: complexity gaps, reveal dynamics, and influence diagnostics.

## Public checkpoint scope

This is a sanitized copy of the completed offline report. Aggregate summaries and three PNGs are included here. Per-question/token diagnostics, full responses, raw inputs, and local analysis metadata remain private and can be regenerated with the commands above. No original artifacts were changed.
