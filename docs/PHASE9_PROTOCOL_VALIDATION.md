# Phase 9 prospective implementation validation

See the [locked protocol, transfer and submission guide](../experiments/phase9/README.md),
[configuration](../configs/phase9_qwen_concise_256.yaml),
[runner](../experiments/phase9/run.py), [PBS template](../experiments/phase9/run.pbs),
[post-run analysis](../src/dllm_unmasking/analyze_phase9.py), and
[CPU tests](../tests/test_phase9.py).

## Status and verified references

Implementation is prepared; no intervention inference, model download, HPC access,
job submission, commit, push or history rewrite occurred. There are no genuine
Phase 9 results or public Phase 9 research report. Synthetic unit-test results
were confined to temporary ignored directories and removed afterwards.

Read-only validation of existing artifacts confirms Qwen-256 28/60, Qwen-512 50/60,
and LLaDA LTR P4 37/60. Qwen references are validated from their raw journals and
historical Phase 8 file receipts. The LLaDA reference uses the immutable Phase 7
per-question table, whose hash and independent raw-journal validation were recorded
in Phase 8. Exact selected question IDs/order, golds, revision, prompt controls and
reference identities are checked before execution. No baseline is rerun.

## Reuse and isolation

The Phase 8 runner gains an opt-in protocol argument, defaulting to its original
behavior. The native generation function, model/tokenizer loading sequence,
CUDA event instrumentation, numerical extractor and shared journal utilities
remain unchanged. The Phase 9 wrapper supplies stricter config/output/cache,
reference, environment, condition and prompt-placement validators. Packaging adds
the Phase 9 subpackage. No existing Phase 1–8.5 numerical result, configuration,
raw journal, manifest, model pin, tokenizer asset, or manual audit is changed.
Original source hashes remain historical; interrupted older runs must use their
original checkout instead of bypassing manifest checks.

Run/configuration/condition/instruction hashes prevent accepting a baseline
journal as an intervention. The sampler remains cached greedy AR. There are no
extra scoring forwards, alternative instructions, few-shot examples or new
stopping rules. At most 15,360 model forwards are required for 60 × 256 tokens,
including prefill; actual EOS stopping can reduce that budget.

## Checks completed

- **91 CPU tests passed**, including eleven new Phase 9 tests. The fixture-only
  suite passes 84 tests with seven explicit existing private-artifact skips.
- Tests cover exact instruction and controls, template role/gold exclusion,
  shared generation-function identity, condition/config hashes, GPU/software
  matching, output isolation, valid resume, duplicate/foreign/corrupt journal
  rejection and owning-runner partial-tail repair.
- Synthetic complete 60-question fixtures exercise post-run analysis, paired
  recovery/regression statistics, native answer offsets, figures and blinded
  packet creation. Incomplete runs and public-output destinations are rejected.
- Blinded packets reproduce with fixed independent ordering streams, contain
  no condition/row-ID metadata, preserve A/B allocation in a separate hidden key,
  and refuse to overwrite ratings. Human-review counts remain zero by design.
- All **58 Python files** passed syntax/import checks; both new CLIs work from
  outside the checkout through the established editable installation.
- All **eight PBS templates** passed Bash syntax checks. PBS acceptance, actual
  cache availability and GPU environment remain untested by design.
- Local Markdown file links/code fences, public-file privacy and PNG metadata,
  ignored tokenizer safeguards, and both whitespace diff checks pass.
- A pre-change SHA256 inventory verifies **204 protected files unchanged**;
  only README navigation, setup packaging and the explicitly reviewed shared
  runner adapter are modified among pre-existing tracked files. Earlier raw
  artifacts, reports and the full Phase 8.5 manual audit remain intact.

The local legacy pip cannot reinstall a modern editable project without newer
build-tool handling; its failed optional install attempt rolled back. No runtime
package upgrade is required: the existing editable parent package already resolves
Phase 9, as verified outside the checkout. Transfer/submission instructions reuse
that established working installation rather than reinstall or upgrade the pinned
Qwen environment. Fresh clones should use the repository installation guide with
current editable-install tooling.

## Post-run review and remaining limits

The analyzer requires 60 completed records, the exact pinned local tokenizer
files, original references, configuration/manifest consistency and the original
recorded GPU/software environment. It writes only ignored results/phase9 staging,
never a public report directory, and refuses to overwrite an analysis/review packet.
Its standalone report and paired tables are generated only after real results
exist. No fabricated prospective accuracy or reasoning-quality claim is supplied.

Before public release, review scientific conclusions, data integrity and privacy;
copy only approved derived metrics/plots/report. Replace private audit hyperlinks
with instructions to generate the local packet. Never publish raw responses,
review forms, keys, environment logs, model/tokenizer assets or journals.

Remaining uncertainty includes real cache/scheduler availability, future accuracy
and timing outcomes, effective reviewer blinding despite stylistic cues, and
reasoning quality until independent evidence is collected. The original balanced
60-question subset and separate timing jobs limit generalization. No architecture
comparison or causal claim about diffusion is part of this intervention.
