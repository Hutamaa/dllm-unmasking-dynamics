# Phase 8.5 offline validation and navigation

See the [standalone report](../reports/phase8_5/PHASE8_5_REPORT.md),
[analysis entry point](../src/dllm_unmasking/analyze_phase8_5.py),
[reusable measurements](../src/dllm_unmasking/reasoning_budget.py), and
[CPU tests](../tests/test_phase8_5.py).

## Inputs and preservation

The analysis reads the immutable Phase 2 LTR P4 journal and the two Phase 8 Qwen
imports. Every imported Qwen file hash and both Phase 2 journal/manifest hashes
match Phase 8 provenance. The exact 60 IDs, question/gold text, evaluation labels,
model/dataset revisions, seed and protocol are rechecked. Both saved Qwen token-ID
and character-prefix relations hold on all 60 questions.

Original inference implementations, Phases 1–8 numerical results, journals and
manifests are unchanged. The Phase 8 report gains only a supplementary-analysis link. A pre-analysis SHA256 inventory verifies 162 earlier
files during the original analysis. The naming cleanup also updates README navigation
and the Phase 8 report link; `.gitignore` continues to exclude `local_tokenizers/`. The analysis checks original raw-input/report inventories
again on exit. Derived full-text cases stay under ignored `results/phase8_5/`.
No GPU inference, HPC access, downloads, commits or pushes occurred.

## Tokenizer verification

Qwen tokenizer JSON and tokenizer-config SHA256 exactly match both pinned run
manifests at revision `a09a35458c702b33eeacc393d103063234e8bc28`.
Optional supplied `vocab.json` and `merges.txt` match the pinned JSON contents.
A direct ByteLevel decoder needs no Transformers/tokenizers runtime or weights.
It exactly reconstructs all 180 saved raw/visible outputs, with strict UTF-8
alignment for every response. Unit tests cover multibyte characters split across
tokens, special/EOS handling, mismatched decoding, missing offsets, and checksums.
No estimated native offsets are substituted.

The historical LLaDA manifest did not record a tokenizer checksum. Its local
vocabulary reproduces all 60 saved raw/visible outputs, validating offsets for
observed IDs, but this does not independently authenticate the tokenizer asset's
revision or reconstruct its unlogged historical chat prompt. That limitation is
explicit in the report and provenance. All six local tokenizer files are ignored
and untracked; their necessary hashes are recorded without personal paths.

## Validation outcomes

- All **80 CPU tests passed** with the available baseline artifacts, including
  eleven Phase 8.5 tests. The fixture-only suite passes 73 tests and explicitly
  skips seven existing artifact integrations.
- All 52 public Python files passed syntax checks; package/experiment imports work.
- All seven PBS templates passed Bash syntax checks; no scheduler was contacted.
- Local Markdown targets, figures and code-fence balance passed checks.
- All 15 Phase 8.5 public artifacts and five private audit files regenerated
  byte-for-byte in separate directories under `results/`.
- Public text and PNG metadata passed bounded privacy/credential-pattern checks.
  Private tokenizer files and full responses remain excluded.
- Both staged and unstaged whitespace checks pass. No files were staged by this task.

## Interpretation safeguards

Native positions are output indices, not equal compute across models. Starts are
zero-based and ends exclusive. Nonempty lines, sentence-like units and numbered
steps are text proxies, not validated reasoning depth. A gold-valued intermediate
mention is not automatically an answer, parser error or successful derivation.
Complete-numeric-prefix diagnostics use full-output hindsight and are not a new
online stopping rule. Error flags overlap. Paired bootstrap intervals are
pointwise and exploratory; both-correct length comparisons select on outcomes.
Raw exact McNemar tests agree with Phase 8; the three-comparison Phase 8.5 Holm
family is labeled separately from Phase 8's nine-comparison family.

The report manually distinguishes a format-completion example (question 30),
a benchmark interpretation shift in the regression (711), and co-occurring marker
failure/premise omission (157). These bounded observations do not automatically
annotate all other cases. The private audit packet contains all 60 full three-way
responses and blank independent human-annotation fields.

There is no inference blocker because no new inference is required. Remaining
uncertainties are faithful reasoning quality, ambiguous intended answers, missing
historical LLaDA tokenizer/prompt provenance, and causal explanations of
compactness/architecture differences. The report proposes only one narrowly
scoped future within-Qwen concise-instruction comparison, without implementing it.

## Reproduction

```bash
python -m dllm_unmasking.analyze_phase8_5 --phase2 "$PHASE2_RESULTS" \
  --qwen-tokenizer local_tokenizers/qwen --llada-tokenizer local_tokenizers/llada
```

Keep `PHASE2_RESULTS` local and restore exact-hash raw imports plus the Phase 2
selection before running. The public report is generated under `reports/phase8_5/`;
full-text side-by-side cases are at `results/phase8_5/manual_audit/comparison.html`.
Use `--output-dir results/phase8_5_recheck --case-dir results/phase8_5/recheck_manual`
for an independent regeneration. The full CPU command is documented in
[repository structure](STRUCTURE.md); private inputs are required only for the
existing artifact integration tests and raw analysis execution.


## Reclassification provenance

The previous offline Phase 9 designation is now Phase 8.5, a supplementary
analysis of Phase 8. Phase 9 is reserved for a future Qwen-256 concise-reasoning
intervention; that intervention has not been implemented or run.

[Renaming provenance](../reports/phase8_5/renaming_provenance.json) retains the
complete original provenance object and its original hash, source hash, artifact
hashes, private audit hashes, movement map and conclusion-section checksum.
Historical names inside this receipt are intentional provenance, not stale commands.
All twelve numerical CSV/PNG artifacts remain byte-identical. Only report/HTML
labels, paths, documentation links and current analysis-source hashes changed.
The full [local manual audit](../results/phase8_5/manual_audit/comparison.html)
remains available with all 60 questions, original responses, candidate inventories,
and annotation forms. No audit data were deleted. Both private audit copies and
the previous independent regeneration were relocated. Current provenance explicitly
links to the original naming/provenance receipt rather than silently replacing it.
