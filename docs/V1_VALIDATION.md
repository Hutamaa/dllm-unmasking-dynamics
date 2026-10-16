# v1.0 consolidation and publication validation

Scope: final technical-report consolidation, 16 October 2026. This prepares a local
v1.0 release candidate; no commit, tag, push or release publication is performed.
The [consolidated report](../reports/FINAL_REPORT.md) is the narrative entry point.
Individual phase reports remain historical records, including their then-future
proposals. No Phase 10 work, inference, model downloads or HPC access is included.

## Complete changed-file inventory for this task

- `README.md` — prominent final-report link, compact final results, and corrected
  limitations covering both model families and the 256/512 budgets.
- `reports/FINAL_REPORT.md` — new integrated technical report, with links to existing
  figures and phase reports; no new scientific measurements or relabeled scores.
- `reports/FINAL_PROVENANCE.json` — new source/figure/report hashes and evidence map.
- `docs/V1_VALIDATION.md` — this validation record and release limitations.

Earlier uncommitted Phase 9 implementation/report files and Phase 8 integration
changes predate this task and are preserved. This inventory describes the task's
actual byte changes, not the entire accumulated uncommitted Git working tree.

## Checks and reproduction

All 98 CPU tests pass with the restored private inputs; the fixture run passes
91 tests with seven explicit skips. Syntax/import checks cover 60 Python files;
eight PBS templates pass `bash -n`. Markdown file targets/code fences, public-candidate
privacy patterns, PNG metadata and staged/unstaged whitespace checks pass. A
pre-task inventory verifies 409 existing files unchanged, excluding only README. Fixture-only
tests explicitly skip seven integrations requiring private artifacts rather than
counting them as passed. No scheduler or GPU behavior is revalidated.

From the repository root in the installed CPU analysis environment:

```bash
export PHASE2_RESULTS="$RESULTS_ROOT/phase2"
export PHASE3_RESULTS="$RESULTS_ROOT/phase3"
OMP_NUM_THREADS=1 python scripts/check_cpu.py --with-artifacts
OMP_NUM_THREADS=1 python scripts/check_cpu.py
python -m dllm_unmasking.analyze_phase7 --output-dir results/v1_validation/phase7
python -m dllm_unmasking.finalize_phase8 --diffusion-root "$RESULTS_ROOT" \
  --output-dir results/v1_validation/phase8
python -m dllm_unmasking.analyze_phase8_5 --phase2 "$PHASE2_RESULTS" \
  --output-dir results/phase8_5/v1_validation_public \
  --case-dir results/phase8_5/v1_validation_audit
python -m dllm_unmasking.finalize_phase9 --phase2-results "$PHASE2_RESULTS" \
  --output-dir results/phase9/v1_validation_public \
  --private-dir results/phase9/v1_validation_private
git diff --check
git diff --cached --check
```

Use fresh ignored output paths for a subsequent independent reproduction. Phase 8, 8.5 and 9 public CSV/PNG artifacts regenerate byte-for-byte. Phase 7
numerical tables agree within 1e-12 (aggregate CSV differences are floating-point
serialization); current plots reproduce the verified data but differ from archived
figure dimensions/style and Matplotlib version. Exact historical Phase 7 rendering
is not reproduced. Original figures are preserved and reused, not overwritten. Provenance documents can differ by output path
and current analysis metadata; historical provenance is preserved, not overwritten.
Raw Phase 1 correctness is independently recounted from all 750 distinct records;
Phases 2–6 public aggregates are checked by the Phase 7 verifier. Phase 8/8.5/9
finalizers validate question/gold alignment, hashes, native decoding, protocols,
raw result labels, timings and statistical tables. The Phase 9 packet is independently
regenerated while original forms and unblinding materials remain unchanged.

A pre-task SHA-256 inventory covers existing public candidates and all local
reports, raw outputs, derived results and tokenizer assets. Every inventoried file
except the authorized README edit must retain identical bytes. New reproduction
copies live only under ignored results directories. Figures are reused from their
canonical reports and their bytes are recorded in consolidation provenance.

## Privacy and remaining risks

Current public candidates are checked for personal paths, allocation identifiers,
credential patterns, private keys and accidental runtime artifacts; local tokenizer
assets, journals, environments and logs remain ignored. The known historical
allocation identifier is searched literally across current files without printing
it. It remains infrastructure metadata in history, not an authentication credential.
History is not rewritten. Pattern scans cannot guarantee every possible secret is
absent; retained queue/module/GPU metadata and historical exposure remain subject
to institutional disclosure policy. See the [existing audit](AUDIT.md).

**Publication verdict: READY as an exploratory technical report, with explicit
limitations.** Human reasoning-quality review remains incomplete; independent
reasoning-quality preservation is not established. The repeatedly inspected 60-item
cohort, exploratory comparisons, unavailable historical bootstrap generator,
private raw-data distribution, tokenizer/template provenance limits and unequal
cross-model timing scopes prevent stronger generalization or architecture claims.
The historical bootstrap intervals are archived and current intervals are labeled
as a deterministic reanalysis. No such limitation is silently resolved by v1.0.
