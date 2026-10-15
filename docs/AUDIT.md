# Repository cleanup, privacy and validation audit

> The sections below record the earlier publication pass at `fa8971a`. The structural-refactor addendum at the end describes the current checkout; the earlier file inventory and unstaged-state statements are historical.

Reviewed 15 October 2026. This is a local audit, not a certification that every
possible secret format has been detected. No GPU inference, model downloads,
HPC connections, submissions, commits, pushes or history rewriting were performed.

## Scope and safeguards

The working tree was clean before cleanup. A compressed backup of all originally
tracked files and a SHA256 inventory were saved outside the repository in the
local temporary directory before editing. Separate SHA256 inventories cover all
existing `outputs/`, `results/`, and downloaded Phase 1–6 artifacts. Historical
scientific Python modules, tests, configs, original result tables and original
figures are unchanged. Only public documentation and scheduler templates were
edited; new offline tooling is isolated.

The review covered all tracked paths, current public-file candidates, ignored
working-tree artifacts, dependency imports, PBS scripts, both reachable commits
and all 68 distinct historical file blobs. Author/committer metadata and commit
messages were reviewed as well. Binary figure files were excluded from text-regex
matching; their PNG metadata was inspected separately. The reachable-history scope
is local `git rev-list --all`; it does not include inaccessible remote forks,
GitHub caches, deleted unreachable objects or files outside this checkout.

Credential checks looked for Hugging Face/GitHub/AWS credential formats, private
key headers and suspicious literal credential assignments. Privacy checks covered
email addresses, user directories, scratch paths, IP literals, SSH/SCP destinations,
allocation identifiers and runtime environment metadata. Matching values were not
dumped into public audit output.

## Findings and classification

| Category | Findings / disposition |
| --- | --- |
| Actual secrets or credentials | None detected in reviewed current files, ignored working-tree text artifacts, reachable file history or commit messages. This is bounded pattern/manual review, not a guarantee. |
| Potentially sensitive infrastructure/personal metadata | A fixed HPC allocation identifier appears in the latest historical commit's Phase 3–6 PBS files and guides. Current files now use `YOUR_PROJECT_CODE`. The original value remains in history; it is an allocation identifier, not an authentication secret. Human review should decide whether institutional policy permits its historical presence. |
| Excluded private runtime artifacts | Ignored analysis metadata includes local absolute paths. These originals were preserved and remain excluded; do not publish ignored directories wholesale. |
| Harmless false positives | Generic scratch-directory examples and binary bytes resembling IP addresses. Examples now use `$SCRATCH_DIR`/`$HF_HOME`; binary matches were not treated as credentials. Both commits use GitHub no-reply author/committer addresses. SSH/SCP examples contain placeholders, not actual login endpoints. |
| Reproducibility metadata retained | Public model/dataset IDs and immutable revisions, hashes, question row IDs/strata, seeds, model dtype, GPU model, software versions, queue/module names, timing scopes and configuration settings. These are scientifically useful and are not credentials. |

History has **not** been rewritten or sanitized. Removing an identifier from the
current tree does not remove it from old commits. No actual credential was found
that requires rotation. If a later review finds one, stop publication work, revoke
or rotate it first, then coordinate any separate history remediation; deletion in
a later commit alone is insufficient.

No personal login name, private email, literal cluster hostname/IP, home-directory
path, or access token was found in public candidates. The public GitHub repository
identity/no-reply commit identity is intentionally public. Queue and module names
are retained as environment requirements; review those against local institutional
policy before publishing the cleanup.

## Scientific and documentation checks

- All 1,020 public per-question rows were matched to original completed Phase 2–6
  journals. Correctness, extraction status, strata and pass counts agree exactly;
  timing differences are limited to insignificant floating-point text serialization.
- Published method aggregates, empirical frontier membership, ten matched
  discordance tables and exact two-sided McNemar p-values reproduce from public data.
- Reported LTR repetition/marker/fallback counts and Phase 4 revision changes were
  independently recalculated from original raw journals. The question 549 parser
  case was replayed with the unchanged extractor: a trailing `####` leaves an empty
  suffix and correctly reproduces the saved extraction failure under that rule.
- Original Phase 7 bootstrap settings/generator are missing from the imported
  package. Historical intervals are retained unchanged in `selected_paired_tests.csv`.
  The report now uses the deterministic v1 reanalysis in `paired_reanalysis.csv`:
  10,000 within-stratum paired samples, NumPy PCG64 seed 42, linear percentile
  quantiles. Six of ten intervals have changed endpoints beyond roundoff. Input
  and script hashes are recorded in `bootstrap_provenance.json`; regenerated CSV
  and metadata match the public files byte-for-byte. A test locks report-table agreement.
- The exact 60 selected question IDs and exclusions are now included as a compact,
  text-free public index. Full selection reconstruction requires the original
  pinned dataset bytes; raw responses and gold texts are not included.

## Validation

- **36 CPU tests passed** with original baseline artifacts available: all 33
  historical tests plus three public-table/provenance validation tests.
- The fresh-clone fixture command exercises **29 passing tests and seven explicit
  artifact-dependent skips**. It does not count those skipped integrations as passed.
- Python syntax and imports checked for all root modules and the CPU test tool.
- All seven PBS templates pass `bash -n`; placeholders/site differences are
  documented. Shell syntax checks cannot establish scheduler acceptance.
- Markdown local file targets and figure paths checked, including the canonical
  report and historical guide; no GPU commands were executed for validation.
- Public-table regeneration ran successfully on CPU into a separate ignored
  directory. Historical CSVs/PNGs and original artifact hashes remain unchanged.
- A separate public-files-only copy (without ignored inputs) passed the fixture
  command. Selection reconstruction from the cached pinned Parquet produced the
  identical original selection SHA256. The installed CPU environment passes
  `pip check`.
- `git diff --check` and final public-candidate privacy review completed. Nothing
  was staged or deleted; no commit/push was requested by this cleanup task.

## Changed-file inventory

- Rewritten `README.md`; preserved old README as `docs/PHASE1.md` with a historical
  status banner and corrected relative links.
- New `docs/REPRODUCIBILITY.md`, `docs/PROJECT_MAP.md`, and this audit.
- Polished `reports/phase7/PHASE7_REPORT.md`: abstract, explicit uncertainty and
  reproduction sections, linked artifacts, and bounded conclusions. Original observed results unchanged; canonical bootstrap intervals are a labeled new analysis.
- New `src/dllm_unmasking/analyze_phase7.py`, `tests/test_phase7.py`, `scripts/check_cpu.py`,
  `research/benchmark_selection.json`, and `requirements-analysis.txt`.
- Updated `.gitignore` and `requirements.txt` for private deployment files and
  directly imported, locally exercised analysis dependencies.
- Sanitized `experiments/phase3/run.pbs`, `experiments/phase4/run.pbs`, `experiments/phase5/run.pbs`, `experiments/phase6/run.pbs`
  and corresponding Phase 3–6 guide references. Phase 4–6 path examples now use
  environment variables. Earlier scheduler flags and scientific behavior are preserved.

## Human review before any future publication

Decide whether the historical allocation identifier and retained queue/module
names comply with institutional policy. Review model/dataset redistribution and
license obligations before sharing raw text or weights; none are newly bundled.
Historical bootstrap provenance cannot be recovered; its values are explicitly
archived and superseded by a fully specified current analysis. The absence of a
public raw-artifact archive remains a reproducibility limitation. GPU dependency installation, cache
preflight, and scheduler acceptance were not revalidated on HPC. This cleanup is
ready for review locally, with no publication action taken.

## Final publication-readiness pass

**Verdict: READY for publication of this exploratory repository.** No credential
was detected in the audited scope. The historical allocation string was extracted
from the old PBS script and searched literally across current tracked files:
zero matches. All current public candidates also passed the privacy scan. The
identifier remains in commit `171a7b0` in Phase 3–6 guides and PBS scripts; it selects
a scheduler billing/allocation project and does not authenticate a user. It is
infrastructure metadata, not an API token, password or private key. No rotation or
history rewrite is indicated by this finding. Institutional metadata-disclosure
rules are not independently known; the residual historical exposure is documented
for the repository owner's review, rather than claimed erased.

Current Phase 7 inference observations are unchanged. The historical bootstrap
procedure is not claimed reproduced. New intervals are generated and published
with deterministic provenance; their pointwise coverage and exploratory scope
remain limitations. Exact McNemar tests and Holm adjustments are separately
identified, including the P4 bootstrap/exact-test disagreement. No new GPU data
or scientific inference implementation changes were made.

The complete candidate inventory below includes all edits from the cleanup and
this final pass. No files are staged, deleted, committed or pushed.

| Status | File |
| --- | --- |
| Modified | `.gitignore` |
| Modified | `experiments/phase3/README.md` |
| Modified | `experiments/phase4/README.md` |
| Modified | `experiments/phase5/README.md` |
| Modified | `experiments/phase6/README.md` |
| Modified | `README.md` |
| New | `src/dllm_unmasking/analyze_phase7.py` |
| New | `docs/AUDIT.md` |
| New | `docs/PHASE1.md` |
| New | `docs/PROJECT_MAP.md` |
| New | `docs/REPRODUCIBILITY.md` |
| Modified | `reports/phase7/PHASE7_REPORT.md` |
| New | `reports/phase7/bootstrap_provenance.json` |
| New | `reports/phase7/paired_reanalysis.csv` |
| New | `requirements-analysis.txt` |
| Modified | `requirements.txt` |
| New | `research/benchmark_selection.json` |
| Modified | `experiments/phase3/run.pbs` |
| Modified | `experiments/phase4/run.pbs` |
| Modified | `experiments/phase5/run.pbs` |
| Modified | `experiments/phase6/run.pbs` |
| New | `tests/test_phase7.py` |
| New | `scripts/check_cpu.py` |

## Structural-refactor addendum

See the [movement plan](STRUCTURE_PLAN.md) and [complete layout/movement inventory](STRUCTURE.md).
Tracked moves were performed with `git mv`, so rename records are staged; subsequent
content edits and new files remain unstaged. No commit or push was performed.
The previous cleanup inventory is historical and does not describe current paths.

All 40 CPU tests passed after editable installation: the original 36 tests and
four structural tests. Reversing only the recorded import/path substitutions
reconstructs every moved module byte-for-byte with its original SHA256. A test
verifies an altered inference seed cannot pass historical reconstruction. Configs,
original samplers without internal imports, and the numerical evaluator retain
identical bytes. Baseline validation accepts direct or exact reconstructed hashes;
new inference manifests retain actual current physical hashes and fail closed
against old interrupted-run manifests.

Public Phase 7 numerical tables, bootstrap provenance and regenerated PNG figures
match their pre-refactor versions. Original private outputs remain untouched.
Only documentation paths in report Markdown were adjusted; reported observations,
intervals and p-values are unchanged. CLI help works from outside the checkout;
PBS commands retain the repository-root submission requirement and use installed
modules. Single-file deployment snippets were replaced to preserve package layout.

Privacy protections and the non-credential historical allocation finding remain
as documented above. No history was rewritten. Local editable-install metadata
may contain absolute paths and is explicitly ignored. Packaging-tool updates
were limited to local setuptools/wheel for editable-install verification; no
models or datasets were downloaded. Scheduler acceptance and actual GPU inference
were not run. Source layout changes require a new run directory; use the original
checkout for resuming pre-refactor interrupted jobs.

Final structural checks also passed on a public-files-only copy: editable install,
33 passing fixture tests with seven explicit private-artifact skips, and byte-exact
Phase 7 table/provenance reproduction. All three regenerated PNGs also match the
pre-move generated PNGs byte-for-byte. Historical Phase 6 offline analysis was
verified on an independent copy of its journal/manifest; original files were not
modified. Python imports/syntax, seven PBS shell templates, Markdown targets,
privacy scans, and both staged/unstaged whitespace checks passed. Reachable
history remains three commits; credential scans found no current or historical
credentials. The known allocation identifier remains historical metadata.
