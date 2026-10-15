# Structural cleanup plan and preservation contract

This plan was prepared before moving files. Baseline: commit `fa8971a`.
A full tracked-file backup and SHA256 inventory were saved outside the repository.

1. Move reusable Python modules into `src/dllm_unmasking/`, retaining their existing
   names and function bodies. Move runners into importable `experiments.phaseN`
   packages, with PBS templates and phase guides alongside them.
2. Move YAML files unchanged into `configs/phaseN.yaml`, CPU tests into `tests/`,
   and the CPU verification utility into `scripts/`. Keep result/figure locations
   and ignored raw artifact directories unchanged.
3. Declare editable installation in `pyproject.toml`. Use qualified imports and
   `python -m` entry points. A shared layout utility finds the checkout root for
   default config/output paths. Explicit relative paths are repository-relative.
4. Preserve legacy source-name lookup for manifests. For historical source hashes,
   reverse only the recorded literal import/path substitutions and require an
   exact original SHA256 match. The mapping stores no original code or raw data.
   New manifests continue to hash actual current files. Do not weaken checks or
   edit old journals/manifests to force resumption.
5. Update PBS root/module invocation, all documentation links and commands, test
   discovery, and Phase 7 generator provenance lookup. Preserve all scientific
   operations, seeds, settings and numerical artifacts.
6. Verify every old module can be reconstructed byte-for-byte by reversing the
   recorded structural changes, run all CPU tests, independently regenerate Phase 7
   summaries/tables/figures, validate links, and repeat privacy checks.

No GPU execution, model downloads, Git history edits, commits or pushes are part
of this work. Historical journals are read-only. Resume with different physical
source hashes remains fail-closed; use the original checkout for interrupted
pre-refactor inference runs rather than rewriting their manifests.
