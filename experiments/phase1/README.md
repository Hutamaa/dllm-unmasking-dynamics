# Phase 1: position-order benchmark

See the [historical Phase 1/1.5 methodology](../../docs/PHASE1.md) and
[current reproduction guide](../../docs/REPRODUCIBILITY.md).

From the repository root after editable installation:

```bash
python -m experiments.phase1.run --config configs/phase1.yaml
```

This performs GPU inference only inside a PBS allocation. Pin the dataset SHA
in a separate reproduction config as documented; the historical default remains
unchanged. The public [PBS template](run.pbs) requires site configuration.
