"""Validate public Phase 7 tables and regenerate figures without raw responses or inference."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Expected an explicit True/False value')
    return value == 'True'


def load(source, selection_path):
    selection = json.loads(selection_path.read_text())
    strata = {q['question_id']: q['stratum'] for q in selection['questions']}
    if len(strata) != 60 or Counter(strata.values()) != {'0-2': 20, '3-4': 20, '5+': 20}:
        raise ValueError('Expected the fixed 60-question selection and 20/20/20 strata')
    if set(strata).intersection(selection['excluded_phase1_ids']):
        raise ValueError('Phase 1 question overlap')
    rows = read_csv(source / 'unified_per_question.csv')
    groups = defaultdict(dict)
    for row in rows:
        qid = int(row['question_id']); method = row['method']
        if qid not in strata or row['stratum'] != strata[qid] or qid in groups[method]:
            raise ValueError('Unknown, mismatched, or duplicate evaluation')
        record = dict(row, question_id=qid, correct=boolean(row['correct']),
                      extraction_failed=boolean(row['extraction_failed']),
                      forward_passes=int(row['forward_passes']))
        for field in ('gpu_seconds', 'wall_seconds'):
            record[field] = float(row[field])
            if not math.isfinite(record[field]) or record[field] < 0:
                raise ValueError('Invalid observed timing')
        if record['extraction_failed'] and record['correct']:
            raise ValueError('Failed extraction cannot be correct')
        if record['forward_passes'] <= 0:
            raise ValueError('Invalid forward count')
        groups[method][qid] = record
    expected = {r['method']: r for r in read_csv(source / 'method_summary.csv')}
    if len(rows) != 1020 or len(groups) != 17 or set(groups) != set(expected):
        raise ValueError('Expected 17 configurations × 60 questions = 1020 rows')
    summaries = []
    for method, records in groups.items():
        if set(records) != set(strata):
            raise ValueError('Incomplete matched set')
        rs = list(records.values())
        summary = dict(method=method, n=len(rs), correct=sum(r['correct'] for r in rs),
                       accuracy=mean(r['correct'] for r in rs),
                       mean_gpu=mean(r['gpu_seconds'] for r in rs),
                       mean_wall=mean(r['wall_seconds'] for r in rs),
                       total_forwards=sum(r['forward_passes'] for r in rs),
                       extraction_failures=sum(r['extraction_failed'] for r in rs))
        for field, value in summary.items():
            if field != 'method' and not math.isclose(value, float(expected[method][field]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Published summary mismatch: {method} / {field}')
        if len({r['forward_passes'] for r in rs}) != 1:
            raise ValueError('Unexpected varying forward budget')
        summaries.append(summary)
    for row in summaries:
        row['pareto'] = not any(other['mean_gpu'] <= row['mean_gpu'] and other['accuracy'] >= row['accuracy']
            and (other['mean_gpu'] < row['mean_gpu'] or other['accuracy'] > row['accuracy']) for other in summaries)
        if row['pareto'] != boolean(expected[row['method']]['pareto']):
            raise ValueError('Published frontier mismatch')
    return groups, sorted(summaries, key=lambda r: r['mean_gpu']), strata


def paired_tests(source, groups, strata, seed, repetitions):
    """New explicitly stratified bootstrap; original report intervals remain unchanged."""
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    ids = sorted(strata)
    for published in read_csv(source / 'selected_paired_tests.csv'):
        a, b = published['a'], published['b']
        differences = np.array([int(groups[b][q]['correct']) - int(groups[a][q]['correct']) for q in ids])
        recovered = int((differences == 1).sum()); regressed = int((differences == -1).sum())
        n = recovered + regressed
        p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(recovered, regressed) + 1)) / 2**n) if n else 1.0
        delta = float(differences.mean() * 100)
        for field, value in [('delta_pp', delta), ('recovered', recovered), ('regressed', regressed), ('mcnemar_exact_p', p)]:
            if not math.isclose(value, float(published[field]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Published paired result mismatch: {a} / {b} / {field}')
        draws = np.zeros(repetitions)
        for stratum in ('0-2', '3-4', '5+'):
            values = differences[[i for i, q in enumerate(ids) if strata[q] == stratum]]
            draws += rng.choice(values, size=(repetitions, len(values)), replace=True).sum(axis=1)
        low, high = np.quantile(draws * 100 / len(ids), [.025, .975], method='linear')
        result.append(dict(a=a, b=b, n_matched=60, delta_pp=delta, recovered=recovered,
                           regressed=regressed, mcnemar_exact_p=p,
                           stratified_bootstrap_95_low_pp=float(low), stratified_bootstrap_95_high_pp=float(high)))
    # Holm adjustment across these 10 exploratory contrasts; original unadjusted values retained.
    running = 0.0
    for rank, row in enumerate(sorted(result, key=lambda r: r['mcnemar_exact_p'])):
        running = max(running, min(1.0, (len(result) - rank) * row['mcnemar_exact_p']))
        row['holm_adjusted_p'] = running
    return result


def paired_markdown(rows):
    lines = ['| Comparison (A → B) | Change (pp) | New-analysis 95% CI (pp) | Exact p | Holm p |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['a']} → {row['b']} | {row['delta_pp']:+.1f} | "
                     f"[{row['stratified_bootstrap_95_low_pp']:.1f}, {row['stratified_bootstrap_95_high_pp']:.1f}] | "
                     f"{row['mcnemar_exact_p']:.6g} | {row['holm_adjusted_p']:.6g} |")
    return '\n'.join(lines)


def plots(out, groups, summaries, strata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_method = {r['method']: r for r in summaries}
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy, label in [('left_to_right', 'Left-to-right'), ('confidence', 'Confidence')]:
        ax.plot([1, 4, 8], [100 * by_method[f'{policy} P{p}']['accuracy'] for p in (1, 4, 8)], marker='o', label=label)
    ax.set(xlabel='New tokens per decoding step', ylabel='Numerical-answer accuracy (%)', xticks=[1, 4, 8], ylim=(0, 100), title='Matched GSM8K questions (n=60 per configuration)')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'accuracy_parallelism.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in summaries:
        ax.scatter(r['mean_gpu'], 100*r['accuracy'], color='tab:orange' if r['pareto'] else 'tab:blue')
        if r['pareto'] or r['method'].startswith('batched_influence'):
            influence = r['method'].startswith('batched_influence')
            offset = (-6, 6) if influence else (4, -14) if r['method'] == 'confidence P8' else (4, 6)
            ax.annotate(r['method'], (r['mean_gpu'], 100*r['accuracy']), xytext=offset, textcoords='offset points', fontsize=8, ha='right' if influence else 'left')
    frontier = sorted((r for r in summaries if r['pareto']), key=lambda r: r['mean_gpu'])
    ax.plot([r['mean_gpu'] for r in frontier], [100*r['accuracy'] for r in frontier], '--', color='tab:orange', label='Observed sample frontier')
    ax.set(xscale='log', xlabel='Mean recorded GPU decode interval (seconds, log scale)', ylabel='Numerical-answer accuracy (%)', ylim=(0, 85), title='Cross-job observed latency; n=60 per configuration')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'accuracy_latency_pareto.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ('0-2', '3-4', '5+')
    for method in ('left_to_right P1', 'left_to_right P4', 'left_to_right P8', 'confidence P8', 'LTR spaced P8'):
        ax.plot(labels, [100*mean(r['correct'] for q, r in groups[method].items() if strata[q] == s) for s in labels], marker='o', label=method)
    ax.set(xlabel='Gold calculation-count proxy', ylabel='Numerical-answer accuracy (%)', ylim=(0, 100), title='Exploratory strata (n=20 per stratum/configuration)')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'accuracy_by_stratum.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='reports/phase7')
    parser.add_argument('--selection', default='research/benchmark_selection.json')
    parser.add_argument('--output-dir', default='results/phase7_reproduced')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args()
    source, out = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    selection = Path(args.selection).resolve()
    if out == source or source in out.parents or out in source.parents or out == selection or out in selection.parents:
        parser.error('Inputs and generated outputs must not overlap')
    if args.bootstrap_repetitions < 1:
        parser.error('Bootstrap repetitions must be positive')
    inputs = [source / n for n in ('unified_per_question.csv', 'method_summary.csv', 'selected_paired_tests.csv')] + [selection]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    groups, summaries, strata = load(source, selection)
    comparisons = paired_tests(source, groups, strata, args.seed, args.bootstrap_repetitions)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'method_summary.csv', summaries)
    write_csv(out / 'paired_reanalysis.csv', comparisons)
    (out / 'paired_reanalysis.md').write_text(paired_markdown(comparisons) + '\n')
    plots(out, groups, summaries, strata)
    import matplotlib
    metadata = dict(analysis_version='phase7-stratified-reanalysis-v1',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    input_sha256=hashes, seed=args.seed, bootstrap_repetitions=args.bootstrap_repetitions,
                    rng='NumPy Generator(PCG64(seed)); one shared stream across contrasts',
                    order='Contrast rows in selected_paired_tests.csv; ascending question IDs; strata 0-2,3-4,5+',
                    resampling='For each contrast and stratum draw a (10000,20) index array with replacement; sum paired B-minus-A correctness differences across strata, divide by 60, multiply by 100. Array first dimension follows bootstrap_repetitions.',
                    interval='Pointwise 95% percentile; quantiles 0.025,0.975; NumPy method=linear',
                    multiplicity='Holm across ten exact McNemar p-values; bootstrap intervals are not multiplicity-adjusted',
                    numpy=np.__version__, matplotlib=matplotlib.__version__,
                    bootstrap='Paired question bootstrap within fixed 20/20/20 strata; percentile intervals',
                    original_intervals='Original bootstrap implementation/provenance unavailable; preserved, not claimed reproduced',
                    observations=1020, methods=17, questions=60)
    (out / 'reproduction_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name] for p in inputs):
        raise RuntimeError('Input changed during analysis')
    print('Validated 1020 observations, 17 matched methods, published aggregates and 10 exact paired tests. Generated three figures; original files unchanged.')


if __name__ == '__main__':
    main()
