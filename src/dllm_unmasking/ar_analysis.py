"""Phase 8 observed AR metrics and matched comparisons; never runs models."""
import csv
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.phase2_data import atomic_json
from dllm_unmasking.ar_baseline import MODEL, model_spec, budget_label

FRONTIER = ('confidence P8', 'Confidence + random revision', 'left_to_right P4', 'left_to_right P1')


def baseline(cfg, questions):
    ids = {q['question_id']: q['stratum'] for q in questions}
    groups = {name: {} for name in FRONTIER}
    with Path(cfg['diffusion_results']).open(newline='') as file:
        for row in csv.DictReader(file):
            if row['method'] not in groups:
                continue
            qid = int(row['question_id'])
            if qid not in ids or row['stratum'] != ids[qid] or qid in groups[row['method']] or row['correct'] not in ('True', 'False'):
                raise ValueError('Invalid diffusion matched set')
            groups[row['method']][qid] = dict(correct=row['correct'] == 'True', gpu_seconds=float(row['gpu_seconds']), wall_seconds=float(row['wall_seconds']))
    if any(set(group) != set(ids) for group in groups.values()):
        raise ValueError('Require all four matched diffusion frontier configurations')
    return groups


def summarize(out, records, questions, cfg):
    cap = cfg['max_new_tokens']; comparison = budget_label(cap)
    if any(r.get('max_new_tokens', 256) != cap for r in records.values()):
        raise ValueError('Never pool records from different token budgets')
    model = cfg['model']; policy = model_spec(model)['policy']
    if any(r.get('model', MODEL) != model or r['policy'] != policy or r.get('model_revision', cfg['model_revision']) != cfg['model_revision'] for r in records.values()):
        raise ValueError('Never pool or relabel records from different AR models')
    out = Path(out); groups = baseline(cfg, questions); ids = sorted(records)
    if not ids:
        atomic_json(out / 'completion.json', dict(n_completed=0, expected=60, complete=False))
        return
    rows = [records[q] for q in ids]
    csv_rows = [dict(max_new_tokens=cap, comparison_kind=comparison, **{k: v for k, v in r.items() if k not in ('generated_token_ids', 'rendered_prompt', 'max_new_tokens')} ) for r in rows]
    write_csv(out / 'results.csv', csv_rows)
    summary = []
    for stratum in ('all', '0-2', '3-4', '5+'):
        rs = [r for r in rows if stratum == 'all' or r['stratum'] == stratum]
        row = dict(max_new_tokens=cap, comparison_kind=comparison, model=model, model_revision=cfg['model_revision'], stratum=stratum, n=len(rs), expected_n=60 if stratum == 'all' else 20,
                   correct=sum(r['correct'] for r in rs), accuracy=mean(r['correct'] for r in rs) if rs else None,
                   extraction_failures=sum(r['extraction_failed'] for r in rs),
                   final_answer_markers=sum('####' in r['response'] for r in rs),
                   parser_fallbacks=sum(r['extraction_method'] == 'last_number' for r in rs),
                   eos_terminated=sum(r['termination'] == 'eos' for r in rs),
                   length_limited=sum(r['termination'] == 'length_limit' for r in rs))
        for field in ('gpu_prefill_decode_seconds', 'gpu_prefill_forward_seconds', 'gpu_decode_forward_seconds', 'generation_wall_seconds', 'generated_tokens', 'response_tokens', 'tokens_per_second', 'peak_allocated_bytes'):
            values = [r[field] for r in rs if r[field] is not None]
            row[field + '_n'] = len(values)
            for label, value in [('mean', mean(values) if values else None), ('median', median(values) if values else None), ('p90', float(np.quantile(values, .9, method='linear')) if values else None)]:
                row[field + '_' + label] = value
        summary.append(row)
    write_csv(out / 'summary.csv', summary)
    rng = np.random.Generator(np.random.PCG64(cfg['bootstrap_seed']))
    paired = []
    for method in FRONTIER:
        # B-minus-A: AR minus diffusion; question IDs matched, never independent observations.
        differences = np.array([int(records[q]['correct']) - int(groups[method][q]['correct']) for q in ids])
        recovered = int((differences == 1).sum()); regressed = int((differences == -1).sum()); n = recovered + regressed
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(recovered, regressed) + 1)) / 2**n) if n else 1.
        draws = np.zeros(cfg['bootstrap_repetitions'])
        for stratum in ('0-2', '3-4', '5+'):
            values = differences[[i for i, q in enumerate(ids) if records[q]['stratum'] == stratum]]
            if len(values):
                draws += rng.choice(values, size=(len(draws), len(values)), replace=True).sum(axis=1)
        low, high = np.quantile(draws * 100 / len(ids), [.025, .975], method='linear')
        paired.append(dict(max_new_tokens=cap, diffusion_token_budget=256, comparison_kind=comparison, diffusion=method, ar=policy, ar_model=model, model_revision=cfg['model_revision'], n_matched=len(ids), expected_n=60,
                           diffusion_correct=sum(groups[method][q]['correct'] for q in ids), ar_correct=sum(r['correct'] for r in rows),
                           ar_minus_diffusion_pp=float(differences.mean() * 100), recovered=recovered, regressed=regressed,
                           mcnemar_exact_p=p, bootstrap_95_low_pp=float(low), bootstrap_95_high_pp=float(high)))
    running = 0.
    for rank, row in enumerate(sorted(paired, key=lambda r: r['mcnemar_exact_p'])):
        running = max(running, min(1., (4-rank) * row['mcnemar_exact_p'])); row['holm_four_tests_p'] = running
    write_csv(out / 'paired_diffusion.csv', paired)
    latency = []
    for q in ids:
        r = records[q]
        latency.append(dict(max_new_tokens=cap, comparison_kind=comparison, question_id=q, method=policy, observed_gpu_seconds=r['gpu_prefill_decode_seconds'],
                            observed_wall_seconds=r['generation_wall_seconds'], generated_tokens=r['generated_tokens'],
                            diffusion_forward_passes=None, scope='AR native generate including prefill and generation scheduling; excludes loading/tokenization', directly_comparable=False))
        for method in FRONTIER:
            p = 256 if method.endswith('P1') else 64 if method.endswith('P4') else 32
            latency.append(dict(max_new_tokens=256, comparison_kind=comparison, question_id=q, method=method, observed_gpu_seconds=groups[method][q]['gpu_seconds'],
                                observed_wall_seconds=groups[method][q]['wall_seconds'], generated_tokens=None,
                                diffusion_forward_passes=p, scope='Historical diffusion full sampler stream interval; no autoregressive prefill equivalent', directly_comparable=False))
    write_csv(out / 'latency_comparison.csv', latency)
    atomic_json(out / 'completion.json', dict(n_completed=len(ids), expected=60, complete=len(ids) == 60))
