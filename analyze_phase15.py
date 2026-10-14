"""Read-only, CPU-only Phase 1.5 analysis. No model/tokenizer imports or downloads."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from statistics import mean, median, pstdev

import numpy as np

STRATEGIES = ('random', 'left_to_right', 'right_to_left', 'confidence', 'influence')
FOCUS = (('left_to_right', 'right_to_left'), ('influence', 'random'), ('influence', 'confidence'))
# Fixed interpretable cutoffs; not fitted to correctness.
GROUPS = {'calculation_count': ((0, 2, '0–2'), (3, 4, '3–4'), (5, math.inf, '5+')),
          'question_words': ((0, 40, '0–40'), (41, 70, '41–70'), (71, math.inf, '71+')),
          'numeric_mentions': ((0, 3, '0–3'), (4, 6, '4–6'), (7, math.inf, '7+'))}
NUMBERS = re.compile(r'(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?|(?<!\w)\.\d+')
SPECIAL = re.compile(r'<\|[^<>]+\|>|\[MASK\]')


def write_csv(out, name, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ['status']
    with (out / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(source):
    paths = [source / name for name in ('records.jsonl', 'questions.json', 'results.csv')]
    for p in paths:
        if not p.is_file():
            raise ValueError(f'Missing required input: {p}')
    if (source / 'manifest.json').exists():
        paths.append(source / 'manifest.json')
    hashes = {p.name: digest(p) for p in paths}
    records = []
    with paths[0].open() as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid journal line {line_number}; input will not be repaired') from e
    questions = json.loads(paths[1].read_text())
    qmap = {q['question_id']: q for q in questions}
    keys = {(r['question_id'], r['strategy']) for r in records}
    expected = {(qid, s) for qid in qmap for s in STRATEGIES}
    if len(questions) != 150 or len(qmap) != 150 or len(records) != 750 or len(keys) != 750 or keys != expected:
        raise ValueError('Require exactly 150 unique questions × all five strategies = 750 unique evaluations')
    for r in records:
        q = qmap[r['question_id']]
        if any(r[k] != q[k] for k in ('question', 'gold_answer')):
            raise ValueError(f'Question/gold mismatch: {r["question_id"]}')
        if type(r['correct']) is not bool or type(r['extraction_failed']) is not bool:
            raise ValueError('Correctness/extraction flags must be JSON booleans')
        if r['correct'] != (r['predicted_numeric'] is not None and r['predicted_numeric'] == r['gold_numeric']):
            raise ValueError('Stored correctness does not match stored extracted answers')
        if r['extraction_failed'] != (r['predicted_numeric'] is None):
            raise ValueError('Extraction flag mismatch')
    with paths[2].open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    csv_map = {(int(r['question_id']), r['strategy']): r for r in csv_rows}
    if len(csv_rows) != 750 or len(csv_map) != 750 or set(csv_map) != expected:
        raise ValueError('results.csv coverage disagrees with records.jsonl')
    for r in records:
        row = csv_map[(r['question_id'], r['strategy'])]
        for key in ('correct', 'extraction_failed'):
            if row[key].lower() != str(r[key]).lower():
                raise ValueError(f'CSV mismatch: {key}')
        for key in ('question', 'gold_answer', 'response', 'raw_response', 'predicted_numeric', 'gold_numeric'):
            if key in row and row[key] != ('' if r[key] is None else str(r[key])):
                raise ValueError(f'CSV mismatch: {key}')
    records.sort(key=lambda r: (r['question_id'], STRATEGIES.index(r['strategy'])))
    audit = []
    for strategy in STRATEGIES:
        rows = [r for r in records if r['strategy'] == strategy]
        audit.append(dict(strategy=strategy, evaluations=len(rows), questions=len({r['question_id'] for r in rows}),
                          correct=sum(r['correct'] for r in rows), trajectory_records=sum(bool(r.get('trajectory')) for r in rows),
                          token_id_records=sum(bool(r.get('generated_token_ids')) for r in rows),
                          greedy_round_records=sum(any(t.get('greedy_rounds') for t in r.get('trajectory', [])) for r in rows)))
    return records, qmap, paths, hashes, audit


def bootstrap(values, rng, repetitions):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return None, None
    draws = rng.integers(0, len(a), size=(repetitions, len(a)))
    samples = a[draws].mean(axis=1)
    return tuple(float(v) for v in np.quantile(samples, [.025, .975]))


def wilson(correct, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = correct / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return center-width, center+width


def paired_row(a, b, qids, lookup, rng, repetitions):
    pairs = [(int(lookup[(q,a)]['correct']), int(lookup[(q,b)]['correct'])) for q in qids]
    n = len(pairs)
    a_only = sum(x == 1 and y == 0 for x,y in pairs)
    b_only = sum(x == 0 and y == 1 for x,y in pairs)
    discordant = a_only + b_only
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(a_only,b_only)+1)) / 2**discordant) if discordant else 1.
    low, high = bootstrap([x-y for x,y in pairs], rng, repetitions)
    return dict(strategy_a=a, strategy_b=b, n_matched=n,
                a_correct=sum(x for x,y in pairs), b_correct=sum(y for x,y in pairs),
                both_correct=sum(x==y==1 for x,y in pairs), both_wrong=sum(x==y==0 for x,y in pairs),
                a_only_correct=a_only, b_only_correct=b_only, discordant=discordant,
                accuracy_gap_a_minus_b=(a_only-b_only)/n if n else None,
                paired_bootstrap_95_low=low, paired_bootstrap_95_high=high,
                mcnemar_exact_p=p if n else None)


def holm(rows, key='mcnemar_exact_p', output='holm_adjusted_p'):
    ordered = sorted([r for r in rows if r[key] is not None], key=lambda r:r[key])
    previous = 0.
    for i,r in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-i)*r[key]))
        r[output] = previous
    for r in rows:
        r.setdefault(output, None)


def corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def complexity(qmap, lookup, rng, repetitions):
    features = []
    for qid,q in sorted(qmap.items()):
        f = dict(question_id=qid, calculation_count=len(re.findall(r'<<.*?>>',q['gold_answer'], re.S)),
                 question_words=len(q['question'].split()), numeric_mentions=len(NUMBERS.findall(q['question'])))
        features.append(f)
    accuracy, gaps, trends = [], [], []
    for feature,bins in GROUPS.items():
        for lo,hi,label in bins:
            ids = [f['question_id'] for f in features if lo <= f[feature] <= hi]
            for strategy in STRATEGIES:
                correct = sum(lookup[(q,strategy)]['correct'] for q in ids)
                low,high = wilson(correct,len(ids))
                accuracy.append(dict(proxy=feature, group=label, strategy=strategy, n_questions=len(ids),
                                     correct=correct, accuracy=correct/len(ids) if ids else None,
                                     wilson_95_low=low, wilson_95_high=high))
            for a,b in FOCUS:
                gaps.append(dict(proxy=feature, group=label, **paired_row(a,b,ids,lookup,rng,repetitions)))
        for a,b in FOCUS:
            x = np.array([f[feature] for f in features],dtype=float)
            y = np.array([int(lookup[(f['question_id'],a)]['correct'])-int(lookup[(f['question_id'],b)]['correct']) for f in features])
            observed = corr(x,y)
            extremes = 0
            if observed is not None:
                for _ in range(repetitions):
                    extremes += abs(corr(x,rng.permutation(y))) >= abs(observed)-1e-12
            trends.append(dict(proxy=feature,strategy_a=a,strategy_b=b,n_questions=len(features),
                               pearson_proxy_vs_paired_gap=observed,
                               permutation_p=(extremes+1)/(repetitions+1) if observed is not None else None))
    holm(trends,'permutation_p','holm_adjusted_p')
    # Subgroup intervals/tests are exploratory; no claim of interaction from differing significance.
    return features,accuracy,gaps,trends


def trajectories(records):
    windows, curves, rounds, question_influence = [], [], [], []
    for r in records:
        trace = r.get('trajectory',[])
        tokens = r.get('generated_token_ids',[])
        if not trace or not tokens:
            continue
        length,steps = len(tokens),len(trace)
        seen = set()
        for i,t in enumerate(trace):
            positions=t['positions']
            if (t['step'] != i or not positions or len(positions)!=len(set(positions))
                or len(positions)!=len(t['confidences']) or len(positions)!=len(t['token_ids'])
                or any(p in seen or not 0<=p<length for p in positions)
                or any(tokens[p]!=token for p,token in zip(positions,t['token_ids']))
                or any(not math.isfinite(c) or not 0<=c<=1 for c in t['confidences'])):
                raise ValueError(f'Invalid trajectory: {r["question_id"]}, {r["strategy"]}')
            seen.update(positions)
            normalized=[p/(length-1) if length>1 else 0. for p in positions]
            curves.append(dict(question_id=r['question_id'],strategy=r['strategy'],correct=r['correct'],
                               step=i,step_fraction=(i+1)/steps,n_positions=len(positions),
                               mean_normalized_position=mean(normalized)))
        if seen != set(range(length)):
            raise ValueError('Trajectory does not reveal each response position exactly once')
        for fraction in (.1,.25,.5):
            count=math.ceil(steps*fraction)
            selected=[p/(length-1) if length>1 else 0. for t in trace[:count] for p in t['positions']]
            windows.append(dict(question_id=r['question_id'],strategy=r['strategy'],correct=r['correct'],
                                window_fraction=fraction,steps_in_window=count,total_steps=steps,
                                n_revealed=len(selected),mean_normalized_position=mean(selected)))
        own=[]
        if r['strategy']=='influence':
            for t in trace:
                stored=t.get('greedy_rounds',[])
                if not stored:
                    continue
                if len(stored)!=len(t['positions']) or len(t['influence_scores'])!=len(stored):
                    raise ValueError('Influence round/selection mismatch')
                for j,g in enumerate(stored):
                    cp,cs=g['candidate_positions'],g['candidate_scores']
                    position,score=t['positions'][j],t['influence_scores'][j]
                    if not cs or len(cp)!=len(cs) or position not in cp or not all(math.isfinite(v) for v in cs):
                        raise ValueError('Invalid influence candidate data')
                    if abs(cs[cp.index(position)]-score)>1e-5 or abs(max(cs)-score)>1e-5:
                        raise ValueError('Stored winner is not maximum influence')
                    sorted_scores=sorted(cs,reverse=True)
                    row=dict(question_id=r['question_id'],correct=r['correct'],step=t['step'],greedy_round=j,
                             step_fraction=(t['step']+1)/steps,position=position,
                             normalized_position=position/(length-1) if length>1 else 0.,
                             selected_score_nats=score,selected_confidence=t['confidences'][j],candidate_count=len(cs),
                             candidate_min_nats=min(cs),candidate_median_nats=median(cs),candidate_max_nats=max(cs),
                             candidate_std_nats=pstdev(cs),candidate_negative_fraction=sum(v<0 for v in cs)/len(cs),
                             winner_margin_nats=sorted_scores[0]-sorted_scores[1] if len(cs)>1 else None,
                             influence_top_k=r.get('influence_top_k'),approximation=r.get('influence_approximation'),
                             candidate_positions=json.dumps(cp),candidate_scores_nats=json.dumps(cs))
                    rounds.append(row);own.append(row)
            if own:
                early=[t for t in own if t['step']<math.ceil(.1*steps)]
                question_influence.append(dict(question_id=r['question_id'],correct=r['correct'],n_greedy_rounds=len(own),
                    mean_selected_score_nats=mean(t['selected_score_nats'] for t in own),
                    early10_score_nats=mean(t['selected_score_nats'] for t in early),
                    early10_normalized_position=mean(t['normalized_position'] for t in early),
                    early10_selected_confidence=mean(t['selected_confidence'] for t in early),
                    mean_selected_confidence=mean(t['selected_confidence'] for t in own),
                    mean_normalized_position=mean(t['normalized_position'] for t in own),
                    mean_candidate_std_nats=mean(t['candidate_std_nats'] for t in own),
                    mean_candidate_negative_fraction=mean(t['candidate_negative_fraction'] for t in own),
                    mean_winner_margin_nats=mean(t['winner_margin_nats'] for t in own if t['winner_margin_nats'] is not None)))
    summary=[]
    for strategy in STRATEGIES:
        for outcome in ('all','correct','incorrect'):
            for fraction in (.1,.25,.5):
                rows=[w for w in windows if w['strategy']==strategy and w['window_fraction']==fraction
                      and (outcome=='all' or w['correct']==(outcome=='correct'))]
                summary.append(dict(strategy=strategy,outcome=outcome,window_fraction=fraction,n_questions=len(rows),
                                    mean_normalized_position=mean(w['mean_normalized_position'] for w in rows) if rows else None))
    curve_summary=[]
    for strategy in STRATEGIES:
        for outcome in ('all','correct','incorrect'):
            rows=[c for c in curves if c['strategy']==strategy and (outcome=='all' or c['correct']==(outcome=='correct'))]
            for step,group in itertools.groupby(sorted(rows,key=lambda c:c['step_fraction']),key=lambda c:c['step_fraction']):
                g=list(group)
                curve_summary.append(dict(strategy=strategy,outcome=outcome,step_fraction=step,n_questions=len(g),
                                          mean_normalized_position=mean(t['mean_normalized_position'] for t in g)))
    return windows,summary,curve_summary,rounds,question_influence


def diagnostics(records, special_ids):
    rows=[]
    for r in records:
        raw=r.get('raw_response','');tokens=r.get('generated_token_ids',[])
        markers=[dict(marker=m.group(),char_offset=m.start()) for m in SPECIAL.finditer(raw)]
        token_positions={name:[i for i,t in enumerate(tokens) if t==identifier] for name,identifier in special_ids.items()}
        rows.append(dict(question_id=r['question_id'],strategy=r['strategy'],correct=r['correct'],
                         extraction_failed=r['extraction_failed'],extraction_method=r['extraction_method'],
                         predicted_numeric=r['predicted_numeric'],response_characters=len(r['response']),
                         response_words=len(r['response'].split()),generated_tokens=len(tokens),
                         observed_special_markers=len(markers),
                         first_special_char_offset=markers[0]['char_offset'] if markers else None,
                         eot_text_present=any(m['marker']=='<|eot_id|>' for m in markers),
                         endoftext_text_present=any(m['marker']=='<|endoftext|>' for m in markers),
                         special_marker_char_positions=json.dumps(markers),
                         confirmed_special_token_positions=json.dumps(token_positions),
                         eos_token_positions_available='eos' in special_ids,
                         first_eos_token_position=next(iter(token_positions.get('eos',[])),None)))
    summary=[]
    for s in STRATEGIES:
        for outcome in ('all','correct','incorrect'):
            a=[r for r in rows if r['strategy']==s and (outcome=='all' or r['correct']==(outcome=='correct'))]
            summary.append(dict(strategy=s,outcome=outcome,n_evaluations=len(a),
                extraction_failures=sum(r['extraction_failed'] for r in a),
                fallback_extractions=sum(r['extraction_method']=='last_number' for r in a),
                marker_extractions=sum(r['extraction_method']=='marker' for r in a),
                mean_response_words=mean(r['response_words'] for r in a) if a else None,
                mean_response_characters=mean(r['response_characters'] for r in a) if a else None,
                eot_text_present=sum(r['eot_text_present'] for r in a),
                endoftext_text_present=sum(r['endoftext_text_present'] for r in a)))
    return rows,summary


def influence_summary(question_rows):
    summary=[]
    metrics=[k for k in question_rows[0] if k.startswith('mean_') or k.startswith('early')] if question_rows else []
    for key in metrics:
        for outcome in ('correct','incorrect'):
            a=[r[key] for r in question_rows if r['correct']==(outcome=='correct')]
            summary.append(dict(metric=key,outcome=outcome,n_questions=len(a),mean=mean(a) if a else None,
                                median=median(a) if a else None,
                                descriptive_correlation_with_correct=corr([r[key] for r in question_rows],
                                                                          [int(r['correct']) for r in question_rows])))
    return summary


def representatives(out,qmap,lookup):
    categories=[]
    for a,b in FOCUS:
        for winner,loser in ((a,b),(b,a)):
            ids=[q for q in sorted(qmap) if lookup[(q,winner)]['correct'] and not lookup[(q,loser)]['correct']]
            categories.append((f'{winner} correct / {loser} incorrect',ids))
    categories += [('Any extraction failure',[q for q in sorted(qmap) if any(lookup[(q,s)]['extraction_failed'] for s in STRATEGIES)]),
                   ('All five incorrect',[q for q in sorted(qmap) if not any(lookup[(q,s)]['correct'] for s in STRATEGIES)])]
    text=['# Representative full responses','',
          'Examples are the two lowest question IDs per category, not a random sample or an error taxonomy.',
          'Stored correctness is numerical-extraction correctness. Arithmetic/reasoning diagnoses require human annotation.','']
    for category,ids in categories:
        text += [f'## {category} ({len(ids)} questions; showing {min(2,len(ids))})','']
        for qid in ids[:2]:
            text += [f'### Question {qid}',qmap[qid]['question'],'','Gold solution:','',qmap[qid]['gold_answer'],'']
            for s in STRATEGIES:
                r=lookup[(qid,s)]
                text += [f'#### {s}',f'Correct: {r["correct"]}; extracted: {r["predicted_numeric"]}; method: {r["extraction_method"]}.','',
                         'Visible response:','',r['response'],'','Full raw response (including special tokens):','',r['raw_response'],'']
    (out/'representative_responses.md').write_text('\n'.join(text))
    return categories


def plots(out,gaps,curves,question_influence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,(proxy,bins) in zip(axes,GROUPS.items()):
        labels=[label for _,_,label in bins]
        for pair_index,(a,b) in enumerate(FOCUS):
            rows=[r for r in gaps if r['proxy']==proxy and (r['strategy_a'],r['strategy_b'])==(a,b) and r['n_matched']]
            x=[labels.index(r['group'])+(pair_index-1)*.13 for r in rows]
            y=[r['accuracy_gap_a_minus_b'] for r in rows]
            ax.errorbar(x,y,yerr=[[r['accuracy_gap_a_minus_b']-r['paired_bootstrap_95_low'] for r in rows],
                                 [r['paired_bootstrap_95_high']-r['accuracy_gap_a_minus_b'] for r in rows]],
                        marker='o',capsize=3,label=f'{a} − {b}')
        counts={r['group']:r['n_matched'] for r in gaps if r['proxy']==proxy}
        ax.set_xticks(range(len(labels)),[f'{g}\nn={counts[g]}' for g in labels])
        ax.set_title(proxy.replace('_',' ')+' (proxy)');ax.axhline(0,color='gray',lw=1)
    axes[0].set_ylabel('Paired accuracy difference; 95% bootstrap CI')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Exploratory accuracy gaps by question proxies; positive favors first strategy')
    fig.tight_layout();fig.savefig(out/'complexity_gaps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,outcome in zip(axes,('all','correct','incorrect')):
        for s in STRATEGIES:
            rows=[r for r in curves if r['strategy']==s and r['outcome']==outcome]
            if rows:
                ax.plot([r['step_fraction'] for r in rows],[r['mean_normalized_position'] for r in rows],
                        label=f'{s} (n={rows[0]["n_questions"]})',lw=1.3)
        ax.set_title(outcome);ax.set_xlabel('Fraction of diffusion steps completed');ax.set_ylim(0,1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Mean reveal position (0=left, 1=right)')
    fig.suptitle('Observed reveal dynamics; correctness stratification is descriptive, not causal')
    fig.tight_layout();fig.savefig(out/'trajectory_dynamics.png',dpi=160);plt.close(fig)
    if question_influence:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for correct,label,color in [(False,'Incorrect','tab:red'),(True,'Correct','tab:blue')]:
            rows=[r for r in question_influence if r['correct']==correct]
            axes[0].scatter([r['early10_selected_confidence'] for r in rows],
                            [r['early10_score_nats'] for r in rows],alpha=.6,s=20,color=color,
                            label=f'{label} (n={len(rows)})')
        axes[0].set_xlabel('Mean selected confidence, first 10% of steps')
        axes[0].set_ylabel('Question mean influence, first 10% of steps (nats)');axes[0].legend()
        values=[[r['mean_selected_score_nats'] for r in question_influence if r['correct']==c] for c in (False,True)]
        if all(values):
            axes[1].boxplot(values,tick_labels=['Incorrect','Correct'],showmeans=True)
        axes[1].set_ylabel('Question mean selected influence (nats)')
        fig.suptitle('Stored top-K influence diagnostics; one observation per question')
        fig.tight_layout();fig.savefig(out/'influence_diagnostics.png',dpi=160);plt.close(fig)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def percent(value):
    return 'NA' if value is None else f'{100*value:.1f}%'


def points(value):
    return 'NA' if value is None else f'{100*value:.1f} pp'


def report(out,audit,paired,gaps,trends,windows,output_summary,inf_summary,categories,metadata):
    lines=['# Phase 1.5: observational analysis of unmasking dynamics','',
           '## Input and scope','',
           f'Validated **750 unique evaluations: 150 matched questions × five strategies**, from `{metadata["input_dir"]}`.',
           'The journal, selected questions, and CSV agree on question identities, answers, correctness, and responses. '
           'Input SHA-256 hashes are stored in `analysis_metadata.json`. No model calls, tokenizer downloads, or Phase 1 writes occur.','',
           'Every trajectory reveals all response positions exactly once. This is one sampled benchmark subset and one '
           'run per question/strategy, not independent decoding replicates. Recorded correctness is exact agreement of extracted '
           'numeric answers with the stored gold answer; it is not a human assessment of reasoning quality.',
           f'Recorded decoding settings: `{json.dumps(metadata["phase1_decoding"])}`. '
           f'Model revision: `{metadata["model_revision"]}`; dataset revision: `{metadata["dataset_revision"]}`.','',
           '## Overall observations','',
           table(['Strategy','Correct / questions','Accuracy'],[[r['strategy'],f'{r["correct"]}/{r["questions"]}',percent(r['correct']/r['questions'])] for r in audit]),'',
           '## Paired comparisons','',
           'Each pair uses the same question IDs. Exact two-sided McNemar tests condition on discordant pairs; '
           'Holm correction covers all ten overall pairwise tests. Bootstrap intervals resample whole questions '
           f'({metadata["bootstrap_repetitions"]:,} draws, seed {metadata["seed"]}) for the mean A−B correctness difference. '
           'They quantify question-sampling uncertainty under an IID-question approximation, not variation across model seeds.','',
           table(['A − B','Matched n','A only / B only','Gap (95% paired bootstrap)','Exact p','Holm p'],
                 [[f'{r["strategy_a"]} − {r["strategy_b"]}',r['n_matched'],f'{r["a_only_correct"]} / {r["b_only_correct"]}',
                   f'{points(r["accuracy_gap_a_minus_b"])} [{points(r["paired_bootstrap_95_low"])}, {points(r["paired_bootstrap_95_high"])}]',
                   f'{r["mcnemar_exact_p"]:.4g}',f'{r["holm_adjusted_p"]:.4g}'] for r in paired]),'',
           '## Do strategy gaps vary with complexity proxies?','',
           'Calculation count is the number of `<<...>>` annotations in the gold solution; question length is whitespace-separated '
           'word count; numerical mentions count digit-written numbers/fractions, not number words. These are **proxies, not validated '
           'reasoning depth**. Gold solutions are used only offline. Fixed bins: calculation count 0–2/3–4/5+; words 0–40/41–70/71+; '
           'numerical mentions 0–3/4–6/7+. Every bin, including empty bins, is retained in the CSVs.','']
    for proxy in GROUPS:
        rows=[r for r in gaps if r['proxy']==proxy]
        lines += [f'### {proxy}','',table(['Group','A − B','Matched n','A only / B only','Gap (95% paired bootstrap)'],
           [[r['group'],f'{r["strategy_a"]} − {r["strategy_b"]}',r['n_matched'],f'{r["a_only_correct"]} / {r["b_only_correct"]}',
             f'{points(r["accuracy_gap_a_minus_b"])} [{points(r["paired_bootstrap_95_low"])}, {points(r["paired_bootstrap_95_high"])}]'] for r in rows]),'']
    lines += ['`complexity_accuracy.csv` includes per-strategy correct/total counts and Wilson intervals; '
              '`complexity_gaps.csv` emphasizes matched gaps. Different subgroup p-values are not evidence that gaps differ between groups. '
              'To directly describe gap–proxy association, `gap_trends.csv` computes Pearson correlation between the numeric proxy and '
              'per-question A−B correctness (−1, 0, +1), with two-sided question-level permutation tests and Holm correction across nine trends. '
              'These are exploratory associations and can reflect correlated question properties, truncation, extraction behavior, or other confounders.','',
              table(['Proxy','A − B','n','Gap–proxy r','Permutation p','Holm p'],
                    [[r['proxy'],f'{r["strategy_a"]} − {r["strategy_b"]}',r['n_questions'],
                      'NA' if r['pearson_proxy_vs_paired_gap'] is None else f'{r["pearson_proxy_vs_paired_gap"]:.3f}',
                      'NA' if r['permutation_p'] is None else f'{r["permutation_p"]:.4g}',
                      'NA' if r['holm_adjusted_p'] is None else f'{r["holm_adjusted_p"]:.4g}'] for r in trends]),'',
              '### Main exploratory pattern','']
    for a,b in FOCUS:
        rows=[r for r in gaps if r['proxy']=='calculation_count' and (r['strategy_a'],r['strategy_b'])==(a,b)]
        description='; '.join(f'{r["group"]} calculations: {points(r["accuracy_gap_a_minus_b"])} gap (n={r["n_matched"]})' for r in rows)
        trend=next(r for r in trends if r['proxy']=='calculation_count' and (r['strategy_a'],r['strategy_b'])==(a,b))
        lines += [f'**{a} − {b}:** {description}. The descriptive gap–calculation-count correlation is '
                  f'{trend["pearson_proxy_vs_paired_gap"]:.3f}, with Holm-adjusted permutation p={trend["holm_adjusted_p"]:.4g}.','']
    lines += ['These are observed gap patterns, not an identified mechanism. In particular, weaker evidence of an '
              'influence gap trend is not proof that influence performance is invariant to complexity. The numeric-mention '
              '7+ bin contains very few questions; its estimates should not guide firm conclusions.','',
              '## Reveal dynamics','',
              'Positions are normalized as position/(response token count−1), with 0=left and 1=right. '
              'The first f fraction of steps uses ceil(f × total steps), including every reveal in those steps. '
              'Each question contributes its own mean; questions, rather than token events, are the unit of aggregation. '
              'The curves use the mean newly revealed position at each step, not cumulative position.','',
              table(['Strategy','Outcome','n','First 10%','First 25%','First 50%'],
                [[s,o,next(r['n_questions'] for r in windows if r['strategy']==s and r['outcome']==o),
                  *[f'{next(r["mean_normalized_position"] for r in windows if r["strategy"]==s and r["outcome"]==o and r["window_fraction"]==f):.3f}'
                    if next(r['n_questions'] for r in windows if r['strategy']==s and r['outcome']==o and r['window_fraction']==f) else 'NA'
                    for f in (.1,.25,.5)]] for s in STRATEGIES for o in ('correct','incorrect')]),'',
              'LTR and RTL are fixed-order reference controls; random is also shown. Correctness-stratified curves '
              'describe observed outputs. Conditioning on correctness and observing position differences does **not** establish '
              'that early reveal position caused correctness.','',
              '## Observable output diagnostics','',
              table(['Strategy','n','Extraction failures','Last-number fallback','Mean visible words','EoT marker present'],
                    [[r['strategy'],r['n_evaluations'],r['extraction_failures'],r['fallback_extractions'],
                      f'{r["mean_response_words"]:.1f}',f'{r["eot_text_present"]}/{r["n_evaluations"]}'] for r in output_summary if r['outcome']=='all']),'',
              'Visible response lengths use stored EOS-truncated, special-token-skipped responses. Generated token count '
              'includes the whole fixed budget. Raw special-marker **character offsets** are observed directly; '
              'they are not token offsets. Confirmed token offsets are reported only for explicit `--special-token-id NAME=ID` '
              'mappings. No tokenizer is loaded and no token identity is guessed. EoT/end-of-text marker presence '
              'alone is not evidence that a response is correct or complete.','',
              'Extraction failures and fallback use are observable. A parsed but wrong number does not, by itself, '
              'identify arithmetic errors, flawed reasoning, truncation, or formatting failure. Full visible and raw '
              'responses plus gold solutions appear in `representative_responses.md` for human review. The two lowest '
              'question IDs are selected per category; these examples are illustrative, not frequency estimates.','',
              table(['Disagreement / diagnostic category','Questions','Full examples'],[[c,len(ids),min(2,len(ids))] for c,ids in categories]),'',
              '## Influence diagnostics','',
              'Only stored `greedy_rounds` are analyzed. Scores are entropy reduction in nats across other masked '
              'positions, evaluated inside the recorded confidence top-K pool; they are not estimates of exhaustive '
              'global influence. Selected scores are checked against stored candidate maxima. Candidate distributions '
              'include min/median/max/std, negative fraction, and winner–runner-up margin (blank for singleton pools). '
              f'All relationship-to-accuracy summaries first average within question; {metadata["influence_rounds"]:,} token decisions are not '
              'treated as 38,400 independent accuracy observations.','',
              table(['Question-level metric','Outcome','n','Mean','Descriptive r with correctness'],
                    [[r['metric'],r['outcome'],r['n_questions'],f'{r["mean"]:.4f}' if r['mean'] is not None else 'NA',
                      f'{r["descriptive_correlation_with_correct"]:.3f}' if r['descriptive_correlation_with_correct'] is not None else 'NA'] for r in inf_summary]),'',
              'Raw influence magnitudes depend on how many other positions remain masked. The first-10% score '
              'provides a common early window but does not remove all selection/confounding effects. Negative scores '
              'mean increased predictive uncertainty elsewhere, not necessarily harmful reveals. Candidate confidence '
              'values are not stored for unselected candidates, so their confidence–influence relationship cannot be recovered. '
              'The all-reveals mean normalized position is exactly 0.5 by construction for every complete trajectory; '
              'its missing correlation is structural, not a finding about influence. Early-window positions carry ordering information. '
              'Whole-response means include special tokens and post-termination positions, so selected confidence is '
              'not a calibrated probability of answer correctness.','',
              '## Limitations and testable Phase 2 hypotheses','',
              'This is an exploratory, single-run analysis of 150 benchmark questions and correlated proxies. '
              'Subgroup bootstrap intervals are pointwise and can be unstable or degenerate in small/zero-discordance '
              'groups; Wilson accuracy intervals likewise do not represent seed variability. The bin analyses are not '
              'multiplicity-adjusted confirmatory tests. Overall paired tests and the nine trend tests use separate Holm '
              'families. No association here establishes that complexity or reveal position causes an accuracy difference. '
              'Accuracy uses the existing extraction and gold labels; reasoning-error taxonomies require annotation.','',
              'The following are proposed tests, not conclusions established by this analysis:','',
              '1. **Order × calculation-count interaction:** use a fresh held-out, pre-stratified question set and test '
              'LTR−RTL paired gap differences across calculation-count groups under identical prompts/budgets. '
              'Repeat random decoding seeds and pre-specify the interaction statistic; observational complexity '
              'strata alone still do not identify a causal effect of complexity.',
              '2. **Early-position intervention:** randomly assign early left-biased versus right-biased reveal constraints '
              'for confidence/influence while keeping per-step counts and later policy fixed. Test paired accuracy '
              'differences to separate an early-position intervention from the descriptive correctness-stratified curves.',
              '3. **Influence candidate-pool ablation:** compare top-8 with larger K and exhaustive scoring on a '
              'preselected held-out subset, alongside confidence/random. Measure paired accuracy, score decisions, '
              'forward passes, and GPU time to test whether the observed influence policy depends on the confidence prefilter.','',
              '## Reproduction','',
              f'`python analyze_phase15.py --input-dir "{metadata["input_dir"]}" --output-dir results/phase15`','',
              'Outputs are separate from Phase 1. `analysis_metadata.json` records source-file hashes, input hashes, '
              'analysis settings, package versions, and token-ID mappings. Only the three PNGs are generated: '
              'complexity gaps, reveal dynamics, and influence diagnostics.']
    # Add concise interpretations that are anchored in observed counts, not error labels.
    failures=sum(r['extraction_failures'] for r in output_summary if r['outcome']=='all')
    incorrect=sum(r['questions']-r['correct'] for r in audit)
    finding=(f'Only {failures}/{incorrect} incorrect evaluations ({failures}/{sum(r["evaluations"] for r in audit)} total) '
             'have extraction failures. Most recorded failures therefore contain a parsed answer that disagrees with gold; '
             'this observation does not distinguish reasoning mistakes from other output or benchmark-label problems.')
    lines.insert(lines.index('## Influence diagnostics'),finding+'\n')
    for strategy in ('confidence','influence'):
        correct=next(r for r in windows if r['strategy']==strategy and r['outcome']=='correct' and r['window_fraction']==.1)
        wrong=next(r for r in windows if r['strategy']==strategy and r['outcome']=='incorrect' and r['window_fraction']==.1)
        if correct['n_questions'] and wrong['n_questions']:
            finding=(f'For {strategy}, first-10%-step mean reveal position is '
                     f'{correct["mean_normalized_position"]:.3f} among correct responses (n={correct["n_questions"]}) and '
                     f'{wrong["mean_normalized_position"]:.3f} among incorrect responses (n={wrong["n_questions"]}). '
                     'This is a correctness-stratified association, not evidence of a causal benefit from moving reveals left.')
            lines.insert(lines.index('## Observable output diagnostics'),finding+'\n')
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',default='outputs/phase1/')
    parser.add_argument('--output-dir',default='results/phase15/')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--bootstrap-repetitions',type=int,default=10000,
                        help='Also used for exploratory permutation tests (default: 10000)')
    parser.add_argument('--special-token-id',action='append',default=[],metavar='NAME=ID',
                        help='Optional confirmed token mapping; NAME=eos enables EOS token offsets. No lookup is performed.')
    args=parser.parse_args()
    source=Path(args.input_dir).expanduser().resolve();out=Path(args.output_dir).expanduser().resolve()
    if source==out or source in out.parents or out in source.parents:
        parser.error('Input/output directories must not overlap')
    if args.bootstrap_repetitions<100:
        parser.error('Use at least 100 resampling repetitions')
    special_ids={}
    for mapping in args.special_token_id:
        name,identifier=mapping.split('=',1);special_ids[name]=int(identifier)
    records,qmap,paths,hashes,audit=load_inputs(source)
    print('Verified 750 unique evaluations: 150 questions × five strategies.',flush=True)
    lookup={(r['question_id'],r['strategy']):r for r in records}
    rng=np.random.default_rng(args.seed)
    pairs=[paired_row(a,b,sorted(qmap),lookup,rng,args.bootstrap_repetitions) for a,b in itertools.combinations(STRATEGIES,2)]
    holm(pairs)
    features,accuracy,gaps,trends=complexity(qmap,lookup,rng,args.bootstrap_repetitions)
    windows,window_summary,curves,rounds,inf_questions=trajectories(records)
    diagnostic_rows,output_summary=diagnostics(records,special_ids)
    inf_summary=influence_summary(inf_questions)
    # All validation and calculations precede output writes.
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('input_audit.csv',audit),('question_features.csv',features),
                       ('complexity_accuracy.csv',accuracy),('complexity_gaps.csv',gaps),('gap_trends.csv',trends),
                       ('paired_comparisons.csv',pairs),('trajectory_windows.csv',windows),
                       ('trajectory_window_summary.csv',window_summary),('trajectory_curves.csv',curves),
                       ('output_diagnostics.csv',diagnostic_rows),('output_summary.csv',output_summary),
                       ('influence_rounds.csv',rounds),('influence_questions.csv',inf_questions),('influence_summary.csv',inf_summary)]:
        write_csv(out,name,rows)
    categories=representatives(out,qmap,lookup)
    plots(out,gaps,curves,inf_questions)
    import matplotlib
    metadata=dict(input_dir=str(source),output_dir=str(out),input_sha256=hashes,
                  script_sha256=digest(Path(__file__)),seed=args.seed,bootstrap_repetitions=args.bootstrap_repetitions,
                  numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
                  special_token_ids=special_ids,complexity_bins={k:[(lo,None if hi==math.inf else hi,label) for lo,hi,label in v] for k,v in GROUPS.items()},
                  observed_evaluations=len(records),observed_questions=len(qmap),influence_rounds=len(rounds))
    manifest=json.loads((source/'manifest.json').read_text()) if (source/'manifest.json').exists() else {}
    metadata.update(phase1_decoding=manifest.get('config',{}).get('decoding',{}),
                    model_revision=manifest.get('model_revision','unavailable'),
                    dataset_revision=manifest.get('dataset_revision','unavailable'))
    report(out,audit,pairs,gaps,trends,window_summary,output_summary,inf_summary,categories,metadata)
    if any(digest(p)!=hashes[p.name] for p in paths):
        raise RuntimeError('Input hashes changed during analysis; investigate before interpreting outputs')
    metadata['input_hashes_unchanged_after_analysis']=True
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Analysis complete. Report: {out / "report.md"}. Phase 1 input hashes unchanged.',flush=True)


if __name__=='__main__':
    main()
