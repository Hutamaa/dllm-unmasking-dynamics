"""Phase 2.5 read-only offline diagnosis; no model imports, inference, or cache access."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from statistics import mean

import numpy as np
from analyze_phase15 import bootstrap,holm
from evaluation import extract_answer,NUMBER

STRATEGIES=('random','left_to_right','right_to_left','confidence','batched_influence')
SPECIAL=re.compile(r'<\|[^<>]+\|>|\[MASK\]')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(out,name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def key(r):return r['question_id'],r['strategy'],r['parallelism']


def load(source):
    journal=source/'journal.jsonl';manifest=json.loads((source/'manifest.json').read_text())
    completed={};pending={};starts=Counter();events=Counter()
    with journal.open() as f:
        for line_number,line in enumerate(f,1):
            if not line.endswith('\n'):raise ValueError(f'Incomplete journal line {line_number}; not repaired')
            event=json.loads(line);k=key(event);events[event['event']]+=1
            if event['event']=='started':
                if k in pending or k in completed:raise ValueError('Duplicate start or restart attempt: investigate journal')
                pending[k]=event;starts[k]+=1
            elif event['event'] in ('completed','finished'):
                if k not in pending or k in completed:raise ValueError('Completion missing start or duplicate key')
                r=event['result']
                if key(r)!=k:raise ValueError('Event/result key mismatch')
                completed[k]=r;del pending[k]
            else:raise ValueError('Unrecognized event')
    ids={k[0] for k in completed};expected={(q,s,p) for q in ids for s in STRATEGIES for p in (4,8)}
    if pending or len(ids)!=60 or len(completed)!=600 or set(completed)!=expected:
        raise ValueError('Require 600 unique complete evaluations, 60 matched questions, no pending/duplicate keys')
    records=sorted(completed.values(),key=key)
    for qid in ids:
        rows=[r for r in records if r['question_id']==qid];base=rows[0]
        count=len(re.findall(r'<<.*?>>',base['gold_answer'],re.S))
        stratum='0-2' if count<=2 else '3-4' if count<=4 else '5+'
        if count!=base['calculation_count'] or stratum!=base['stratum']:raise ValueError('Gold complexity metadata mismatch')
        for r in rows:
            if any(r[k]!=base[k] for k in ('question','gold_answer','stratum','calculation_count')):
                raise ValueError('Question metadata differs across conditions')
            gold,_=extract_answer(r['gold_answer'],gold=True);predicted,method=extract_answer(r['response'])
            if (gold,predicted,method,predicted is None,predicted==gold)!=(r['gold_numeric'],r['predicted_numeric'],r['extraction_method'],r['extraction_failed'],r['correct']):
                raise ValueError('Primary extraction/correctness mismatch')
    strata=Counter(completed[(q,'random',4)]['stratum'] for q in ids)
    if strata!={'0-2':20,'3-4':20,'5+':20}:raise ValueError('Strata do not contain 20 questions each')
    with (source/'results.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
    csvmap={(int(r['question_id']),r['strategy'],int(r['parallelism'])):r for r in csv_rows}
    if len(csv_rows)!=600 or set(csvmap)!=expected:raise ValueError('CSV coverage mismatch')
    for r in records:
        c=csvmap[key(r)]
        for field in ('correct','extraction_failed','response','raw_response','predicted_numeric'):
            expected_text='' if r[field] is None else str(r[field])
            if c[field]!=expected_text:raise ValueError(f'CSV mismatch: {field}')
    return records,completed,manifest,dict(events)


def geometry(positions):
    sorted_positions=sorted(positions);gaps=np.diff(sorted_positions).tolist()
    runs=[];run=1
    for gap in gaps:
        if gap==1:run+=1
        else:runs.append(run);run=1
    runs.append(run)
    adjacent=sum(g==1 for g in gaps)
    return dict(adjacent_pairs=adjacent,adjacent_fraction=adjacent/(len(positions)-1),
                min_gap=min(gaps),mean_gap=mean(gaps),spread=sorted_positions[-1]-sorted_positions[0],
                max_contiguous_run=max(runs),mean_contiguous_run=mean(runs),n_contiguous_runs=len(runs))


def audit_trajectories(records):
    audit=[];questions=[]
    for r in records:
        p=r['parallelism'];steps=256//p;trace=r['trajectory'];tokens=r['generated_token_ids']
        if r['steps']!=steps or len(trace)!=steps or len(tokens)!=256:raise ValueError('Trajectory shape mismatch')
        seen=set();remaining=list(range(256));rng=random.Random(r['position_seed']);geometries=[]
        for i,t in enumerate(trace):
            pos=t['positions'];ts=t['token_ids'];conf=t['confidences']
            if t['step']!=i or len(pos)!=p or len(set(pos))!=p or len(ts)!=p or len(conf)!=p:
                raise ValueError('Batch schedule/length mismatch')
            if any(not 0<=x<256 or x in seen for x in pos):raise ValueError('Position overlap/out of range')
            if any(tokens[x]!=v for x,v in zip(pos,ts)):raise ValueError('Committed/final token mismatch')
            if any(not math.isfinite(c) or not 0<=c<=1 for c in conf):raise ValueError('Invalid confidence')
            s=r['strategy']
            if s=='left_to_right' and pos!=list(range(i*p,(i+1)*p)):raise ValueError('LTR grouping mismatch')
            if s=='right_to_left' and pos!=list(range(255-i*p,255-(i+1)*p,-1)):raise ValueError('RTL grouping mismatch')
            if s=='random':
                if pos!=rng.sample(remaining,p):raise ValueError('Random positions do not reproduce recorded seed')
                remaining=[x for x in remaining if x not in pos]
            if s=='confidence' and list(zip(pos,conf))!=sorted(zip(pos,conf),key=lambda v:(-v[1],v[0])):
                raise ValueError('Confidence selected ranking/ties mismatch')
            passes=1
            if s=='batched_influence':
                pool=t['candidate_pool'];cp=pool['candidate_positions'];cs=pool['candidate_scores'];cc=pool['candidate_confidences']
                if len(cp)!=min(16,256-len(seen)) or len(set(cp))!=len(cp) or len(cp)!=len(cs) or len(cp)!=len(cc):
                    raise ValueError('Influence candidate pool mismatch')
                if any(x in seen or not 0<=x<256 for x in cp) or not all(math.isfinite(x) for x in cs):
                    raise ValueError('Invalid influence candidates/scores')
                ranked=sorted(zip(cp,cs),key=lambda v:(-v[1],v[0]))
                if pos!=[x for x,_ in ranked[:p]]:raise ValueError('Influence winners/ranking mismatch')
                if t['influence_scores']!=[dict(zip(cp,cs))[x] for x in pos]:raise ValueError('Selected influence scores mismatch')
                if conf!=[dict(zip(cp,cc))[x] for x in pos]:raise ValueError('Influence selected confidence mismatch')
                passes+=len(cp)
            if t['forward_passes']!=passes:raise ValueError('Per-step forward-pass count mismatch')
            seen.update(pos);geometries.append(geometry(pos))
        if seen!=set(range(256)) or sum(t['forward_passes'] for t in trace)!=r['forward_passes']:
            raise ValueError('Trajectory uniqueness/pass count mismatch')
        questions.append(dict(question_id=r['question_id'],strategy=r['strategy'],parallelism=p,correct=r['correct'],
                              **{'mean_'+field:mean(g[field] for g in geometries) for field in geometries[0]}))
    for s in STRATEGIES:
        for p in (4,8):
            audit.append(dict(strategy=s,parallelism=p,n_evaluations=60,expected_steps=256//p,unique_positions=256,
                trajectory_checks='passed',selection_check=('exact seeded regeneration' if s=='random' else
                    'exact consecutive groups' if s in ('left_to_right','right_to_left') else
                    'selected confidence ordering only; unselected logits absent' if s=='confidence' else
                    'exact top-score winners within logged pool; global confidence prefilter unverifiable'),
                state_semantics_from_trajectory='not established'))
    return audit,questions


def max_run(sequence):
    return max((sum(1 for _ in g) for _,g in itertools.groupby(sequence)),default=0)


def repeat_fraction(words,n):
    grams=[tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))]
    return (len(grams)-len(set(grams)))/len(grams) if grams else None


def outputs(records,mask_id):
    result=[]
    for r in records:
        text=r['response'];raw=r['raw_response'];tokens=r['generated_token_ids']
        words=re.findall(r'\w+',text.lower())
        eos=[i for i,t in enumerate(tokens) if t==r['eos_token_id']]
        markers=[dict(marker=m.group(),char_offset=m.start()) for m in SPECIAL.finditer(raw)]
        tail=text.rsplit('####',1)[1] if '####' in text else None
        matches=list(re.finditer(NUMBER,tail)) if tail is not None else []
        # Surface-format flags, not correctness/reasoning annotations.
        row=dict(question_id=r['question_id'],strategy=r['strategy'],parallelism=r['parallelism'],correct=r['correct'],stratum=r['stratum'],
            response_words=len(words),response_characters=len(text),
            repeated_word_fraction=repeat_fraction(words,1),repeated_bigram_fraction=repeat_fraction(words,2),
            repeated_trigram_fraction=repeat_fraction(words,3),max_consecutive_word_run=max_run(words),
            max_identical_character_run=max((sum(1 for _ in g) for c,g in itertools.groupby(text) if not c.isspace()),default=0),generated_token_repeat_fraction=repeat_fraction(tokens,1),
            eos_present=bool(eos),first_eos_position=eos[0] if eos else None,
            first_eos_or_budget=eos[0] if eos else 256,eos_count=len(eos),
            first_eos_reveal_step_fraction=next((t['step']+1)/r['steps'] for t in r['trajectory'] if eos[0] in t['positions']) if eos else None,
            eos_positions=json.dumps(eos),eot_marker_present=any(x['marker']=='<|eot_id|>' for x in markers),
            special_marker_count=len(markers),unexpected_special_marker_count=sum(x['marker'] not in ('<|eot_id|>','<|endoftext|>') for x in markers),
            mask_token_count=tokens.count(mask_id),special_character_positions=json.dumps(markers),
            final_marker_present=tail is not None,final_marker_no_numeric=tail is not None and not matches,
            final_marker_multiple_numeric=tail is not None and len(matches)>1,
            final_marker_non_numeric_prefix=bool(matches) and bool(re.sub(r'[\s$*_{}\\]','',tail[:matches[0].start()])),
            extraction_failed=r['extraction_failed'],fallback=r['extraction_method']=='last_number',
            extraction_method=r['extraction_method'])
        result.append(row)
    return result


def aggregate(rows,fields):
    summaries=[]
    for s in STRATEGIES:
        for p in (4,8):
            for outcome in ('all','correct','incorrect'):
                group=[r for r in rows if r['strategy']==s and r['parallelism']==p and (outcome=='all' or r['correct']==(outcome=='correct'))]
                row=dict(strategy=s,parallelism=p,outcome=outcome,n_questions=len(group))
                for field in fields:
                    values=[r[field] for r in group if r[field] is not None]
                    row[field+'_n']=len(values);row[field+'_mean']=mean(values) if values else None
                summaries.append(row)
    return summaries


def paired(a,b,rng,repetitions,strata=None):
    n=len(a);differences=np.array(b,dtype=float)-np.array(a,dtype=float)
    if strata is None or not n:
        low,high=bootstrap(differences,rng,repetitions)
    else:
        draws=np.zeros(repetitions)
        for stratum in sorted(set(strata)):
            values=differences[np.array([s==stratum for s in strata])]
            draws += values[rng.integers(0,len(values),size=(repetitions,len(values)))].sum(axis=1)
        low,high=(float(x) for x in np.quantile(draws/n,[.025,.975]))
    return dict(n_matched=n,p4_mean=mean(a) if n else None,p8_mean=mean(b) if n else None,
                delta_p8_minus_p4=float(differences.mean()) if n else None,bootstrap_95_low=low,bootstrap_95_high=high)


def transitions(records,lookup,rng,repetitions):
    ids=sorted({r['question_id'] for r in records});summary=[];question_rows=[];complexity=[]
    for s in STRATEGIES:
        a=[lookup[q,s,4]['correct'] for q in ids];b=[lookup[q,s,8]['correct'] for q in ids]
        counts=Counter(zip(a,b));lost=counts[True,False];gained=counts[False,True];d=lost+gained
        p=min(1,2*sum(math.comb(d,k) for k in range(min(lost,gained)+1))/2**d) if d else 1
        summary.append(dict(strategy=s,**paired(a,b,rng,repetitions,[lookup[q,s,4]['stratum'] for q in ids]),both_correct=counts[True,True],
                            p4_correct_p8_incorrect=lost,p4_incorrect_p8_correct=gained,both_incorrect=counts[False,False],mcnemar_exact_p=p))
        for q in ids:
            r=lookup[q,s,4];r8=lookup[q,s,8]
            category='both_correct' if r['correct'] and r8['correct'] else 'p4_only_correct' if r['correct'] else 'p8_only_correct' if r8['correct'] else 'both_incorrect'
            question_rows.append(dict(question_id=q,strategy=s,stratum=r['stratum'],calculation_count=r['calculation_count'],
                question_words=len(r['question'].split()),p4_correct=r['correct'],p8_correct=r8['correct'],transition=category))
        for feature,bins in [('calculation_count',[(0,2,'0-2'),(3,4,'3-4'),(5,math.inf,'5+')]),
                             ('question_words',[(0,40,'0-40'),(41,70,'41-70'),(71,math.inf,'71+')])]:
            for lower,upper,label in bins:
                qids=[q for q in ids if lower<=(lookup[q,s,4]['calculation_count'] if feature=='calculation_count' else len(lookup[q,s,4]['question'].split()))<=upper]
                aa=[lookup[q,s,4]['correct'] for q in qids];bb=[lookup[q,s,8]['correct'] for q in qids]
                complexity.append(dict(strategy=s,proxy=feature,group=label,**paired(aa,bb,rng,repetitions),
                                       p4_correct=sum(aa),p8_correct=sum(bb),lost=sum(x and not y for x,y in zip(aa,bb)),
                                       gained=sum(y and not x for x,y in zip(aa,bb))))
    holm(summary)
    return summary,question_rows,complexity


def source_audit(manifest,root):
    rows=[]
    for name,expected in manifest['source_sha256'].items():
        path=root/name;match=path.exists() and sha(path)==expected
        rows.append(dict(check=name+' matches executed manifest',status='passed' if match else 'unverified',
                         evidence='SHA256 match' if match else 'Missing/different local source; execution semantics cannot be attributed to it'))
    source=(root/'phase2_samplers.py').read_text();tree=ast.parse(source)
    gen=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='generate')
    step_loop=next(n for n in gen.body if isinstance(n,ast.For))
    assignments=[n for n in ast.walk(step_loop) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='x' for t in n.targets)]
    checks=[('One permanent x write per step',len(assignments)==1,'Single indexed simultaneous x[0,start+positions]=selected_tokens assignment'),
            ('Baseline tokens committed', 'selected_tokens=tokens[positions]' in source,'Selected tokens taken from baseline, not counterfactual outputs'),
            ('Independent counterfactual clones','probe = x.clone()' in source,'Every candidate starts from original x; helper writes probe only'),
            ('Eligibility mask and deterministic ranking','masked_fill(revealed,-torch.inf)' in source and 'stable=True' in source,'Revealed positions excluded; ties stable'),
            ('No early EOS sampler termination','eos' not in ast.get_source_segment(source,gen).lower(),'All requested steps run; runner truncates decoded response separately'),
            ('No sequential 256-step setting','((256,256,64),(256,256,32))' in source,'Schedule rejects all other settings')]
    for name,ok,evidence in checks:rows.append(dict(check=name,status='source reviewed' if ok else 'needs review',evidence=evidence))
    runner=(root/'run_phase2.py').read_text()
    rows += [dict(check='Prompt/gold separation',status='source reviewed',evidence='Prompt contains question+fixed suffix; gold is read only for extraction/evaluation'),
             dict(check='EOS and extraction path',status='source reviewed',evidence='Runner decodes IDs before first recorded EOS, skips special tokens, calls existing extract_answer'),
             dict(check='Unrecorded logits/candidate prefilter',status='limitation',evidence='Cannot prove globally highest-confidence selection or recompute entropy scores from saved selected probabilities'),
             dict(check='Runtime state semantics',status='limitation',evidence='Matching source + records support implementation audit; trajectories alone cannot certify actual forward input states')]
    return rows


def metric_contrasts(rows,fields,rng,repetitions):
    lookup={key(r):r for r in rows};ids=sorted({r['question_id'] for r in rows});results=[]
    for s in STRATEGIES:
        for field in fields:
            matched=[q for q in ids if lookup[q,s,4][field] is not None and lookup[q,s,8][field] is not None]
            results.append(dict(strategy=s,metric=field,**paired([lookup[q,s,4][field] for q in matched],
                                                               [lookup[q,s,8][field] for q in matched],rng,repetitions,[lookup[q,s,4]['stratum'] for q in matched])))
    return results


def policy_contrasts(lookup,rng,repetitions):
    ids=sorted({k[0] for k in lookup});rows=[]
    for s in STRATEGIES:
        if s=='left_to_right':continue
        other8=[int(lookup[q,s,8]['correct']) for q in ids]
        ltr8=[int(lookup[q,'left_to_right',8]['correct']) for q in ids]
        other_change=[int(lookup[q,s,8]['correct'])-int(lookup[q,s,4]['correct']) for q in ids]
        ltr_change=[int(lookup[q,'left_to_right',8]['correct'])-int(lookup[q,'left_to_right',4]['correct']) for q in ids]
        for label,a,b in [('P8 accuracy: other minus LTR',ltr8,other8),
                          ('P8-P4 change: other minus LTR',ltr_change,other_change)]:
            rows.append(dict(comparator=s,contrast=label,**paired(a,b,rng,repetitions,[lookup[q,s,4]['stratum'] for q in ids])))
    return rows


def examples(out,lookup,transition_rows,diagnostics):
    diagnostic={key(r):r for r in diagnostics};pieces=['# Matched full-response examples','',
        f'All {sum(r["strategy"]=="left_to_right" and r["transition"]=="p4_only_correct" for r in transition_rows)} LTR P4-correct/P8-incorrect cases and all reverse cases are included. Other categories show the two lowest question IDs.',
        'These are stored outputs and numerical-extraction labels, not automatic reasoning/arithmetic annotations.','']
    tables=[]
    for s in STRATEGIES:
        for category in ('p4_only_correct','p8_only_correct','both_incorrect','both_correct'):
            ids=sorted(r['question_id'] for r in transition_rows if r['strategy']==s and r['transition']==category)
            shown=ids if s=='left_to_right' and category in ('p4_only_correct','p8_only_correct') else ids[:2]
            tables.append(dict(strategy=s,transition=category,n_questions=len(ids),n_full_examples=len(shown)))
            pieces += [f'## {s}: {category} (n={len(ids)}, showing {len(shown)})','']
            for q in shown:
                a,b=lookup[q,s,4],lookup[q,s,8]
                pieces += [f'### Question {q} — {a["stratum"]} calculations',a['question'],'','Gold solution:',a['gold_answer'],'']
                def cell(r):return r['response'].replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('|','&#124;').replace('\n','<br>')
                pieces += ['| Parallel-4: full visible response | Parallel-8: full visible response |','| --- | --- |',
                           f'| {cell(a)} | {cell(b)} |','']
                for p,r in ((4,a),(8,b)):
                    d=diagnostic[q,s,p]
                    pieces += [f'**P{p}:** correct={r["correct"]}, extracted={r["predicted_numeric"]}, method={r["extraction_method"]}; '
                               f'trigram repeat fraction={d["repeated_trigram_fraction"]}; first EOS={d["first_eos_position"]}.','',
                               f'Full raw response P{p} (special tokens retained):','',r['raw_response'],'']
    (out/'representative_responses.md').write_text('\n'.join(pieces)+'\n')
    return tables


def plots(out,geometry_summary,output_summary,complexity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['tab:blue','tab:orange']
    labels=['Random','LTR','RTL','Confidence','Influence']
    for name,summary,metrics,title in [
        ('batch_geometry.png',geometry_summary,('mean_adjacent_fraction','mean_spread','mean_max_contiguous_run'),'Selection geometry; question-level means, n=60 per condition'),
        ('output_diagnostics.png',output_summary,('repeated_trigram_fraction','max_identical_character_run','fallback'),'Observed repetition and extraction; question-level means, n=60 per condition')]:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,metric in zip(axes,metrics):
            for shift,p,color in zip((-.18,.18),(4,8),colors):
                rows=[next(r for r in summary if r['strategy']==s and r['parallelism']==p and r['outcome']=='all') for s in STRATEGIES]
                bars=ax.bar(np.arange(5)+shift,[r[metric+'_mean'] or 0 for r in rows],width=.35,color=color,label=f'P{p}')
                ax.bar_label(bars,labels=[f'n={r[metric+"_n"]}' for r in rows],fontsize=6,padding=2)
            ax.set_xticks(range(5),labels,rotation=25);ax.set_ylabel(metric.replace('_',' '));ax.legend()
        fig.suptitle(title);fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,group in zip(axes,('0-2','3-4','5+')):
        rows=[next(r for r in complexity if r['proxy']=='calculation_count' and r['group']==group and r['strategy']==s) for s in STRATEGIES]
        y=[r['delta_p8_minus_p4'] for r in rows]
        ax.errorbar(range(5),y,yerr=[[r['delta_p8_minus_p4']-r['bootstrap_95_low'] for r in rows],
                                  [r['bootstrap_95_high']-r['delta_p8_minus_p4'] for r in rows]],fmt='o',capsize=4)
        ax.set_xticks(range(5),labels,rotation=25);ax.axhline(0,color='gray');ax.set_title(f'{group} calculations (proxy), n=20')
    axes[0].set_ylabel('Accuracy change P8 − P4 (95% paired CI)')
    fig.suptitle('Matched degradation by calculation-count stratum; exploratory intervals')
    fig.tight_layout();fig.savefig(out/'complexity_degradation.png',dpi=160);plt.close(fig)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def fmt(x):return 'NA' if x is None else f'{x:.3f}'

def pp(x):return 'NA' if x is None else f'{100*x:.1f} pp'


def report(out,records,lookup,events,taudit,saudit,geom_summary,output_summary,contrasts,policy,trans,complexity,transition_rows,diagnostics,meta):
    n_wrong=sum(not r['correct'] for r in records);n_fail=sum(r['extraction_failed'] for r in records)
    lines=['# Phase 2.5: offline diagnosis of parallel unmasking collapse','',
        '## Scope and main observations','',
        f'Input: `{meta["input_dir"]}`. Validated **600 distinct completed evaluations**, 60 matched questions × five policies × two settings. '
        f'Journal events: `{events}`; no pending starts, duplicate keys, or truncated records. Complexity strata each contain 20 distinct questions.',
        'This analysis reads stored outputs only. No inference, HPC access, model downloads, modified scientific implementations, or new intervention occurred.','',
        table(['Policy','P4 correct/60','P8 correct/60','Lost / gained','P8−P4 gap [95% CI]','Exact McNemar p','Holm p'],
        [[r['strategy'],round(60*r['p4_mean']),round(60*r['p8_mean']),f'{r["p4_correct_p8_incorrect"]} / {r["p4_incorrect_p8_correct"]}',
          f'{pp(r["delta_p8_minus_p4"])} [{pp(r["bootstrap_95_low"])}, {pp(r["bootstrap_95_high"])}]',
          f'{r["mcnemar_exact_p"]:.4g}',f'{r["holm_adjusted_p"]:.4g}'] for r in trans]),'',
        '## Integrity and implementation audit','',
        'Every evaluation reveals exactly 256 unique response-relative positions, matches the final token IDs, '
        'and has 64 P4 steps of four reveals or 32 P8 steps of eight reveals. Forward counts match per-step records. '
        'LTR selects exact consecutive ascending groups; RTL selects exact consecutive descending groups. Random selection '
        'is reproduced with its stored private RNG seed. Confidence ordering/ties are checked among selected positions; '
        'unselected confidences were not saved, so global top-confidence eligibility cannot be reconstructed. Influence winners '
        'match top candidate-pool scores and stable position ties; the global top-16 confidence prefilter cannot be proven from its pool alone.','',
        table(['Check','Status','Evidence'],[[r['check'],r['status'],r['evidence']] for r in saudit]),'',
        'Source inspection is separate from trajectory validation. Matching recorded source hashes supports attributing the '
        'reviewed implementation to the run. The inspected sampler predicts once from x, probes cloned states only, and '
        'writes baseline tokens once per step; it does not sequentially recompute winners. These facts do not turn output '
        'trajectories into a runtime trace of actual model inputs. Model-wrapper internals and unrecorded logits remain outside this audit.','',
        '## Batch geometry: adjacency is not dependency','',
        'Positions are sorted within each batch to measure geometric gaps, independent of logged selection order. '
        'Adjacent pairs have gap one; adjacency fraction divides by batch-size−1. Spread is max−min position; '
        'contiguous runs are maximal gap-one sets. Batch metrics are first averaged within question, then across questions. '
        'No token-level observations are treated as independent accuracy samples.','',
        table(['Policy','Batch size','n','Adjacent fraction','Mean gap','Spread','Max run'],
              [[r['strategy'],r['parallelism'],r['n_questions'],fmt(r['mean_adjacent_fraction_mean']),fmt(r['mean_mean_gap_mean']),
                fmt(r['mean_spread_mean']),fmt(r['mean_max_contiguous_run_mean'])] for r in geom_summary if r['outcome']=='all']),'',
        'LTR and RTL have the same within-batch geometry at each parallelism: all positions in each batch are adjacent. '
        'The difference between LTR P8 and RTL P8 therefore cannot be attributed to adjacency count alone. '
        'Geometry measures selected positions, not actual token-dependency strength. Direction changes which committed '
        'context is available; parallelism also halves diffusion steps (64→32). Those changes are not disentangled by this experiment.','',
        '## Output degeneration and formatting','',
        'Measurements are continuous descriptive features, not a validated binary degeneration classifier. '
        'Words are lowercase Unicode word-regex tokens (including digits); repeated n-gram fraction is '
        '(number of n-grams−number of distinct n-grams)/number of n-grams, blank when there are too few tokens. '
        'Consecutive identical word runs use those normalized words. Character runs count literal consecutive '
        'identical non-whitespace characters; whitespace runs are excluded. Longer outputs naturally have more '
        'opportunities for repetition, so length and repetition must be read together. Each CSV mean records its valid denominator; '
        'trigram n in the table excludes responses shorter than three regex words (not treated as zero repetition).','',
        table(['Policy','P','n','Mean words','Trigram n','Repeat trigram fraction','Max word run','Max char run','Fallback / n','Failures / n','EOS present / n'],
            [[r['strategy'],r['parallelism'],r['n_questions'],fmt(r['response_words_mean']),r['repeated_trigram_fraction_n'],fmt(r['repeated_trigram_fraction_mean']),
              fmt(r['max_consecutive_word_run_mean']),fmt(r['max_identical_character_run_mean']),
              f'{round(r["fallback_mean"]*r["n_questions"])}/{r["n_questions"]}',
              f'{round(r["extraction_failed_mean"]*r["n_questions"])}/{r["n_questions"]}',
              f'{round(r["eos_present_mean"]*r["n_questions"])}/{r["n_questions"]}'] for r in output_summary if r['outcome']=='all']),'',
        'EOS is identified by each result\'s stored `eos_token_id` in the full generated IDs. First-EOS position is '
        'blank if absent; `first_eos_or_budget` uses 256 for absence and is labeled as a combined termination/length '
        'measure, not a discovered EOS. First-EOS reveal-step fraction separately records when that final EOS position '
        'was committed in the trajectory; it is not the same as EOS position in the text. EoT and other special-token locations use observable raw character offsets, '
        'not assumed tokenizer mappings. Unexpected special markers are markers other than EoT/end-of-text; '
        'their counts are observable, not necessarily model errors. MASK-token counts use the saved config ID. '
        'Visible responses already exclude the first EOS and later tokens; no retokenization is performed.','',
        'Final-marker flags record absence, no parseable number after the last ####, multiple numeric mentions, '
        'or a nonnumeric prefix before its first number. They are mechanical surface indicators, not a universal '
        'definition of malformed output. Missing #### can still allow valid boxed/phrase/fallback extraction. '
        'The unchanged primary extractor accepts a last-number fallback; a parsed answer does not guarantee a '
        'well-formed final answer or coherent reasoning.','',
        f'Across all 600 evaluations, {n_fail}/{n_wrong} incorrect outputs have extraction failures. '
        'The remainder have extracted numeric answers that disagree with gold; this does not automatically identify arithmetic or reasoning errors.','']
    for metric in ('repeated_trigram_fraction','max_consecutive_word_run','max_identical_character_run','fallback','eos_present'):
        row=next(r for r in contrasts if r['strategy']=='left_to_right' and r['metric']==metric)
        lines += [f'LTR `{metric}`: P4 mean {fmt(row["p4_mean"])}, P8 mean {fmt(row["p8_mean"])}; '
                  f'matched change {fmt(row["delta_p8_minus_p4"])} [{fmt(row["bootstrap_95_low"])}, {fmt(row["bootstrap_95_high"])}], n={row["n_matched"]}.','']
    losses=[r['question_id'] for r in transition_rows if r['strategy']=='left_to_right' and r['transition']=='p4_only_correct']
    diag={key(r):r for r in diagnostics}
    more_repeat=sum((diag[q,'left_to_right',8]['repeated_trigram_fraction'] or 0)>(diag[q,'left_to_right',4]['repeated_trigram_fraction'] or 0) for q in losses)
    fallback=sum(diag[q,'left_to_right',8]['fallback'] for q in losses)
    lines += [f'Among the **{len(losses)} LTR lost-correctness cases**, {more_repeat}/{len(losses)} show a higher '
              f'trigram-repeat fraction at P8, and {fallback}/{len(losses)} P8 outputs use last-number fallback. '
              'These are comparisons of visible text, not diagnoses of the underlying inference mechanism.','',
              '## Matched transitions and complexity','',
              '`matched_transitions.csv` has all 300 question–policy transitions; `representative_responses.md` '
              'contains side-by-side full visible outputs and full raw outputs for every LTR loss/reverse case, '
              'plus deterministic examples of other categories. No wrong-number case is labeled a reasoning failure.','',
              'Paired accuracy differences use question-level correctness differences (P8−P4). Percentile bootstrap '
              f'intervals use {meta["repetitions"]:,} question resamples with seed {meta["seed"]}; overall, output-metric and '
              'cross-policy intervals preserve the observed calculation-stratum counts and keep paired outcomes together. '
              'Subgroup intervals resample within the displayed group. Overall '
              'exact two-sided McNemar tests use discordant questions, with Holm correction across five strategies. '
              'Intervals and subgroup analyses are exploratory, pointwise, and can be degenerate in small groups. '
              'They describe question-sampling uncertainty, not variation across decoding seeds.','']
    for proxy in ('calculation_count','question_words'):
        lines += [f'### {proxy}','',table(['Policy','Group','n','P4 correct','P8 correct','Lost / gained','P8−P4 [95% CI]'],
            [[r['strategy'],r['group'],r['n_matched'],r['p4_correct'],r['p8_correct'],f'{r["lost"]} / {r["gained"]}',
              f'{pp(r["delta_p8_minus_p4"])} [{pp(r["bootstrap_95_low"])}, {pp(r["bootstrap_95_high"])}]'] for r in complexity if r['proxy']==proxy]),'']
    lines += ['Calculation count counts gold `<<...>>` annotations and is a **proxy, not validated reasoning depth**. '
              'Question length is whitespace word count, with fixed bins 0–40/41–70/71+. Neither proxy is a '
              'randomized intervention; correlated language, difficulty, truncation and formatting can confound associations. '
              'Differences in subgroup significance are not a test of gap heterogeneity.','',
              '## Competing hypotheses','',
              table(['Explanation','Assessment from these stored results','What would distinguish it'],[
                ['Local dependency interference','Compatible with larger-batch collapse and visible repetition; actual conditional token dependencies are unobserved. Same adjacency for LTR/RTL prevents treating adjacency alone as sufficient.','After bug checklist, compare matched contiguous versus noncontiguous batches while controlling reveal direction, scheduled progress and forward budget; measure conditional dependency explicitly.'],
                ['Directionality/context effects','Descriptively supported as a policy asymmetry: LTR P8 and RTL P8 differ despite identical batch geometry. Direction also changes early versus late context and termination tokens, so mechanism remains uncertain.','A pre-specified factorial intervention on reveal direction and geometry with matched questions/budgets and repeated random seeds.'],
                ['Termination/formatting artifacts','Repetition, missing final markers, fallback use, and EOS positions are observable; they can affect extracted accuracy. They do not explain how generation collapsed, nor establish that fixing extraction would restore reasoning.','First annotate existing full visible/raw outputs blindly; separate complete-answer formatting, termination, repetition and numerical correctness before considering a controlled termination-only intervention.']]),'',
              'No new decoding algorithm is established as a solution. The two settings change both batch size and number '
              'of model updates; only one random seed and one stratified subset were tested. Treat the three explanations '
              'as competing and potentially interacting, not mutually exclusive causal findings.','',
              '## Implementation-bug checklist before any GPU experiment','',
              '- [x] Unique paired keys, no incomplete starts, complete 600-evaluation coverage and CSV consistency.',
              '- [x] Identical question/gold metadata, valid 20/20/20 strata and regenerated primary correctness.',
              '- [x] Exactly 256 unique reveals, correct step/batch counts, final-ID consistency and pass counts.',
              '- [x] LTR/RTL exact grouping, seeded random replay, selected confidence order and influence winner order.',
              '- [x] Source hashes compared to executed manifest; baseline token commitment and independent probes inspected.',
              '- [x] Primary extraction, MASK suppression, EOS handling and prompt/gold separation inspected.',
              '- [ ] Obtain baseline/unselected logits or instrumented state provenance if runtime top-K/state semantics need independent certification; logs here cannot supply them.',
              '- [ ] Blind human review of the 33 LTR loss cases and reverse case for observable text/formatting/termination errors before naming reasoning mechanisms.',
              '- [ ] Pre-register any follow-up contrasts; account for step-budget and direction/geometry confounding, uncertainty and multiple comparisons.','',
              'The completed structural/source checks did not reveal a schedule, overlap, ranking, or greedy-commitment bug '
              'in the inspected matching implementation. This is a bounded audit, not proof that all bugs or model-wrapper '
              'issues are absent. Repetition and fallback behavior are observed output issues, not automatic evidence of a sampler bug.','',
              '## Reproduction and files','',
              '`python analyze_phase25.py --input-dir ~/Documents/dllm-unmasking-results/phase2 --output-dir results/phase25`','',
              'The analyzer does not invoke inference or journal recovery. Inputs and existing source hashes are checked '
              'before/after execution. CSVs contain integrity/source audit, question-level geometry and output measurements, '
              'question-level matched changes, overall transitions, and subgroup accuracy contrasts. Three PNGs visualize '
              'geometry, repetition/extraction and stratified degradation. `analysis_metadata.json` records input/source '
              'hashes, resampling settings and package versions. Original artifacts remain unchanged.']
    l4=next(r for r in output_summary if r['strategy']=='left_to_right' and r['parallelism']==4 and r['outcome']=='all')
    l8=next(r for r in output_summary if r['strategy']=='left_to_right' and r['parallelism']==8 and r['outcome']=='all')
    finding=(f'LTR final-marker presence falls from {round(l4["final_marker_present_mean"]*60)}/60 to '
             f'{round(l8["final_marker_present_mean"]*60)}/60. EOS is present in {sum(d["eos_present"] for d in diagnostics)} of 600 outputs; '
             f'LTR mean first-EOS index is {l4["first_eos_or_budget_mean"]:.1f} at P4 and '
             f'{l8["first_eos_or_budget_mean"]:.1f} at P8. Thus missing EOS or systematically earlier LTR '
             'termination is not the obvious descriptive explanation. Formatting/fallback often co-occurs '
             'with visibly degraded text, so an extraction-only explanation is not established. RTL P8 has '
             'an even larger mean literal character run than LTR P8 despite its higher accuracy; repetition '
             'metrics are not interchangeable or sufficient explanations on their own.')
    lines.insert(lines.index('## Matched transitions and complexity'),finding+'\n')
    cross=['### Matched cross-policy contrasts','',
           'Each contrast resamples all outcomes for a question together. Change contrasts compare '
           '(other P8−P4)−(LTR P8−P4), avoiding independence assumptions across policy/settings. '
           'These are exploratory pointwise intervals, not mechanistic evidence.','',
           table(['Comparator','Contrast','n','Gap [95% paired CI]'],
                 [[r['comparator'],r['contrast'],r['n_matched'],
                   f'{pp(r["delta_p8_minus_p4"])} [{pp(r["bootstrap_95_low"])}, {pp(r["bootstrap_95_high"])}]'] for r in policy]),'']
    lines[lines.index('## Competing hypotheses'):lines.index('## Competing hypotheses')]=cross
    note=[]
    for r in trans:
        if r['strategy']=='confidence':
            note.append(f'Confidence has similar aggregate accuracy but changes outcomes on '
                        f'{r["p4_correct_p8_incorrect"]+r["p4_incorrect_p8_correct"]}/60 questions: '
                        f'{r["p4_correct_p8_incorrect"]} losses and {r["p4_incorrect_p8_correct"]} gains. '
                        'Aggregate stability does not mean identical successful questions.')
    strata=[r for r in complexity if r['strategy']=='left_to_right' and r['proxy']=='calculation_count']
    note.append('LTR calculation-count contrasts: '+ '; '.join(
        f'{r["group"]}: {r["p4_correct"]}/{r["n_matched"]} correct at P4 versus {r["p8_correct"]}/{r["n_matched"]} at P8' for r in strata)+
        '. The largest observed drop is in the middle stratum; this is not a monotone increase of collapse with calculation count.')
    where=lines.index('### Matched cross-policy contrasts')
    lines[where:where]=note+['']
    # Bug checklist precedes the table describing possible future controlled experiments.
    start=lines.index('## Implementation-bug checklist before any GPU experiment')
    end=lines.index('## Reproduction and files')
    checklist=lines[start:end];del lines[start:end]
    where=lines.index('## Competing hypotheses');lines[where:where]=checklist
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',default='~/Documents/dllm-unmasking-results/phase2')
    parser.add_argument('--output-dir',default='results/phase25')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--repetitions',type=int,default=10000)
    args=parser.parse_args();source=Path(args.input_dir).expanduser().resolve();out=Path(args.output_dir).expanduser().resolve()
    if source==out or source in out.parents or out in source.parents:parser.error('Input/output paths must not overlap')
    if args.repetitions<100:parser.error('At least 100 bootstrap draws required')
    root=Path(__file__).resolve().parent
    protected=[p for p in source.iterdir() if p.is_file()]
    protected += [root/n for n in ('phase2_samplers.py','phase2_data.py','run_phase2.py','evaluation.py','samplers.py','analyze_phase15.py')]
    hashes={str(p):sha(p) for p in protected}
    records,lookup,manifest,events=load(source)
    taudit,geom=audit_trajectories(records);saudit=source_audit(manifest,root)
    diagnostics=outputs(records,manifest['config']['mask_id'])
    geom_fields=[k for k in geom[0] if k.startswith('mean_')]
    output_fields=('response_words','response_characters','repeated_word_fraction','repeated_bigram_fraction',
        'repeated_trigram_fraction','max_consecutive_word_run','max_identical_character_run','eos_present',
        'first_eos_or_budget','first_eos_reveal_step_fraction','eos_count','eot_marker_present','unexpected_special_marker_count','mask_token_count',
        'final_marker_present','final_marker_no_numeric','final_marker_multiple_numeric','final_marker_non_numeric_prefix',
        'extraction_failed','fallback')
    gs=aggregate(geom,geom_fields);osummary=aggregate(diagnostics,output_fields)
    rng=np.random.default_rng(args.seed)
    trans,transition_rows,complexity=transitions(records,lookup,rng,args.repetitions)
    contrasts=metric_contrasts(diagnostics,output_fields,rng,args.repetitions)
    policy=policy_contrasts(lookup,rng,args.repetitions)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('integrity_audit.csv',taudit),('implementation_audit.csv',saudit),
                      ('batch_geometry_questions.csv',geom),('batch_geometry_summary.csv',gs),
                      ('output_diagnostics.csv',diagnostics),('output_summary.csv',osummary),
                      ('matched_output_changes.csv',contrasts),('policy_degradation_contrasts.csv',policy),('transition_summary.csv',trans),
                      ('matched_transitions.csv',transition_rows),('complexity_degradation.csv',complexity)]:
        write_csv(out,name,rows)
    example_counts=examples(out,lookup,transition_rows,diagnostics);write_csv(out,'example_categories.csv',example_counts)
    plots(out,gs,osummary,complexity)
    import matplotlib
    meta=dict(input_dir=str(source),seed=args.seed,repetitions=args.repetitions,input_and_source_sha256=hashes,
              script_sha256=sha(Path(__file__)),numpy=np.__version__,matplotlib=matplotlib.__version__,
              events=events,completed_evaluations=600,n_questions=60,bootstrap_method='question-level; overall/metric/policy contrasts preserve calculation-stratum counts; subgroup contrasts resample displayed group',source_manifest_matches=all(r['status']=='passed' for r in saudit[:len(manifest['source_sha256'])]))
    report(out,records,lookup,events,taudit,saudit,gs,osummary,contrasts,policy,trans,complexity,transition_rows,diagnostics,meta)
    if any(sha(Path(p))!=digest for p,digest in hashes.items()):raise RuntimeError('Protected input/source changed during analysis')
    meta['protected_files_unchanged']=True
    (out/'analysis_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'PASS: 600 complete evaluations audited; 60 matched questions; protected hashes unchanged. Report: {out/"report.md"}')


if __name__=='__main__':main()
