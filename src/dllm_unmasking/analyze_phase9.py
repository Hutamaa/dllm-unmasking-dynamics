"""Post-inference Phase 9 analysis, staged privately for scientific/privacy review."""
import argparse
import json
import math
import platform
from pathlib import Path
import re
from statistics import mean,median

import numpy as np
from dllm_unmasking import phase9_protocol as protocol
from dllm_unmasking.ar_baseline import digest,object_hash,read_journal
from dllm_unmasking.finalize_phase8 import paired,inventory,table
from dllm_unmasking.reasoning_budget import ByteLevelDecoder,metrics,placement
from dllm_unmasking.prepare_ltr_audit import selected_span
from dllm_unmasking.evaluation import NUMBER,extract_answer
from dllm_unmasking.phase2_data import atomic_json
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.layout import enter_repository,repository_root
from dllm_unmasking.phase9_review import make_review


def load_completed(path):
    path=Path(path).resolve();m=json.loads((path/'manifest.json').read_text())
    if object_hash({k:v for k,v in m.items() if k!='run_hash'})!=m['run_hash']:raise ValueError('Manifest hash mismatch')
    cfg=protocol.validate_config(m['config']);qs=protocol.inputs(cfg)
    if m.get('configuration_hash')!=object_hash(cfg):raise ValueError('Immutable configuration hash mismatch')
    _,refs=protocol.references(cfg);base=refs[256][1]
    for key in ('model','model_revision','tokenizer_revision','model_config','native_generation_config','tokenizer_sha256','tokenizer_config_sha256','chat_template','selection_sha256'):
        if m[key]!=base[key]:raise ValueError('Intervention changed native model/template/selection identity')
    if m['intervention'].get('phase8_provenance_sha256')!=digest(cfg['phase8_provenance']):raise ValueError('Historical Phase 8 provenance mismatch')
    if m['intervention']['instruction']!=protocol.CONCISE or m['intervention']['condition']!=protocol.CONDITION:raise ValueError('Unrecognized prompt intervention')
    for cap in (256,512):
        for name,key in [('manifest.json','reference_manifest_sha256'),('journal.jsonl','reference_journal_sha256')]:
            if m['intervention'][key][str(cap)]!=digest(Path(cfg[f'baseline{cap}_dir'])/name):raise ValueError('Reference identity changed')
    rs=read_journal(path/'journal.jsonl',qs,m['run_hash'],repair=False,model=cfg['model'],revision=cfg['model_revision'],cap=256)
    protocol.validate_records(cfg,rs)
    if len(rs)!=60:raise ValueError('Analysis requires all 60 completed evaluations; use runner progress summaries for partial runs')
    protocol.validate_environment(cfg,json.loads((path/'environment.json').read_text()))
    return cfg,qs,rs,refs,m


def first_explicit(record,view):
    full={m.span() for m in re.finditer(NUMBER,record['response'])}
    for count,text in view.prefixes():
        value,method=extract_answer(text);span=selected_span(text)
        if value is not None and method!='last_number' and span in full:
            return dict(first_explicit_prefix_tokens=count,**{'first_explicit_'+k:v for k,v in placement(record['response'],span,view).items()})
    return dict(first_explicit_prefix_tokens=None,**{'first_explicit_'+k:v for k,v in placement(record['response'],None,view).items()})


def aggregate(rows):
    result=[]
    for condition in ('original','concise'):
        for stratum in ('all','0-2','3-4','5+'):
            rs=[r for r in rows if r['condition']==condition and (stratum=='all' or r['stratum']==stratum)]
            row=dict(condition=condition,stratum=stratum,n=len(rs),correct=sum(r['correct'] for r in rs),accuracy=mean(r['correct'] for r in rs),
                eos=sum(r['termination']=='eos' for r in rs),length_limited=sum(r['termination']=='length_limit' for r in rs),
                final_markers=sum(r['final_marker_present'] for r in rs),extraction_failures=sum(r['extraction_failed'] for r in rs),
                parser_fallbacks=sum(r['extraction_method']=='last_number' for r in rs),explicit_answer_available_n=sum(r['first_explicit_prefix_tokens'] is not None for r in rs))
            for field in ('chars','words','native_generated_tokens','native_visible_tokens','first_explicit_prefix_tokens','gpu_prefill_decode_seconds','generation_wall_seconds'):
                values=[r[field] for r in rs if r[field] is not None]
                row[field+'_n']=len(values)
                for label,value in [('mean',mean(values) if values else None),('median',median(values) if values else None),('p90',float(np.quantile(values,.9,method='linear')) if values else None)]:row[field+'_'+label]=value
            result.append(row)
    return result


def paired_outcomes(rows,seed,repetitions):
    indexed={(r['condition'],r['question_id']):r for r in rows};ids=sorted(q for condition,q in indexed if condition=='original')
    if {q for c,q in indexed if c=='concise'}!=set(ids):raise ValueError('Unmatched outcomes')
    stats=[]
    for field in ('eos_indicator','marker_indicator','chars','words','native_generated_tokens','gpu_prefill_decode_seconds','generation_wall_seconds'):
        differences=np.array([float(indexed['concise',q][field])-float(indexed['original',q][field]) for q in ids])
        rng=np.random.Generator(np.random.PCG64(seed));draws=np.zeros(repetitions)
        for stratum in ('0-2','3-4','5+'):
            ds=differences[[i for i,q in enumerate(ids) if indexed['original',q]['stratum']==stratum]]
            draws+=rng.choice(ds,(repetitions,len(ds)),replace=True).sum(axis=1)
        lo,hi=np.quantile(draws/len(ids),[.025,.975],method='linear')
        wins=int((differences==1).sum());losses=int((differences==-1).sum());discordant=wins+losses
        exact=(min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.) if field in ('eos_indicator','marker_indicator') else None
        stats.append(dict(outcome=field,n=len(ids),units='probability_fraction' if field.endswith('_indicator') else field,mean_concise_minus_original=float(differences.mean()),bootstrap_low=float(lo),bootstrap_high=float(hi),mcnemar_exact_p=exact))
    return stats


def figures(out,summary,rows,references):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    overall=[r for r in summary if r['stratum']=='all']
    fig,ax=plt.subplots(figsize=(8,5))
    for r in overall:ax.scatter(r['gpu_prefill_decode_seconds_mean'],100*r['accuracy'],label='Qwen-256 '+r['condition'],s=65)
    q512=references[512][0]
    ax.scatter(mean(r['gpu_prefill_decode_seconds'] for r in q512.values()),100*mean(r['correct'] for r in q512.values()),marker='^',label='Qwen-512 original (extended-budget reference)',s=65)
    ax.set(xlabel='Mean GPU prefill + native cached decode interval (seconds/question)',ylabel='Numerical-answer accuracy (%)',ylim=(0,100));ax.legend(fontsize=8)
    fig.text(.5,.01,'Paired prompt comparison; separate jobs, no latency repetitions. Qwen-512 is reference-only.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'accuracy_latency.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,field in zip(axes,('native_generated_tokens','words','chars')):
        ax.boxplot([[r[field] for r in rows if r['condition']==c] for c in ('original','concise')],tick_labels=['Original','Concise']);ax.set(ylabel=field.replace('_',' '))
    fig.tight_layout();fig.savefig(out/'output_lengths.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([r['condition'] for r in overall],[r['eos'] for r in overall],label='EOS')
    ax.bar([r['condition'] for r in overall],[r['length_limited'] for r in overall],bottom=[r['eos'] for r in overall],label='256-token cap')
    ax.set(ylabel='Questions (n=60 per condition)');ax.legend();fig.tight_layout();fig.savefig(out/'termination.png',dpi=160);plt.close(fig)


def analyze(path,out,tokenizer_dir):
    path=Path(path).resolve();out=Path(out).resolve();root=repository_root()
    if not out.is_relative_to(root/'results/phase9') or out==root/'results/phase9':raise ValueError('Unreviewed analysis must stay in an independent ignored results/phase9 subdirectory')
    if out.exists():raise ValueError('Never overwrite analysis or review annotations; choose a fresh output directory')
    cfg,questions,rs,refs,manifest=load_completed(path)
    for source in (path,Path(cfg['baseline256_dir']).resolve(),Path(cfg['baseline512_dir']).resolve(),Path(tokenizer_dir).resolve()):
        if out==source or out in source.parents or source in out.parents:raise ValueError('Input/output overlap')
    sources=[path,Path(cfg['baseline256_dir']),Path(cfg['baseline512_dir'])];protected={str(p):inventory(p) for p in sources}
    if digest(Path(tokenizer_dir)/'tokenizer_config.json')!=manifest['tokenizer_config_sha256']:raise ValueError('Tokenizer config mismatch')
    decoder=ByteLevelDecoder(Path(tokenizer_dir)/'tokenizer.json',manifest['tokenizer_sha256']);rows=[]
    for condition,records in [('original',refs[256][0]),('concise',rs)]:
        for q in questions:
            r=records[q['question_id']];view=decoder.view(r)
            if not view.offsets_available:raise ValueError('Exact native offsets unavailable; no estimates')
            rows.append(dict(condition=condition,question_id=q['question_id'],stratum=q['stratum'],**metrics(r,view),**first_explicit(r,view),
                termination=r['termination'],eos_indicator=int(r['termination']=='eos'),marker_indicator=int('####' in r['response']),
                gpu_prefill_decode_seconds=r['gpu_prefill_decode_seconds'],generation_wall_seconds=r['generation_wall_seconds']))
    summary=aggregate(rows);strata={q['question_id']:q['stratum'] for q in questions}
    contrast=paired(refs[256][0],rs,strata,'Qwen-256 original','Qwen-256 concise',cfg['bootstrap_seed'],cfg['bootstrap_repetitions'])
    secondary=paired_outcomes(rows,cfg['bootstrap_seed'],cfg['bootstrap_repetitions'])
    transitions=[]
    for q in questions:
        qid=q['question_id'];a,b=refs[256][0][qid],rs[qid]
        category='recovery' if not a['correct'] and b['correct'] else 'regression' if a['correct'] and not b['correct'] else 'both_correct' if a['correct'] else 'both_incorrect'
        transitions.append(dict(question_id=qid,stratum=q['stratum'],category=category,original_correct=a['correct'],concise_correct=b['correct'],gold_numeric=a['gold_numeric'],original_extracted=a['predicted_numeric'],concise_extracted=b['predicted_numeric']))
    out.mkdir(parents=True)
    for name,data in [('per_question.csv',rows),('summary.csv',summary),('paired_correctness.csv',[contrast]),('paired_outcomes.csv',secondary),('recoveries_regressions.csv',transitions)]:write_csv(out/name,data)
    review=make_review(out,questions,refs[256][0],rs,cfg['review_seed']);figures(out,summary,rows,refs)
    provenance=dict(phase=9,condition=protocol.CONDITION,preselected_instruction=protocol.CONCISE,instruction_sha256=object_hash(protocol.CONCISE),
        configuration_hash=object_hash(cfg),run_hash=manifest['run_hash'],model=cfg['model'],model_revision=cfg['model_revision'],tokenizer_revision=manifest['tokenizer_revision'],selection_sha256=manifest['selection_sha256'],analysis_environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=__import__('matplotlib').__version__),input_sha256={f'{i}/{n}':digest(p/n) for i,p in enumerate(sources) for n in ('manifest.json','journal.jsonl','environment.json')},
        tokenizer_sha256=manifest['tokenizer_sha256'],review=review,bootstrap=dict(seed=cfg['bootstrap_seed'],repetitions=cfg['bootstrap_repetitions'],generator='NumPy PCG64 reset per contrast',method='Paired within-stratum resampling, 20 per stratum, linear percentile 2.5/97.5; pointwise exploratory intervals'),
        tests='One primary exact two-sided McNemar test; secondary outcomes exploratory, no multiplicity-adjusted intervals',
        source_sha256={p:digest(p) for p in ('src/dllm_unmasking/analyze_phase9.py','src/dllm_unmasking/phase9_review.py','src/dllm_unmasking/phase9_protocol.py','src/dllm_unmasking/reasoning_budget.py')},
        publication_status='UNREVIEWED: staging only; human reasoning quality not assessed')
    atomic_json(out/'analysis_manifest.json',provenance)
    display=[]
    for r in summary:
        display.append(dict(condition=r['condition'],stratum=r['stratum'],n=r['n'],correct=r['correct'],accuracy=f"{100*r['accuracy']:.1f}%",eos=r['eos'],cap=r['length_limited'],marker=r['final_markers'],tokens=f"{r['native_generated_tokens_mean']:.2f}",words=f"{r['words_mean']:.2f}",chars=f"{r['chars_mean']:.2f}",gpu=f"{r['gpu_prefill_decode_seconds_mean']:.4f}",median=f"{r['gpu_prefill_decode_seconds_median']:.4f}",p90=f"{r['gpu_prefill_decode_seconds_p90']:.4f}"))
    report=f'''# Phase 9 — Concise Reasoning Intervention (unreviewed analysis)

This prospective single-prompt intervention compares pinned Qwen2.5-7B-Instruct
on the same 60 GSM8K questions, in original order, at the same 256-token cap.
The sole preselected instruction is:

> {protocol.CONCISE}

Its original comparator is:

> {protocol.ORIGINAL}

Placement is the same user role in the identical native template. Model/tokenizer
revision is `{cfg['model_revision']}`; greedy generation, cache, BF16, attention,
seed, evaluator and the recorded GPU/software environment match the original.
No wording variants were tuned. Future variants require a separate exploratory
label and independent outputs. This is not an architecture or decoding-algorithm test.
Qwen-512 (50/60) and LLaDA LTR P4 (37/60) are historical references only.

{table(display,[('condition','Condition'),('stratum','Stratum'),('n','n'),('correct','Correct'),('accuracy','Accuracy'),('eos','EOS'),('cap','Cap'),('marker','####'),('tokens','Mean tokens'),('words','Mean words'),('chars','Mean chars'),('gpu','Mean GPU s'),('median','Median GPU s'),('p90','p90 GPU s')])}

## Paired primary outcome

Concise minus original: {contrast['b_minus_a_pp']:.2f} percentage points;
95% paired bootstrap interval [{contrast['bootstrap_low_pp']:.2f}, {contrast['bootstrap_high_pp']:.2f}].
Recoveries: {contrast['b_recovers']}/60; regressions: {contrast['b_regresses']}/60.
Exact two-sided McNemar p = {contrast['mcnemar_exact_p']:.8g}.
The one primary comparison uses matched question differences, not independent
strategy samples. Bootstrap: 10,000 within-stratum paired resamples (20 each),
NumPy PCG64 seed 42 reset per contrast, linear percentile 2.5/97.5.
Secondary intervals are exploratory and pointwise; no simultaneous coverage is
claimed. The balanced subset is not a random estimate of full GSM8K performance.

[Per-question transitions](recoveries_regressions.csv) include every improvement
and regression. [Secondary contrasts](paired_outcomes.csv) cover EOS, markers,
lengths and timing. [Per-question measurements](per_question.csv) retain first
complete explicit-answer positions in characters, whitespace words and native
output tokens. Starts are zero-based and ends exclusive. Explicit-answer availability
uses the unchanged parser on stored native prefixes, excludes last-number fallback
and partial digit matches using full-output numeric spans. This is a hindsight
diagnostic, not a new online stopping rule or a reasoning-quality judgment.

![Accuracy and recorded latency](accuracy_latency.png)
![Output lengths](output_lengths.png)
![EOS versus truncation](termination.png)

## Timing and reasoning quality limitations

GPU timing uses the original synchronized native generate interval covering
prefill plus cached decode, excluding loading, tokenization and CPU text decoding;
it includes stream scheduling gaps and is not pure kernel time. Generation wall
seconds have the original generate-only scope. Separate jobs and no latency
repetitions limit timing claims. Qwen-512 is extended-budget reference-only;
LLaDA decode-only timings are not treated as harmonized AR timings.

The [blinded packet](review_packet/comparison.html) randomizes case order and A/B
condition order using fixed seed {cfg['review_seed']} and a separate seed+1 stream.
Share only review_packet/; keep unblinding/ hidden. Question identifiers, condition
labels, automated correctness and timing/length summaries are omitted from review.
Style can reveal the intervention, so label concealment does not guarantee blinding.
Criteria cover arithmetic, essential-step completeness, logical coherence,
unsupported jumps and final-answer correctness, with evidence and unclear options.
Shorter reasoning is not automatically worse. **No human ratings have been imported;
reasoning quality has not been assessed or shown preserved.** Numerical correctness
is a separate outcome. Lock independent ratings before consulting the key.

## Reproducibility and release

analysis_manifest.json records immutable run/input/source hashes, the preselected
instruction, statistics settings and review seed. Raw journals and references
are read-only and checked unchanged on exit. This entire directory is ignored,
unreviewed staging. Only after scientific/privacy review should selected derived
CSV/PNG/report files be copied to reports/phase9/. Do not copy full responses,
review forms, unblinding keys, environments, tokenizer assets or raw journals.
Reproduce with the documented Phase 9 analyzer and a fresh output directory.
'''
    (out/'PHASE9_REPORT.md').write_text(report)
    for p,hashes in protected.items():
        if inventory(Path(p))!=hashes:raise RuntimeError('Input artifacts changed during analysis')
    return contrast,review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',default='outputs/phase9/qwen_concise_256/run')
    p.add_argument('--output-dir',default='results/phase9/analysis')
    p.add_argument('--tokenizer-dir',default='local_tokenizers/qwen')
    args=p.parse_args();contrast,review=analyze(args.input_dir,args.output_dir,args.tokenizer_dir)
    print(f'Unreviewed private analysis staged; {review["n_questions"]} blinded pairs; recoveries={contrast["b_recovers"]}, regressions={contrast["b_regresses"]}.')


if __name__=='__main__':enter_repository();main()
