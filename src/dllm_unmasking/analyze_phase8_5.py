"""Phase 8.5: strictly offline, matched reasoning-budget utilization observations."""
import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import platform
import re
from statistics import mean, median

import numpy as np
from dllm_unmasking.ar_baseline import digest
from dllm_unmasking.ar_compare import load_run
from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.finalize_phase8 import inventory, paired, table, verify_events
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.layout import enter_repository, repository_root
from dllm_unmasking.phase2_data import DATASET_REVISION, atomic_json
from dllm_unmasking.prepare_ltr_audit import candidates, highlight, selected_span
from dllm_unmasking.reasoning_budget import ByteLevelDecoder, metrics, error_flags, paired_continuous

METHODS=('LLaDA LTR P4','Qwen-256','Qwen-512')


def load_inputs(phase2,qwen256,qwen512,qwen_tokenizer,llada_tokenizer):
    p8=json.loads(Path('reports/phase8/provenance.json').read_text())
    protected={};records={};manifests={}
    for cap,path in ((256,qwen256),(512,qwen512)):
        path=Path(path).resolve();protected[str(path)]=inventory(path)
        expected=p8['imports'][str(cap)]
        for name,sha in expected.items():
            if digest(path/name)!=sha:raise ValueError('Phase 8 raw artifact checksum mismatch')
        cfg,questions,rs,manifest=load_run(path)
        verify_events(path/'journal.jsonl')
        if cfg['max_new_tokens']!=cap or set(rs)!={q['question_id'] for q in questions}:raise ValueError('Incomplete or mixed budget run')
        records[f'Qwen-{cap}']=rs;manifests[f'Qwen-{cap}']=manifest
    if manifests['Qwen-256']['model_revision']!=p8['model_revision'] or manifests['Qwen-512']['model_revision']!=p8['model_revision']:
        raise ValueError('Model revision mismatch')
    ca,cb=manifests['Qwen-256']['config'],manifests['Qwen-512']['config']
    if {k for k in ca if ca[k]!=cb[k]}!={'max_new_tokens','output_dir'}:raise ValueError('Qwen protocol mismatch')
    for name in ('tokenizer.json','tokenizer_config.json'):
        key='tokenizer_sha256' if name=='tokenizer.json' else 'tokenizer_config_sha256'
        for m in manifests.values():
            if digest(Path(qwen_tokenizer)/name)!=m[key]:raise ValueError('Pinned Qwen tokenizer checksum mismatch; no estimated offsets')
    qdata=json.loads((Path(qwen_tokenizer)/'tokenizer.json').read_text())
    ancillary=Path(qwen_tokenizer)/'vocab.json'
    if ancillary.exists() and json.loads(ancillary.read_text())!=qdata['model']['vocab']:
        raise ValueError('Qwen vocab.json differs from pinned tokenizer JSON')
    ancillary=Path(qwen_tokenizer)/'merges.txt'
    if ancillary.exists():
        merges=[x for x in ancillary.read_text().splitlines() if not x.startswith('#version:')]
        expected=qdata['model']['merges']
        if merges!=[' '.join(x) if isinstance(x,list) else x for x in expected]:
            raise ValueError('Qwen merges.txt differs from pinned tokenizer JSON')
    qconfig=json.loads((Path(qwen_tokenizer)/'tokenizer_config.json').read_text())
    if qconfig['chat_template']!=manifests['Qwen-256']['chat_template']:raise ValueError('Native Qwen template mismatch')
    phase2=Path(phase2).resolve();protected[str(phase2)]=inventory(phase2)
    reference=next(r for r in p8['diffusion_integrity'] if r['check']=='left_to_right P4')
    if digest(phase2/'journal.jsonl')!=reference['journal_sha256'] or digest(phase2/'manifest.json')!=reference['manifest_sha256']:
        raise ValueError('Original LLaDA artifact checksum mismatch')
    manifest=json.loads((phase2/'manifest.json').read_text());manifests[METHODS[0]]=manifest;c=manifest['config']
    if (c['model_revision'],c['dataset_revision'],c['prompt_suffix'],c['inference_seed'],c['gen_length'],c['block_length'])!=(
        '852152c60bdfed4632635bfcf3e83408ad3f65bf',DATASET_REVISION,ca['prompt_suffix'],ca['inference_seed'],256,256):raise ValueError('LLaDA protocol mismatch')
    if manifest['selection_sha256']!=p8['selection_sha256']:raise ValueError('Selection mismatch')
    qmap={q['question_id']:q for q in questions};rs={};pending=set()
    for line in (phase2/'journal.jsonl').open():
        e=json.loads(line)
        if e.get('strategy')!='left_to_right' or e.get('parallelism')!=4:continue
        qid=e['question_id']
        if e['event']=='started' and qid not in rs:pending.add(qid);continue
        if e['event']!='completed' or qid not in pending or qid in rs:raise ValueError('Invalid matched LLaDA journal event')
        r=e['result'];q=qmap[qid];value,mode=extract_answer(r['response']);gold=extract_answer(q['gold_answer'],True)[0]
        if (r['question'],r['gold_answer'],r['stratum'],r['calculation_count'])!=(q['question'],q['gold_answer'],q['stratum'],q['calculation_count']):raise ValueError('Question content mismatch')
        if (r['gold_numeric'],r['predicted_numeric'],r['correct'],r['extraction_method'],r['extraction_failed'])!=(gold,value,value==gold,mode,value is None):raise ValueError('Saved labels mismatch')
        if len(r['generated_token_ids'])!=256 or r['forward_passes']!=64:raise ValueError('LLaDA decode settings mismatch')
        rs[qid]=r;pending.remove(qid)
    if pending or set(rs)!=set(qmap):raise ValueError('Incomplete matched LLaDA set')
    records[METHODS[0]]=rs
    for path in (qwen_tokenizer,llada_tokenizer):protected[str(Path(path).resolve())]=inventory(Path(path).resolve())
    qdecoder=ByteLevelDecoder(Path(qwen_tokenizer)/'tokenizer.json',manifests['Qwen-256']['tokenizer_sha256']);ldecoder=ByteLevelDecoder(Path(llada_tokenizer)/'tokenizer.json')
    views={}
    for name in METHODS:
        for qid,r in records[name].items():views[name,qid]=(ldecoder if name==METHODS[0] else qdecoder).view(r)
    for qid in qmap:
        x,y=records['Qwen-256'][qid],records['Qwen-512'][qid]
        if y['generated_token_ids'][:len(x['generated_token_ids'])]!=x['generated_token_ids'] or not y['response'].startswith(x['response']):raise ValueError('Qwen exact prefix relation failed')
        if x['rendered_prompt']!=y['rendered_prompt']:raise ValueError('Prompt mismatch across budgets')
    return records,views,qmap,manifests,protected


def summarize_rows(rows):
    output=[]
    for name in METHODS:
        for subset in ('all','numerically_correct','numerically_incorrect'):
            rs=[r for r in rows if r['method']==name and (subset=='all' or r['correct']==(subset=='numerically_correct'))]
            row=dict(method=name,subset=subset,n=len(rs),correct=sum(r['correct'] for r in rs))
            for field in ('chars','words','native_visible_tokens','nonempty_lines','numbered_step_lines','sentence_units','arithmetic_symbol_lines','max_repeated_word_run',
                          'last_marker_char_start','last_marker_word_start','last_marker_native_start','selected_answer_char_end','selected_answer_word_end','selected_answer_native_end'):
                values=[r[field] for r in rs if r[field] is not None]
                row[field+'_n']=len(values);row[field+'_mean']=mean(values) if values else None;row[field+'_median']=median(values) if values else None
            output.append(row)
    return output


def manual_packet(out,records,questions,transitions,rows):
    """Full text stays in ignored results, never in the public report directory."""
    out.mkdir(parents=True,exist_ok=True);lookup={(r['method'],r['question_id']):r for r in rows}
    articles=[];mentions=[];annotation=[]
    for qid in sorted(questions):
        q=questions[qid];panels=[]
        for name in METHODS:
            r=records[name][qid];text=r['response'];span=selected_span(text);m=lookup[name,qid]
            for c in candidates(text):mentions.append(dict(method=name,question_id=qid,**c,parser_selected=(c['start'],c['end'])==span,matches_gold=c['normalized']==r['gold_numeric']))
            annotation.append(dict(question_id=qid,method=name,correct=r['correct'],intended_final_answer_human='',parser_mistake_human='',arithmetic_error_human='',reasoning_quality_human='',format_completion_human='',verbosity_human='',evidence_human=''))
            panels.append(f'<section><h3>{html.escape(name)}</h3><p>Extracted {html.escape(str(r["predicted_numeric"]))}; gold {html.escape(r["gold_numeric"])}; numerical correctness {r["correct"]}; parser {html.escape(r["extraction_method"])}.</p><p>{m["chars"]} chars; {m["words"]} whitespace words; {m["native_visible_tokens"]} native visible tokens. Last marker native start: {m["last_marker_native_start"]}; selected answer native end: {m["selected_answer_native_end"]}.</p><pre>{highlight(text,span)}</pre><details><summary>Raw response, including special/post-EOS tokens</summary><pre>{html.escape(r["raw_response"])}</pre></details></section>')
        articles.append(f'<article id="q{qid}"><h2>Question {qid} — {html.escape(q["stratum"])}</h2><p>{html.escape(q["question"])}</p><details><summary>Gold reference</summary><pre>{html.escape(q["gold_answer"])}</pre></details><div class="triple">'+''.join(panels)+'</div></article>')
    style='body{font-family:system-ui;margin:24px}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}section{min-width:0;padding:12px;border:1px solid #ccc}pre{white-space:pre-wrap;overflow-wrap:anywhere}.number{background:#fff2a8}.selected{background:#aee8ff}article{border-top:3px solid #333;margin-top:30px}@media(max-width:1000px){.triple{grid-template-columns:1fr}}'
    nav=' · '.join(f'<a href="#q{q}">{q}</a>' for q in sorted(questions))
    (out/'comparison.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 8.5 matched audit</title><style>'+style+'</style><h1>Phase 8.5: all 60 matched outputs</h1><p>Yellow: all numerical mentions. Blue: parser-selected span. Native starts are zero-based and ends exclusive. Numerical correctness is not a reasoning-quality label. Compare Qwen-512 continuations; do not automatically call a missing or malformed answer a parser bug.</p><nav>'+nav+'</nav>'+''.join(articles)+'</html>')
    write_csv(out/'numerical_candidates.csv',mentions);write_csv(out/'annotations.csv',annotation)
    write_csv(out/'transition_index.csv',transitions)
    (out/'README.md').write_text('''# Manual audit packet

Open comparison.html. It contains all 60 questions with full three-way responses,
all numerical mentions, extracted answers, token placements and expandable raw
responses. Work on copies of annotations.csv. Fields are intentionally blank.
A parser mistake needs evidence of a clearly intended final answer incorrectly
selected or parsed; an intermediate gold-valued number is insufficient.
Arithmetic errors require a quoted false calculation and a checked correction.
Reasoning quality, format completion, and verbosity need human assessment even
for numerically correct solutions. Use evidence spans and yes/no/uncertain labels.
Categories can overlap. Do not replace official correctness with manual judgments
without defining a separate protocol. Raw/post-EOS text does not change the
original evaluation. This packet is ignored; do not publish full artifact trees.
''')


def plots(out,rows,errors,transitions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,5))
    for ax,field,label in zip(axes,('chars','words','native_visible_tokens'),('Characters','Whitespace words','Native visible tokens (different vocabularies)')):
        ax.boxplot([[r[field] for r in rows if r['method']==m] for m in METHODS],tick_labels=['LTR P4','Qwen-256','Qwen-512'],showfliers=True)
        ax.set(ylabel=label);ax.tick_params(axis='x',rotation=20)
    fig.text(.5,.015,'60 matched outputs per method. Length is not verified reasoning content; capped outputs are censored.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'response_lengths.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,field,label in zip(axes,('last_marker_word_start','last_marker_native_start'),('Marker start: whitespace-word index','Marker start: native-token index')):
        for i,m in enumerate(METHODS):
            values=[r[field] for r in rows if r['method']==m and r[field] is not None]
            ax.scatter([i]*len(values),values,alpha=.45,s=18)
            if values:ax.scatter(i,median(values),marker='_',s=250,color='black')
        ax.set(xticks=range(3),xticklabels=['LTR P4','Qwen-256','Qwen-512'],ylabel=label)
    fig.text(.5,.015,'Missing markers omitted (denominators in tables). Black bars: medians. Token axes use each native vocabulary.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'answer_placement.png',dpi=160);plt.close(fig)
    recoveries=[t for t in transitions if t['transition']=='recovery']
    values=sorted(t['first_complete_explicit_parser_gold_native_512'] for t in recoveries if t['first_complete_explicit_parser_gold_native_512'] is not None)
    fig,ax=plt.subplots(figsize=(9,5))
    if values:ax.step(values,np.arange(1,len(values)+1),where='post',label=f'Explicit parser-correct complete answer available ({len(values)}/{len(recoveries)} recoveries)')
    ax.axvline(256,color='tab:red',linestyle='--',label='Original 256-token cap')
    ax.set(xlabel='Qwen-512 native prefix length (tokens)',ylabel='Cumulative recovered questions',ylim=(0,len(recoveries)+1));ax.legend(fontsize=8)
    fig.text(.5,.015,'Stored prefixes only; complete numeric spans use full-output reference. Availability does not establish reasoning quality.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'recovery_answer_budget.png',dpi=160);plt.close(fig)


def run(args):
    root=repository_root();out=Path(args.output_dir).resolve();private=Path(args.case_dir).resolve()
    if out!=root/'reports/phase8_5' and not out.is_relative_to(root/'results'):raise ValueError('Output must be reports/phase8_5 or a new results directory')
    if not private.is_relative_to(root/'results/phase8_5'):raise ValueError('Full-text cases must remain under ignored results/phase8_5')
    if out==private or out in private.parents or private in out.parents:raise ValueError('Public and private output overlap')
    for source in (args.phase2,args.qwen256,args.qwen512,args.qwen_tokenizer,args.llada_tokenizer,'reports/phase7','reports/phase8'):
        source=Path(source).resolve()
        if any(target==source or target in source.parents or source in target.parents for target in (out,private)):
            raise ValueError('Analysis output overlaps a protected input')
    old={str(p):digest(p) for base in ('reports/phase7','reports/phase8') for p in Path(base).rglob('*') if p.is_file()}
    records,views,questions,manifests,protected=load_inputs(args.phase2,args.qwen256,args.qwen512,args.qwen_tokenizer,args.llada_tokenizer)
    rows=[]
    for name in METHODS:
        for qid in sorted(questions):
            r=records[name][qid];m=metrics(r,views[name,qid])
            rows.append(dict(method=name,question_id=qid,stratum=questions[qid]['stratum'],gold_numeric=r['gold_numeric'],**m))
    lookup={(r['method'],r['question_id']):r for r in rows}
    errors=[dict(question_id=qid,stratum=questions[qid]['stratum'],**error_flags(r)) for qid,r in sorted(records['Qwen-256'].items())]
    error_summary=[]
    for key in errors[0]:
        if key in ('question_id','stratum'):continue
        n=sum(r[key] for r in errors)
        error_summary.append(dict(flag=key,n=n,all_questions=60,qwen256_errors=sum(not r['correct'] for r in records['Qwen-256'].values()),nonexclusive=True))
    transitions=[]
    for qid in sorted(questions):
        x,y,z=(lookup[name,qid] for name in ('Qwen-256','Qwen-512','LLaDA LTR P4'))
        transition='recovery' if not x['correct'] and y['correct'] else 'regression' if x['correct'] and not y['correct'] else 'both_correct' if x['correct'] else 'both_incorrect'
        transitions.append(dict(question_id=qid,stratum=x['stratum'],transition=transition,llada_correct=z['correct'],qwen256_correct=x['correct'],qwen512_correct=y['correct'],
            exact_token_prefix=True,exact_character_prefix=True,cap_limited_256=records['Qwen-256'][qid]['termination']=='length_limit',
            marker_256=x['final_marker_present'],marker_512=y['final_marker_present'],
            first_explicit_gold_char_end_512=y['first_explicit_gold_char_end'],first_explicit_gold_word_end_512=y['first_explicit_gold_word_end'],first_explicit_gold_native_end_512=y['first_explicit_gold_native_end'],
            first_complete_explicit_parser_gold_native_512=y['first_complete_explicit_parser_gold_native'],first_complete_parser_gold_native_512=y['first_complete_parser_gold_native'],
            stable_parser_gold_from_native_512=y['stable_parser_gold_from_native'],
            gold_mentioned_256=x['any_gold_mention'],explicit_gold_256=x['explicit_gold_mention']))
    summaries=summarize_rows(rows)
    termination=[]
    for name in METHODS:
        for label in ('all','eos_observed','no_eos'):
            rr=[r for q,r in records[name].items() if label=='all' or (len(views[name,q].ids)<len(r['generated_token_ids']))==(label=='eos_observed')]
            termination.append(dict(method=name,subset=label,n=len(rr),correct=sum(r['correct'] for r in rr),marker_present=sum('####' in r['response'] for r in rr),extraction_failures=sum(r['extraction_failed'] for r in rr)))
    continuous=[]
    for b in ('Qwen-256','Qwen-512'):
        for field in ('chars','words','native_visible_tokens','nonempty_lines','numbered_step_lines'):
            for both in (False,True):continuous.append(paired_continuous(rows,'LLaDA LTR P4',b,field,correct_both=both))
    strata={q:questions[q]['stratum'] for q in questions}
    correctness=[paired(records[a],records[b],strata,a,b) for a,b in ((METHODS[0],'Qwen-256'),(METHODS[0],'Qwen-512'),('Qwen-256','Qwen-512'))]
    # Explicit exploratory family of three, independent of historical Phase 8 family.
    running=0.
    for rank,r in enumerate(sorted(correctness,key=lambda r:r['mcnemar_exact_p'])):
        running=max(running,min(1.,(3-rank)*r['mcnemar_exact_p']));r['holm_three_tests_p']=running
    cohorts=[]
    for name in METHODS:
        for label,ids in [('all',list(questions)),('llada_correct_qwen256_wrong',[q for q in questions if records[METHODS[0]][q]['correct'] and not records['Qwen-256'][q]['correct']]),
                          ('both_llada_qwen512_correct',[q for q in questions if records[METHODS[0]][q]['correct'] and records['Qwen-512'][q]['correct']])]:
            rr=[lookup[name,q] for q in ids]
            cohorts.append(dict(method=name,cohort=label,n=len(rr),correct=sum(r['correct'] for r in rr),mean_chars=mean(r['chars'] for r in rr),mean_words=mean(r['words'] for r in rr),mean_native_visible_tokens=mean(r['native_visible_tokens'] for r in rr),mean_nonempty_lines=mean(r['nonempty_lines'] for r in rr)))
    out.mkdir(parents=True,exist_ok=True)
    for filename,data in [('per_question.csv',rows),('summary.csv',summaries),('qwen256_error_flags.csv',errors),('error_summary.csv',error_summary),('budget_transitions.csv',transitions),('paired_lengths.csv',continuous),('paired_correctness.csv',correctness),('cohorts.csv',cohorts),('termination.csv',termination)]:write_csv(out/filename,data)
    manual_packet(private,records,questions,transitions,rows);plots(out,rows,errors,transitions)
    llada_config=json.loads((Path(args.llada_tokenizer)/'tokenizer_config.json').read_text()) if (Path(args.llada_tokenizer)/'tokenizer_config.json').exists() else {}
    provenance=dict(phase="8.5",analysis_version=1,observational=True,model_revisions={name:manifests[name]['config']['model_revision'] for name in METHODS},
        phase8_provenance_sha256=digest('reports/phase8/provenance.json'),selection_sha256=digest('outputs/phase2/selection.json'),
        input_artifacts={name:{p:digest(Path(path)/p) for p in ('manifest.json','journal.jsonl')} for name,path in [('LLaDA LTR P4',args.phase2),('Qwen-256',args.qwen256),('Qwen-512',args.qwen512)]},
        tokenizers=dict(qwen=dict(verification='Exact tokenizer JSON/config SHA256 matches both historical model manifests',files={p:digest(Path(args.qwen_tokenizer)/p) for p in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt') if (Path(args.qwen_tokenizer)/p).exists()}),
            llada=dict(verification='Historical manifest has no tokenizer checksum; local ByteLevel vocabulary exactly reproduces all 60 raw and visible response sequences; revision identity of tokenizer asset not independently authenticated',files={p:digest(Path(args.llada_tokenizer)/p) for p in ('tokenizer.json','tokenizer_config.json') if (Path(args.llada_tokenizer)/p).exists()})),
        exact_decoder_checks={name:dict(n=60,raw_and_visible_match=60,strict_utf8_offsets=sum(views[name,q].offsets_available for q in questions)) for name in METHODS},
        bootstrap=dict(repetitions=10000,seed=42,generator='NumPy PCG64 reset per contrast',method='Paired question differences, resample within gold calculation strata; linear percentile 2.5/97.5; pointwise. Both-correct subsets resample their observed stratum sizes. Length contrasts are exploratory, no multiplicity-adjusted intervals.'),
        environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=__import__('matplotlib').__version__),
        source_sha256={p:digest(p) for p in ('src/dllm_unmasking/reasoning_budget.py','src/dllm_unmasking/analyze_phase8_5.py','src/dllm_unmasking/evaluation.py','src/dllm_unmasking/prepare_ltr_audit.py')},
        prompt_audit=dict(qwen_rendered_prompts_recorded=True,qwen_default_system_observed=all('You are Qwen, created by Alibaba Cloud.' in r['rendered_prompt'] for r in records['Qwen-256'].values()),llada_rendered_prompts_not_logged=True,llada_local_template_available=bool(llada_config.get('chat_template')),same_question_and_suffix=True),
        limitations=['Native token counts use different vocabularies','Output-order prefixes are not causal reasoning steps; LLaDA initially commits groups of four','Complete numeric-span prefixes use hindsight, not an online stopping rule','No automatic arithmetic/reasoning-error or parser-bug annotation'])
    receipt=Path('reports/phase8_5/renaming_provenance.json')
    if out!=receipt.resolve().parent:(out/'renaming_provenance.json').write_bytes(receipt.read_bytes())
    provenance['naming_cleanup']=dict(receipt_sha256=digest(receipt),original_provenance_sha256=json.loads(receipt.read_text())['original_provenance_sha256'],scientific_results_unchanged=True)
    atomic_json(out/'provenance.json',provenance)
    write_report(out,rows,summaries,errors,transitions,continuous,correctness,cohorts,records,provenance,termination)
    if any(digest(p)!=sha for p,sha in old.items()):raise RuntimeError('Earlier reports changed')
    for path,hashes in protected.items():
        if inventory(Path(path))!=hashes:raise RuntimeError('Raw artifacts changed')
    print('Verified 180 matched responses; exact Qwen prefix matches: 60/60; Phase 8.5 report:',out/'PHASE8_5_REPORT.md')


def write_report(out,rows,summaries,errors,transitions,lengths,correctness,cohorts,records,provenance,termination):
    index={(r['method'],r['question_id']):r for r in rows}
    aggregate=[]
    for name in METHODS:
        r=next(r for r in summaries if r['method']==name and r['subset']=='all')
        aggregate.append(dict(method=name,correct=f"{r['correct']}/{r['n']}",chars=f"{r['chars_mean']:.1f}",words=f"{r['words_mean']:.1f}",native=f"{r['native_visible_tokens_mean']:.1f}",lines=f"{r['nonempty_lines_mean']:.1f}",marker_n=r['last_marker_native_start_n'],marker_native=f"{r['last_marker_native_start_mean']:.1f}",answer_native=f"{r['selected_answer_native_end_mean']:.1f}"))
    length_table=[dict(a=r['a'],b=r['b'],subset=r['subset'],metric=r['metric'],n=r['n'],delta=f"{r['mean_b_minus_a']:.2f}",ci=f"[{r['bootstrap_low']:.2f}, {r['bootstrap_high']:.2f}]") for r in lengths if r['metric'] in ('chars','words')]
    termination_display=[r for r in termination if r['method']=='Qwen-256' and r['subset']!='all']
    error_rows=[];nwrong=sum(e['incorrect'] for e in errors)
    labels={'incorrect_and_cap_limited':'Wrong and cap-limited (unfinished status not guaranteed)','incorrect_missing_marker':'Wrong and missing ####','incorrect_with_explicit_wrong_number':'Wrong explicit-marker/box/phrase selected number','incorrect_with_marker_wrong_number':'Wrong #### selected number','incorrect_extraction_failure':'Extraction failure','incorrect_ambiguous_answer_role':'No parseable explicit numerical answer; role ambiguous','incorrect_with_gold_elsewhere':'Gold-valued numerical mention elsewhere','incorrect_with_explicit_gold_elsewhere':'Explicit gold-valued assertion elsewhere','incomplete_numeric_candidate_at_boundary':'Selected numeric span touches capped boundary (possible incomplete number)'}
    for key,label in labels.items():error_rows.append(dict(label=label,n=sum(e[key] for e in errors),denom=nwrong))
    recoveries=[t for t in transitions if t['transition']=='recovery'];regressions=[t for t in transitions if t['transition']=='regression']
    available=[t['first_complete_explicit_parser_gold_native_512'] for t in recoveries if t['first_complete_explicit_parser_gold_native_512'] is not None]
    beyond=sum(v>256 for v in available)
    mismatch=[t for t in transitions if t['llada_correct'] and not t['qwen256_correct']]
    mismatch_recover=sum(t['qwen512_correct'] for t in mismatch)
    cases=[]
    selections=[('regression',[t for t in transitions if t['transition']=='regression']),
        ('LTR P4 correct, Qwen recovered',[t for t in mismatch if t['qwen512_correct']]),
        ('LTR P4 correct, Qwen still wrong',[t for t in mismatch if not t['qwen512_correct']]),
        ('Qwen256 explicit wrong answer',[t for t in transitions if not t['qwen256_correct'] and index['Qwen-256',t['question_id']]['final_marker_present']]),
        ('Qwen512 extraction failure',[t for t in transitions if index['Qwen-512',t['question_id']]['extraction_failed']])]
    selected_ids=set()
    for label,ts in selections:
        if not ts:continue
        t=min(ts,key=lambda t:t['question_id']);q=t['question_id'];selected_ids.add(q)
        a,b,c=(index[name,q] for name in METHODS)
        cases.append(dict(category=label,qid=q,stratum=t['stratum'],gold=records['Qwen-256'][q]['gold_numeric'],llada=str(a['predicted_numeric']),q256=str(b['predicted_numeric']),q512=str(c['predicted_numeric']),end=c['first_complete_explicit_parser_gold_native']))
    mismatch_lengths=[r for r in cohorts if r['cohort']=='llada_correct_qwen256_wrong']
    cohort_display=[dict(method=r['method'],n=r['n'],correct=r['correct'],words=f"{r['mean_words']:.1f}",chars=f"{r['mean_chars']:.1f}",native=f"{r['mean_native_visible_tokens']:.1f}") for r in mismatch_lengths]
    mismatch_cap=sum(t['cap_limited_256'] for t in mismatch)
    paired_table=[dict(a=r['a'],b=r['b'],n=r['n'],recover=r['b_recovers'],regress=r['b_regresses'],delta=f"{r['b_minus_a_pp']:.2f}",ci=f"[{r['bootstrap_low_pp']:.2f}, {r['bootstrap_high_pp']:.2f}]",p=f"{r['mcnemar_exact_p']:.6g}",holm=f"{r['holm_three_tests_p']:.6g}") for r in correctness]
    placement_table=[]
    for name in METHODS:
        mr=[r for r in rows if r['method']==name and r['final_marker_present']]
        ar=[r for r in rows if r['method']==name and r['selected_answer_char_end'] is not None]
        placement_table.append(dict(method=name,marker_n=len(mr),marker_chars=f"{mean(r['last_marker_char_start'] for r in mr):.1f}",marker_words=f"{mean(r['last_marker_word_start'] for r in mr):.1f}",marker_native=f"{mean(r['last_marker_native_start'] for r in mr):.1f}",answer_n=len(ar),answer_chars=f"{mean(r['selected_answer_char_end'] for r in ar):.1f}",answer_words=f"{mean(r['selected_answer_word_end'] for r in ar):.1f}",answer_native=f"{mean(r['selected_answer_native_end'] for r in ar):.1f}"))
    q256_words=next(r for r in lengths if r['b']=='Qwen-256' and r['metric']=='words' and r['subset']=='all')
    earlier_gold=sum(t['gold_mentioned_256'] for t in recoveries)
    compact=next(r for r in lengths if r['b']=='Qwen-512' and r['metric']=='words' and r['subset']=='both_numerically_correct')
    text=f'''# Phase 8.5 — Reasoning Budget Utilization Analysis

## Supplementary Phase 8 analysis: question and design

Why does LLaDA LTR P4 attain 37/60 numerical-answer correctness at 256 generated
positions while Qwen reaches 28/60 at cap 256 and 50/60 at cap 512? Phase 8.5 is a supplementary analysis of Phase 8, not a separate inference experiment.
Phase 9 is reserved for a future Qwen-256 concise-reasoning intervention. This is an
**observational analysis of existing outputs**, not a GPU experiment or causal
architecture test. It audits all 60 matched questions in three outputs (180 total).
The exact question IDs, gold texts/labels, fixed instruction suffix, selection hash,
model revisions, numerical evaluator and raw artifact hashes match the immutable
[Phase 8 benchmark](../phase8/PHASE8_REPORT.md). Phase 2's LTR P4 journal and
manifest match the Phase 8 recorded hashes; both Qwen imports match every recorded
imported-file hash. Duplicate, incomplete, or mismatched inputs fail closed.

Model revisions: LLaDA `852152c60bdfed4632635bfcf3e83408ad3f65bf`; Qwen
`a09a35458c702b33eeacc393d103063234e8bc28`. Dataset revision:
`740312add88f781978c0658806c59bc2815b9866`. Each gold calculation-count stratum
(0–2, 3–4, 5+) has 20 questions; these strata are proxies, not reasoning depth.
The nominal 256 budgets differ in tokenization, compute, stopping and architecture.
LLaDA always commits 256 positions in 64 four-position steps; Qwen uses native
cached greedy generation and stops at EOS or its cap. Latency is not remeasured:
Qwen prefill+decode and LLaDA decode-only intervals remain unequal scopes.

## Measurements and definitions

Character offsets count Python Unicode code points. Words are whitespace-delimited
`\\S+` spans, not linguistic word segmentation. Starts are **zero-based**, ends are
**exclusive**; an answer native end of 270 means its covering output tokens occupy
positions before 270. Native positions refer to output order, not forward-pass cost
or verified reasoning steps. LLaDA commits four positions together, so availability
at an intermediate token index is a prefix diagnostic, not an actual one-token step.
Generated counts include special/EOS tokens; visible native counts end before the
first EOS. Both decoders reproduce all 180 raw and visible responses exactly.
Exact offset mapping requires strict UTF-8; no approximate offsets are substituted.

The final marker is the last literal `####`. The final numerical answer column is
**the unchanged parser's selected numerical span**, not an independently annotated
intended answer. Missing markers and extraction failures remain missing, not zero.
Full-output numerical matches are used to identify complete numeric spans: a prefix
`1` inside a later `150` does not count as a complete correct answer for gold 1.
This hindsight rule is an offline diagnostic, not a proposed stopping policy.
First parser-correct prefixes, first explicit-cue parser-correct complete numbers,
and suffix-stable correctness are separate measurements. Gold values appearing
in a given quantity or intermediate calculation do not establish a final answer.

Estimated reasoning-step measures are **text-structure proxies**: nonempty lines,
numbered step lines, sentence-like units, and lines containing arithmetic symbols.
Markdown, copied givens, repetitions and formatting affect them; they do not measure
validated operations or reasoning quality. No arithmetic/reasoning-error label is
automatically assigned to a wrong numeric answer.

## Length and answer placement

{table(aggregate,[('method','Output'),('correct','Correct/n'),('chars','Mean chars'),('words','Mean words'),('native','Mean visible native tokens'),('lines','Mean nonempty lines')])}

Tokenizer-independent character/word counts are separated from native-token counts.
Capped Qwen-256 texts are censored solution trajectories, not completed solutions
whose shortness proves efficiency. Numerical correctness does not prove faithful
reasoning. Correct-only comparisons are selected cohorts and can introduce bias.

{table(placement_table,[('method','Output'),('marker_n','Marker n/60'),('marker_chars','Mean marker char start'),('marker_words','Mean marker word start'),('marker_native','Mean marker native start'),('answer_n','Selected numeric-span n/60'),('answer_chars','Mean answer char end'),('answer_words','Mean answer word end'),('answer_native','Mean answer native end')])}

All placement means are conditional on the indicated available observations.
For missing markers, the parser-selected fallback number may be an intermediate
quantity, not an intended final answer. Full distributions, denominators, correctness
strata and native/character/word positions are in [per_question.csv](per_question.csv)
and [summary.csv](summary.csv).

![Observed response lengths](response_lengths.png)

![Last marker placement](answer_placement.png)

## All Qwen-256 errors: overlapping observable flags

There are {nwrong}/60 Qwen-256 incorrect outputs. These flags deliberately overlap;
a cap-limited output may have no marker, contain a wrong fallback number, and fail
extraction simultaneously. A cap flag is observed stopping, **not proof that the
text is semantically unfinished**. Boundary-number flags likewise identify possible
incompletion, not confirmed corrupted numbers. An explicit wrong number describes
the selected answer, not a verified arithmetic error.

{table(error_rows,[('label','Observable flag'),('n','Count'),('denom','Incorrect-output denominator')])}

Qwen-256 stopping status is strongly associated with numerical correctness in
this sample. EOS versus cap status is not randomized; their conditional accuracies
cannot establish a causal effect of truncation.

{table(termination_display,[('subset','Qwen-256 termination'),('n','n'),('correct','Correct'),('marker_present','#### present'),('extraction_failures','Extraction failures')])}

[termination.csv](termination.csv) also reports all three methods. For LLaDA,
no-EOS does not mean early stopping: it still filled all 256 positions; its first
EOS only defines the decoded evaluation prefix.

[qwen256_error_flags.csv](qwen256_error_flags.csv) enumerates every question,
including the correct controls; [error_summary.csv](error_summary.csv) contains
all flag counts and denominators. Ambiguous answer-role cases require human review.
Gold-valued mentions elsewhere are **not automatically parser mistakes**: they can
be given or intermediate values. Human parser mistakes require a clearly intended
answer and evidence the extractor mishandled it.

## Budget continuation: 23 recoveries and one regression

Both the saved token-ID and visible-character prefix relations hold exactly on
60/60 questions. Among the {len(recoveries)} recoveries, an explicit-cue complete
parser-correct answer is first available in {len(available)}/{len(recoveries)};
{beyond}/{len(recoveries)} first occur beyond token 256. For the available cases,
the first native prefix lengths range from {min(available) if available else 'NA'}
to {max(available) if available else 'NA'}, median {median(available) if available else 'NA'}.
In {earlier_gold}/{len(recoveries)} recovered outputs, the gold value already
appears somewhere in Qwen-256, but none has a complete explicit parser-correct
answer by the cap. An intermediate gold-valued quantity is not itself a final answer.
This gives the actual continuation budget at which the answer becomes available,
not a claim about when reasoning was completed or whether a shorter derivation exists.

Regression cases: {len(regressions)}/60. The corresponding earlier correct
extracted value and later incorrect value are displayed in the manual packet;
the concrete regression is examined below rather than assumed to be a fallback
coincidence or damaged correct reasoning. [budget_transitions.csv](budget_transitions.csv)
contains all recoveries/regressions, first explicit/any gold placements in three
units, first parser-correct prefixes, and suffix-stability counts.

![Explicit answer availability within recovered continuations](recovery_answer_budget.png)

## Matched LLaDA-correct / Qwen-256-wrong cases

There are {len(mismatch)}/60 such questions; Qwen-512 recovers {mismatch_recover}/{len(mismatch)}.
All {mismatch_cap}/{len(mismatch)} of these Qwen-256 outputs exhaust the cap.
This comparison identifies a truncation/format/budget association, not a mechanism
specific to diffusion.

{table(cohort_display,[('method','Output on these matched disagreements'),('n','n'),('correct','Correct'),('words','Mean words'),('chars','Mean chars'),('native','Mean visible native tokens')])} Representative cases below are deterministic minimum-ID
examples from each named category; they are not prevalence estimates. The full
manual packet contains all 60 side-by-side outputs, including every one of these
{len(mismatch)} disagreements and all recovery/regression cases.

{table(cases,[('category','Selection category'),('qid','Question ID'),('stratum','Stratum'),('gold','Gold'),('llada','LTR P4 extracted'),('q256','Qwen256 extracted'),('q512','Qwen512 extracted'),('end','Qwen512 first explicit complete parser-correct native prefix')])}

### Evidence from individual outputs

- **Question 30 (recovery, gold 109):** Qwen-256 explicitly reaches `99 + 10 = 109`
  but ends with an incomplete `#### <final` suffix. The unchanged marker-priority
  parser returns failure. Qwen-512 completes the marked numerical answer. This
  particular recovery is consistent with answer-format completion, not evidence
  that additional arithmetic was required. The extractor is applying its specified
  rule; whether to rescue an earlier answer is a separate evaluation policy.
- **Question 711 (the regression, gold 6):** the 256-prefix derives six total scoops
  and the last-number fallback selects 6. The continuation distinguishes four
  paid scoops from six including free scoops, then marks 4. The question uses
  "buy" while the reference counts total scoops. This is an observable
  benchmark-answer/interpretation shift, not an automatic arithmetic-error label
  or proof that continuation spoiled valid reasoning.
- **Question 157 (512 extraction failure, gold 80):** Qwen ends with a marker and
  literal placeholder but no marked number. Its earlier computed difference is
  280; it omits the initial 200 yards in Kelly's total that the question/reference
  includes. Rescuing the earlier number would still disagree with gold. Formatting
  failure and a substantive premise omission can coexist; 3000 − 2720 = 280 is
  locally correct arithmetic, so that equation is not labeled an arithmetic error.

These are bounded manual observations of the named outputs, not automatic labels
for the remaining cases. Full questions, gold references, numerical candidates and raw/visible responses
are deliberately kept out of public artifacts. Open
[full 60-question manual audit](../../results/phase8_5/manual_audit/comparison.html)
(generated locally, excluded from Git); `annotations.csv` has blank independent
fields for intended answer, parser mistake, arithmetic evidence, reasoning quality,
format completion and verbosity. Successful extraction is not successful reasoning.

## Does compactness explain the gap?

Below, Δ is Qwen minus LLaDA on matched question IDs. Pointwise 95% intervals use
10,000 paired within-stratum bootstrap resamples, NumPy PCG64 seed 42 reset per
contrast, linear percentile quantiles. Both-correct subsets preserve their observed
stratum sizes. These are exploratory correlated contrasts without simultaneous
coverage or multiplicity-adjusted length intervals.

{table(length_table,[('b','Compared Qwen run'),('subset','Subset'),('metric','Unit'),('n','Pairs'),('delta','Mean Δ'),('ci','Bootstrap 95% CI')])}

At the matched nominal 256 budget, the Qwen-minus-LLaDA word difference is
{q256_words['mean_b_minus_a']:.2f} (CI [{q256_words['bootstrap_low']:.2f}, {q256_words['bootstrap_high']:.2f}]);
this does not establish a uniform word-count compactness difference. Among {compact['n']} questions numerically correct for both LLaDA and Qwen-512,
Qwen uses {compact['mean_b_minus_a']:.2f} more whitespace words on average
(CI [{compact['bootstrap_low']:.2f}, {compact['bootstrap_high']:.2f}]). This is a
matched descriptive compactness comparison; the both-correct filter selects an
outcome-dependent cohort. Token-count contrasts in [paired_lengths.csv](paired_lengths.csv)
remain vocabulary-dependent and must not be interpreted as equal semantic budgets.
Shorter output does not establish a more efficient correct mathematical derivation.
We cannot establish that compactness mediates or causes the accuracy gap from these
records: that would require controlled response-budget/format interventions and
independent validation of useful reasoning content.

## Correctness uncertainty

Contrasts use B minus A, with recoveries/regressions on matched questions. Bootstrap
settings match the Phase 8 versioned analysis; exact two-sided McNemar tests have a
separate **three-contrast exploratory Holm family**, so adjusted p-values differ
from Phase 8's nine-contrast family while raw counts and exact p-values agree.

{table(paired_table,[('a','A'),('b','B'),('n','n'),('recover','Recoveries'),('regress','Regressions'),('delta','Δ pp'),('ci','95% CI pp'),('p','Exact p'),('holm','Holm p (3)')])}

## Parser, prompt, output-format and verbosity audit

The shared numerical evaluator is unchanged; stored labels are independently
recomputed. Marker counts, fallback methods, malformed final suffixes and gold-valued
intermediate mentions are visible separately. Any alternative-parser success is a
sensitivity diagnostic, not revised benchmark accuracy. No alternative answer is
silently substituted for the original selected answer.

Qwen's native chat template and tokenizer JSON/config hashes exactly match both
archived model manifests. Recorded prompts include a model-specific default system
message and native ChatML delimiters, in addition to the identical question and
instruction suffix. LLaDA used its own native template in the historical runner;
rendered prompts/tokenized prompt IDs were not logged in Phase 2, so identical full
prompt templates cannot be asserted. A local LLaDA tokenizer asset reproduces every
saved output exactly but its **revision identity is not authenticated by a historical
tokenizer checksum**, which was not recorded. Exact response decoding validates the
observed native offsets, not the provenance of an unlogged prompt template.

More words, lines or tokens can reflect model-specific style, formatting, repeated
premises, useful computation or degeneration. This analysis does not equate any
of those with faithful reasoning or automatically annotate arithmetic mistakes.
The native tokenizer-dependent and tokenizer-independent comparisons remain separate.

## Confirmed observations, uncertainties and one targeted follow-up

Confirmed: matched identities/labels/hashes; native decoding and offsets on the
observed outputs; exact Qwen continuations; the observable error flags and answer
placements above; and descriptive paired length/correctness contrasts. The lower
Qwen-256 score is associated with answer availability after the cap in many cases.

Unresolved: faithful versus merely longer reasoning, intended answers in ambiguous
outputs, parser mistakes versus malformed output, semantic unfinishedness, the
role of native templates/training, and whether compactness explains any causal part
of the gap. Gold-assisted span matching is hindsight, not a stopping rule. No causal
claim about diffusion architecture is warranted by these three observational runs.

One narrowly targeted **future experiment**, if authorized separately: hold Qwen's
model, exact questions, template and decoding settings fixed, randomize a single
instruction requesting a concise derivation within the existing 256-token cap
versus the original instruction, and predefine marker completion, numerical accuracy
and blinded reasoning-quality annotation outcomes. This tests a response-compactness
intervention within one model; it does not isolate architecture or guarantee gains.
No such prompt change, inference run or experiment is implemented here.

## Reproduction and preservation

From the repository root, with CPU analysis dependencies and local tokenizer JSONs:

```bash
python -m dllm_unmasking.analyze_phase8_5 --phase2 "$PHASE2_RESULTS" \\
  --qwen-tokenizer local_tokenizers/qwen --llada-tokenizer local_tokenizers/llada
```

Qwen inputs default to the immutable Phase 8 imports under
`outputs/phase8/imported/qwen_256/` and `outputs/phase8/imported/qwen_512/`;
`--qwen256` and `--qwen512` can explicitly select exact-hash copies. Alternative
public-output reproduction can use `--output-dir results/phase8_5_recheck`; full-text
case output stays under ignored `results/phase8_5/`. No tokenizer encoding library,
Transformers runtime, weights, network or GPU is required: the audited ByteLevel
decoder maps stored IDs to UTF-8 bytes and checks raw/visible equality before use.
Unsupported decoders, mismatched Qwen tokenizer hashes, changed journals or invalid
labels cause failure rather than estimated offsets.

[Renaming provenance](renaming_provenance.json) preserves the original analysis
provenance and pre-cleanup artifact hashes. Only labels, paths and current-source
hashes changed; scientific tables and figures retain their original bytes.

Local tokenizer files remain ignored by Git. The source inference implementations
and Phase 1–8 reports/results are unchanged; inventories verify raw inputs and earlier
reports on exit. [provenance.json](provenance.json) records hashes, tokenizer
verification bounds, analysis settings, environments and source hashes. Public
artifacts contain derived metrics and compact numerical case indexes; full responses
and human annotation forms remain ignored. All categories are observational and
may overlap. No GPU inference, model download, commit or push was performed.
'''
    (out/'PHASE8_5_REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase2',type=Path,required=True,help='Local completed Phase 2 directory; never accessed remotely')
    p.add_argument('--qwen256',default='outputs/phase8/imported/qwen_256')
    p.add_argument('--qwen512',default='outputs/phase8/imported/qwen_512')
    p.add_argument('--qwen-tokenizer',default='local_tokenizers/qwen')
    p.add_argument('--llada-tokenizer',default='local_tokenizers/llada')
    p.add_argument('--output-dir',default='reports/phase8_5')
    p.add_argument('--case-dir',default='results/phase8_5/manual_audit')
    run(p.parse_args())


if __name__=='__main__':enter_repository();main()
