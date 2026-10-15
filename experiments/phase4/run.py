"""Phase 4 only; immutable Phase 3 baseline, offline cache and 32 forwards/evaluation."""
from dllm_unmasking.layout import source_matches,enter_repository
import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
import numpy as np

os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from dllm_unmasking.phase3_schedule import schedule
from dllm_unmasking.phase2_data import sha,atomic_json,append_event,recover_journal
from experiments.phase2.run import cached_model,csv_atomic
from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.analyze_phase25 import outputs,paired
from dllm_unmasking.analyze_phase15 import holm

STRATEGIES=('random_remask','low_confidence_remask')


def validate(cfg,source):
    manifest=json.loads((source/'manifest.json').read_text());selection=json.loads(Path(cfg['selection_file']).read_text())
    if sha(cfg['selection_file'])!=manifest['selection_sha256']:raise ValueError('Exact Phase 3 selection required')
    for field in ('model','model_revision','dataset_revision','dataset_test_parquet_sha256','selection_seed','inference_seed',
                  'mask_id','dtype','prompt_suffix','gen_length','block_length','steps','parallelism'):
        if cfg[field]!=manifest['config'][field]:raise ValueError(f'Phase 3 setting mismatch: {field}')
    if (cfg['gen_length'],cfg['steps'],cfg['parallelism'],cfg['revision_budget'],cfg['max_revisions_per_token'])!=(256,32,8,2,1):
        raise ValueError('Only 256 tokens, 32 steps, P8, two revisions/step, once/token allowed')
    if cfg['strategies']!=list(STRATEGIES) or manifest['schedule']!=schedule():raise ValueError('Policy/schedule mismatch')
    for name,digest in manifest['source_sha256'].items():
        if not source_matches(name,digest):raise ValueError(f'Phase 3 source mismatch: {name}')
    records={};pending=set()
    with (source/'journal.jsonl').open() as f:
        for line in f:
            if not line.endswith('\n'):raise ValueError('Incomplete Phase 3 journal; not repaired')
            e=json.loads(line);k=(e['question_id'],e['strategy'],e['parallelism'])
            if k[1:]!=('ltr_spaced',8):raise ValueError('Unexpected Phase 3 policy')
            if e['event']=='started':
                if k[0] in records:raise ValueError('Phase 3 start after completion')
                pending.add(k)
            elif e['event']=='completed':
                if k not in pending or k[0] in records:raise ValueError('Duplicate/unstarted baseline completion')
                r=e['result']
                if (r['question_id'],r['strategy'],r['parallelism'])!=k:raise ValueError('Baseline key mismatch')
                records[k[0]]=r;pending.remove(k)
            else:raise ValueError('Unknown baseline journal event')
    ids={q['question_id'] for q in selection['questions']}
    if pending or len(selection['questions'])!=60 or len(ids)!=60 or set(records)!=ids:
        raise ValueError('Require all 60 complete matched Phase 3 questions')
    for q in selection['questions']:
        r=records[q['question_id']]
        if any(q[k]!=r[k] for k in ('question','gold_answer','calculation_count','stratum')):raise ValueError('Baseline question mismatch')
        if r['position_seed']!=cfg['inference_seed']+q['question_id']:raise ValueError('Baseline seed mismatch')
        if [t['positions'] for t in r['trajectory']]!=schedule() or r['forward_passes']!=32:raise ValueError('Baseline schedule mismatch')
        pred,method=extract_answer(r['response']);gold,_=extract_answer(q['gold_answer'],gold=True)
        if (pred,method,gold,pred==gold)!=(r['predicted_numeric'],r['extraction_method'],r['gold_numeric'],r['correct']):
            raise ValueError('Baseline extraction mismatch')
    return selection,records,manifest


def summarize(out,records,baseline,target,mask_id):
    if not records:return
    rows=list(records.values());diagnostics=outputs(rows,mask_id)
    # Earlier diagnostics assume irreversible reveals; revision-aware EOS timing must use its last commit.
    for d in diagnostics:
        r=next(r for r in rows if r['question_id']==d['question_id'] and r['strategy']==d['strategy'])
        position=d['first_eos_position'];last=None
        if position is not None:
            for t in r['trajectory']:
                if position in t['positions'] or any(v['position']==position for v in t['revisions']):last=t['step']
        d.pop('first_eos_reveal_step_fraction',None)
        d['first_eos_last_commit_step_fraction']=(last+1)/32 if last is not None else None
    diag={(r['question_id'],r['strategy']):r for r in diagnostics}
    revisions=[];result_rows=[]
    for r in rows:
        rr=[v for step in r['trajectory'] for v in step['revisions']]
        for v in rr:revisions.append(dict(question_id=r['question_id'],strategy=r['strategy'],**v))
        d=diag[r['question_id'],r['strategy']]
        result_rows.append({k:r[k] for k in ('question_id','stratum','strategy','correct','predicted_numeric','gold_numeric',
             'extraction_method','extraction_failed','forward_passes','gpu_seconds','wall_seconds','revision_seed','response','raw_response')}|
            dict(n_revised=len(rr),n_changed=sum(v['changed'] for v in rr),changed_fraction=mean(v['changed'] for v in rr),
                 max_word_run=d['max_consecutive_word_run'],repeated_token_fraction=d['generated_token_repeat_fraction'],
                 final_marker_present=d['final_marker_present'],parser_fallback=d['fallback']))
    csv_atomic(out/'results.csv',result_rows);csv_atomic(out/'response_diagnostics.csv',diagnostics)
    csv_atomic(out/'revisions.csv',revisions)
    summary=[]
    for strategy in STRATEGIES:
        for stratum in ('all','0-2','3-4','5+'):
            a=[r for r in result_rows if r['strategy']==strategy and (stratum=='all' or r['stratum']==stratum)]
            n=len(a);changes=sum(r['n_changed'] for r in a);revised=sum(r['n_revised'] for r in a)
            summary.append(dict(strategy=strategy,stratum=stratum,n=n,expected_n=sum(q['stratum']==stratum or stratum=='all' for q in target),
                correct=sum(r['correct'] for r in a),accuracy=mean(r['correct'] for r in a) if n else None,
                extraction_failures=sum(r['extraction_failed'] for r in a),parser_fallbacks=sum(r['parser_fallback'] for r in a),
                final_markers=sum(r['final_marker_present'] for r in a),n_revised=revised,n_changed=changes,
                changed_fraction=changes/revised if revised else None,
                mean_max_word_run=mean(r['max_word_run'] for r in a) if n else None,
                mean_repeated_token_fraction=mean(r['repeated_token_fraction'] for r in a) if n else None,
                forward_passes=sum(r['forward_passes'] for r in a),mean_gpu_seconds=mean(r['gpu_seconds'] for r in a) if n else None,
                mean_wall_seconds=mean(r['wall_seconds'] for r in a) if n else None))
    csv_atomic(out/'summary.csv',summary)
    confidence_rows=[]
    for strategy in STRATEGIES:
        for half in ('all','first','second'):
            rr=[v for v in revisions if v['strategy']==strategy and (half=='all' or v['decoding_half']==half)]
            if not rr:continue
            for field in ('original_confidence','replacement_confidence'):
                values=[v[field] for v in rr]
                confidence_rows.append(dict(strategy=strategy,half=half,confidence_type=field,n_revisions=len(values),
                    mean=mean(values),min=min(values),q25=float(np.quantile(values,.25)),median=float(np.median(values)),
                    q75=float(np.quantile(values,.75)),max=max(values)))
    csv_atomic(out/'revision_confidence.csv',confidence_rows)
    comparisons=[];cases=[];rng=np.random.default_rng(42)
    for strategy in STRATEGIES:
        a=[r for r in rows if r['strategy']==strategy]
        if not a:continue
        old=[int(baseline[r['question_id']]['correct']) for r in a];new=[int(r['correct']) for r in a]
        lost=sum(x and not y for x,y in zip(old,new));gained=sum(y and not x for x,y in zip(old,new));d=lost+gained
        comparisons.append(dict(strategy=strategy,**paired(old,new,rng,10000,[r['stratum'] for r in a]),
            phase3_only_correct=lost,phase4_only_correct=gained,both_correct=sum(x and y for x,y in zip(old,new)),
            both_incorrect=sum(not x and not y for x,y in zip(old,new)),
            mcnemar_exact_p=min(1.,2*sum(math.comb(d,j) for j in range(min(lost,gained)+1))/2**d) if d else 1.))
        for r in a:
            b=baseline[r['question_id']]
            category='improved' if r['correct'] and not b['correct'] else 'regressed' if b['correct'] and not r['correct'] else 'unchanged_correct' if r['correct'] else 'unchanged_incorrect'
            cases.append(dict(question_id=r['question_id'],strategy=strategy,category=category,
                phase3_correct=b['correct'],phase4_correct=r['correct'],phase3_answer=b['predicted_numeric'],phase4_answer=r['predicted_numeric']))
    holm(comparisons);csv_atomic(out/'paired_phase3.csv',comparisons);csv_atomic(out/'matched_cases.csv',cases)
    report=['# Phase 4 exploratory matched cases','',
        'Correctness means unchanged numerical extraction, not proof of successful reasoning. '
        'Revision changes do not distinguish arithmetic correction from formatting/text changes. '
        'Partial runs and pilot subsets are exploratory; do not infer error mechanisms automatically.','']
    for c in cases:
        if c['category'] not in ('improved','regressed'):continue
        r=next(r for r in rows if r['question_id']==c['question_id'] and r['strategy']==c['strategy']);b=baseline[c['question_id']]
        report += [f'## {c["question_id"]}: {c["strategy"]} — {c["category"]}',r['question'],'',
            f'Gold: {r["gold_numeric"]}; Phase 3 extracted: {b["predicted_numeric"]}; Phase 4 extracted: {r["predicted_numeric"]}.','',
            '### Phase 3 full visible response',b['response'],'','### Phase 4 full visible response',r['response'],'',
            '### Phase 4 full raw response',r['raw_response'],'']
    (out/'matched_report.md').write_text('\n'.join(report)+'\n')


def workload(selection,limit,smoke):
    groups=[[q for q in selection['questions'] if q['stratum']==s] for s in ('0-2','3-4','5+')]
    ordered=[group[i] for i in range(20) for group in groups]
    target=ordered[:limit]
    return target,target if smoke else ordered


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/phase4.yaml')
    parser.add_argument('--phase3-results',help='Existing downloaded baseline directory override')
    parser.add_argument('--output-dir',help='Independent Phase 4 output override')
    parser.add_argument('--limit',type=int,choices=range(1,61))
    group=parser.add_mutually_exclusive_group();group.add_argument('--check-cache',action='store_true');group.add_argument('--smoke-test',action='store_true');group.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text());source=Path(args.phase3_results or cfg['phase3_results']).expanduser().resolve()
    selection,baseline,previous=validate(cfg,source)
    limit=args.limit or (2 if args.smoke_test else 60)
    if args.smoke_test and limit>2:parser.error('Smoke test limited to at most two questions')
    # Round-robin strata: pilot-20 has 7/7/6; full run retains all 60 original IDs.
    target,experiment_target=workload(selection,limit,args.smoke_test)
    if args.output_dir:out=Path(args.output_dir).resolve()
    elif args.smoke_test:out=Path('outputs/phase4/smoke').resolve()
    else:out=Path(cfg['output_dir']).resolve()
    if not out.is_relative_to(Path('outputs/phase4').resolve()):raise ValueError('Phase 4 outputs must stay under outputs/phase4/')
    if source==out or source in out.parents or out in source.parents:raise ValueError('Baseline/output overlap')
    print(f'Validated Phase 3 baseline and selection; {limit} questions × 2 policies = {limit*2} evaluations',flush=True)
    if args.check_cache:cached_model();print('Pinned cache files available; no inference performed.');return
    if not args.summarize and not os.environ.get('PBS_JOBID'):raise SystemExit('GPU inference only inside PBS allocation')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=('src/dllm_unmasking/phase4_sampler.py','experiments/phase4/run.py','src/dllm_unmasking/phase3_schedule.py','src/dllm_unmasking/phase2_samplers.py','src/dllm_unmasking/phase2_data.py',
             'experiments/phase2/run.py','src/dllm_unmasking/evaluation.py','src/dllm_unmasking/analyze_phase25.py','src/dllm_unmasking/analyze_phase15.py')
    provenance=dict(config=cfg,question_ids=[q['question_id'] for q in experiment_target],mode='smoke' if args.smoke_test else 'production',
        selection_sha256=sha(cfg['selection_file']),phase3_manifest_sha256=sha(source/'manifest.json'),
        phase3_journal_sha256=sha(source/'journal.jsonl'),source_sha256={n:sha(n) for n in sources},schedule=schedule(),
        n_evaluations=len(experiment_target)*2,forward_passes=len(experiment_target)*2*32)
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=provenance:raise ValueError('Different source/config/mode/IDs: use fresh output directory')
    elif args.summarize:raise FileNotFoundError('No Phase 4 manifest')
    else:atomic_json(manifest,provenance)
    allowed={(q['question_id'],s,8) for q in experiment_target for s in STRATEGIES}
    work_allowed={(q['question_id'],s,8) for q in target for s in STRATEGIES}
    records,attempts=recover_journal(out/'journal.jsonl',allowed);summarize(out,records,baseline,experiment_target,cfg['mask_id'])
    if args.summarize or work_allowed.issubset(records):print(f'{len(records)}/{len(allowed)} complete; no inference.');return
    import random,torch
    from transformers import AutoModel,AutoTokenizer
    from dllm_unmasking.phase4_sampler import generate
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required')
    random.seed(cfg['inference_seed']);torch.manual_seed(cfg['inference_seed']);torch.cuda.manual_seed_all(cfg['inference_seed'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=cached_model();loading=dict(local_files_only=True,trust_remote_code=True)
    tokenizer=AutoTokenizer.from_pretrained(str(path),**loading)
    model=AutoModel.from_pretrained(str(path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**loading).to('cuda').eval()
    atomic_json(out/f'environment-{os.environ["PBS_JOBID"]}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    with (out/'journal.jsonl').open('a') as journal:
        for i,q in enumerate(target):
            text=tokenizer.apply_chat_template([dict(role='user',content=q['question']+cfg['prompt_suffix'])],tokenize=False,add_generation_prompt=True)
            prompt=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if prompt.shape[1]+256>getattr(model.config,'max_position_embeddings',4096):raise ValueError('Context limit exceeded')
            gold,_=extract_answer(q['gold_answer'],gold=True)
            if gold is None:raise ValueError('Gold extraction failed')
            policies=STRATEGIES if i%2==0 else STRATEGIES[::-1]
            for strategy in policies:
                k=(q['question_id'],strategy,8)
                if k in records:continue
                header=dict(question_id=q['question_id'],strategy=strategy,parallelism=8,attempt=attempts.get(k,0)+1)
                append_event(journal,dict(event='started',**header))
                revision_seed=cfg['revision_seed']+q['question_id']
                ids,seconds,wall,trace=generate(model,prompt,strategy,mask_id=cfg['mask_id'],revision_seed=revision_seed)
                if [t['positions'] for t in trace]!=schedule() or sum(t['forward_passes'] for t in trace)!=32:raise RuntimeError('Schedule/pass mismatch')
                eos=tokenizer.eos_token_id;visible=ids[:ids.index(eos)] if eos in ids else ids
                response=tokenizer.decode(visible,skip_special_tokens=True);pred,method=extract_answer(response)
                r=dict(**q,strategy=strategy,parallelism=8,steps=32,gold_numeric=gold,predicted_numeric=pred,correct=pred==gold,
                    extraction_failed=pred is None,extraction_method=method,position_seed=cfg['inference_seed']+q['question_id'],
                    revision_seed=revision_seed,response=response,raw_response=tokenizer.decode(ids,skip_special_tokens=False),
                    generated_token_ids=ids,eos_token_id=eos,trajectory=trace,forward_passes=32,gpu_seconds=seconds,wall_seconds=wall)
                append_event(journal,dict(event='completed',result=r,**header));records[k]=r
                summarize(out,records,baseline,experiment_target,cfg['mask_id'])
                print(f'[{len(records)}/{len(allowed)}] {strategy} q={q["question_id"]} correct={r["correct"]} GPU={seconds:.2f}s',flush=True)


if __name__=='__main__':main()
