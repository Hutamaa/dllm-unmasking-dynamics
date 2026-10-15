"""Phase 5 confidence remasking; read-only Phase 2 P8 baseline, fixed 32 forwards."""
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
from dllm_unmasking.phase2_data import sha,atomic_json,append_event,recover_journal
from experiments.phase2.run import cached_model,csv_atomic
from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.analyze_phase25 import load,audit_trajectories,outputs,paired
from dllm_unmasking.analyze_phase15 import holm

STRATEGIES=('confidence_random_remask','confidence_low_confidence_remask')


def validate(cfg,source):
    records,lookup,manifest,_=load(source)
    audit_trajectories(records)
    selection=json.loads(Path(cfg['selection_file']).read_text())
    if sha(cfg['selection_file'])!=manifest['selection_sha256']:raise ValueError('Exact Phase 2 selection required')
    for field in ('model','model_revision','dataset_revision','dataset_test_parquet_sha256','selection_seed','inference_seed',
                  'mask_id','dtype','prompt_suffix','gen_length','block_length'):
        if cfg[field]!=manifest['config'][field]:raise ValueError(f'Baseline setting mismatch: {field}')
    if (cfg['gen_length'],cfg['block_length'],cfg['steps'],cfg['parallelism'],cfg['revision_budget'],cfg['max_revisions_per_token'],cfg['revision_seed'])!=(256,256,32,8,2,1,314159):
        raise ValueError('Only Phase 4 revision controls with confidence P8/32 passes allowed')
    if cfg['strategies']!=list(STRATEGIES):raise ValueError('Unexpected policies')
    for name,digest in manifest['source_sha256'].items():
        if not source_matches(name,digest):raise ValueError(f'Baseline source mismatch: {name}')
    baseline={q['question_id']:lookup[q['question_id'],'confidence',8] for q in selection['questions']}
    if len(selection['questions'])!=60 or len(baseline)!=60 or set(baseline)!={r['question_id'] for r in records}:
        raise ValueError('Require exact 60 distinct selected questions')
    for q in selection['questions']:
        r=baseline[q['question_id']]
        if any(q[k]!=r[k] for k in ('question','gold_answer','stratum','calculation_count')):raise ValueError('Question mismatch')
        if r['position_seed']!=cfg['inference_seed']+q['question_id']:raise ValueError('Inference seed mismatch')
    if sum(r['correct'] for r in baseline.values())!=26:raise ValueError('Expected completed baseline accuracy 26/60')
    return selection,baseline,manifest


def audit_result(r,q,cfg):
    """Replay every saved commit/revision before trusting a resume completion."""
    if any(r[k]!=q[k] for k in ('question_id','question','gold_answer','stratum','calculation_count')):raise ValueError('Saved question mismatch')
    if r['strategy'] not in STRATEGIES or (r['parallelism'],r['steps'],r['forward_passes'])!=(8,32,32):raise ValueError('Saved condition mismatch')
    if r['position_seed']!=cfg['inference_seed']+q['question_id'] or r['revision_seed']!=cfg['revision_seed']+q['question_id']:raise ValueError('Saved seed mismatch')
    import random
    rng=random.Random(r['revision_seed']);state=[cfg['mask_id']]*256;seen=set();revised=set();confidence={}
    if len(r['trajectory'])!=32:raise ValueError('Saved trajectory length mismatch')
    for step,t in enumerate(r['trajectory']):
        new=t['positions'];tokens=t['token_ids'];conf=t['confidences'];rr=t['revisions']
        eligible=sorted(seen-revised);budget=0 if step==0 else min(2,len(eligible))
        if t['step']!=step or t['forward_passes']!=1 or len(new)!=8 or len(set(new))!=8 or len(tokens)!=8 or len(conf)!=8:
            raise ValueError('Saved initial commit shape mismatch')
        if any(p in seen or not 0<=p<256 for p in new):raise ValueError('Saved new-position eligibility violation')
        if any(not math.isfinite(c) or not 0<=c<=1 for c in conf):raise ValueError('Invalid saved confidence')
        if list(zip(new,conf))!=sorted(zip(new,conf),key=lambda v:(-v[1],v[0])):raise ValueError('Saved confidence ranking mismatch')
        chosen=[v['position'] for v in rr]
        expected=rng.sample(eligible,budget) if r['strategy']=='confidence_random_remask' else sorted(eligible,key=lambda p:(confidence[p],p))[:budget]
        if chosen!=expected or t['eligible_revision_count']!=len(eligible) or t['revision_budget']!=budget:
            raise ValueError('Saved revision selection/budget mismatch')
        if t['remaining_masks_before']!=256-8*step or t['remaining_masks_after']!=256-8*(step+1):raise ValueError('Saved mask invariant mismatch')
        for v in rr:
            p=v['position']
            if v['original_token']!=state[p] or v['original_confidence']!=confidence[p]:raise ValueError('Saved original revision token/confidence mismatch')
            if v['revision_step']!=step or v['decoding_half']!=('first' if step<16 else 'second') or v['changed']!=(v['original_token']!=v['replacement_token']):
                raise ValueError('Saved revision log inconsistency')
            if v['replacement_token']==cfg['mask_id'] or not math.isfinite(v['replacement_confidence']) or not 0<=v['replacement_confidence']<=1:raise ValueError('Invalid replacement')
            state[p]=v['replacement_token']
        for p,token,c in zip(new,tokens,conf):
            if token==cfg['mask_id']:raise ValueError('Saved commitment predicts MASK')
            state[p]=token;confidence[p]=c
        seen.update(new);revised.update(chosen)
    if len(seen)!=256 or len(revised)!=62 or state!=r['generated_token_ids']:raise ValueError('Saved final state/coverage mismatch')
    predicted,method=extract_answer(r['response']);gold,_=extract_answer(q['gold_answer'],gold=True)
    if (predicted,method,gold,predicted is None,predicted==gold)!=(r['predicted_numeric'],r['extraction_method'],r['gold_numeric'],r['extraction_failed'],r['correct']):
        raise ValueError('Saved extraction/correctness mismatch')


def summarize(out,records,baseline,target,mask_id):
    if not records:return
    rows=sorted(records.values(),key=lambda r:(r['question_id'],r['strategy']));diagnostics=outputs(rows,mask_id)
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
        statistics=paired(old,new,rng,10000,[r['stratum'] for r in a])
        for old_name,new_name in (('p4_mean','phase2_confidence_p8_accuracy'),('p8_mean','phase5_accuracy'),('delta_p8_minus_p4','delta_phase5_minus_baseline')):
            statistics[new_name]=statistics.pop(old_name)
        comparisons.append(dict(strategy=strategy,**statistics,
            phase2_only_correct=lost,phase5_only_correct=gained,both_correct=sum(x and y for x,y in zip(old,new)),
            both_incorrect=sum(not x and not y for x,y in zip(old,new)),
            mcnemar_exact_p=min(1.,2*sum(math.comb(d,j) for j in range(min(lost,gained)+1))/2**d) if d else 1.))
        for r in a:
            b=baseline[r['question_id']]
            category='improved' if r['correct'] and not b['correct'] else 'regressed' if b['correct'] and not r['correct'] else 'unchanged_correct' if r['correct'] else 'unchanged_incorrect'
            cases.append(dict(question_id=r['question_id'],strategy=strategy,category=category,
                phase2_correct=b['correct'],phase5_correct=r['correct'],phase2_answer=b['predicted_numeric'],phase5_answer=r['predicted_numeric']))
    holm(comparisons);csv_atomic(out/'paired_phase2.csv',comparisons);csv_atomic(out/'matched_cases.csv',cases)
    report=['# Phase 5 exploratory matched cases','',
        'Correctness means unchanged numerical extraction, not proof of successful reasoning. '
        'Revision changes do not distinguish arithmetic correction from formatting/text changes. '
        'Partial-run comparisons are exploratory; do not infer error mechanisms automatically.','']
    for c in cases:
        if c['category'] not in ('improved','regressed'):continue
        r=next(r for r in rows if r['question_id']==c['question_id'] and r['strategy']==c['strategy']);b=baseline[c['question_id']]
        report += [f'## {c["question_id"]}: {c["strategy"]} — {c["category"]}',r['question'],'',
            f'Gold: {r["gold_numeric"]}; Phase 2 confidence P8 extracted: {b["predicted_numeric"]}; Phase 5 extracted: {r["predicted_numeric"]}.','',
            '### Phase 2 confidence P8 full visible response',b['response'],'','### Phase 5 full visible response',r['response'],'',
            '### Phase 5 full raw response',r['raw_response'],'']
    (out/'matched_report.md').write_text('\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/phase5.yaml')
    parser.add_argument('--phase2-results',help='Existing downloaded baseline directory override')
    parser.add_argument('--output-dir',help='Independent Phase 5 output override')
    group=parser.add_mutually_exclusive_group();group.add_argument('--check-cache',action='store_true');group.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text());source=Path(args.phase2_results or cfg['phase2_results']).expanduser().resolve()
    selection,baseline,previous=validate(cfg,source)
    target=experiment_target=selection['questions']
    out=Path(args.output_dir or cfg['output_dir']).resolve()
    if not out.is_relative_to(Path('outputs/phase5').resolve()):raise ValueError('Phase 5 outputs must stay under outputs/phase5/')
    if source==out or source in out.parents or out in source.parents:raise ValueError('Baseline/output overlap')
    print('Validated confidence P8 baseline: 26/60; 60 questions × 2 policies = 120 new evaluations',flush=True)
    if args.check_cache:cached_model();print('Pinned cache files available; no inference performed.');return
    if not args.summarize and not os.environ.get('PBS_JOBID'):raise SystemExit('GPU inference only inside PBS allocation')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=('src/dllm_unmasking/phase5_sampler.py','experiments/phase5/run.py','src/dllm_unmasking/phase2_samplers.py','src/dllm_unmasking/phase2_data.py',
             'experiments/phase2/run.py','src/dllm_unmasking/evaluation.py','src/dllm_unmasking/analyze_phase25.py','src/dllm_unmasking/analyze_phase15.py')
    provenance=dict(config=cfg,question_ids=[q['question_id'] for q in experiment_target],mode='full',
        selection_sha256=sha(cfg['selection_file']),phase2_manifest_sha256=sha(source/'manifest.json'),
        phase2_journal_sha256=sha(source/'journal.jsonl'),source_sha256={n:sha(n) for n in sources},new_selection='Phase2 confidence: stable descending masked-uncommitted probability',
        n_evaluations=len(experiment_target)*2,forward_passes=len(experiment_target)*2*32)
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=provenance:raise ValueError('Different source/config/mode/IDs: use fresh output directory')
    elif args.summarize:raise FileNotFoundError('No Phase 5 manifest')
    else:atomic_json(manifest,provenance)
    allowed={(q['question_id'],s,8) for q in experiment_target for s in STRATEGIES}
    work_allowed={(q['question_id'],s,8) for q in target for s in STRATEGIES}
    records,attempts=recover_journal(out/'journal.jsonl',allowed)
    questions={q['question_id']:q for q in target}
    for saved in records.values():audit_result(saved,questions[saved['question_id']],cfg)
    summarize(out,records,baseline,experiment_target,cfg['mask_id'])
    if args.summarize or work_allowed.issubset(records):print(f'{len(records)}/{len(allowed)} complete; no inference.');return
    import random,torch
    from transformers import AutoModel,AutoTokenizer
    from dllm_unmasking.phase5_sampler import generate
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
                if len(trace)!=32 or sum(t['forward_passes'] for t in trace)!=32:raise RuntimeError('Pass mismatch')
                eos=tokenizer.eos_token_id;visible=ids[:ids.index(eos)] if eos in ids else ids
                response=tokenizer.decode(visible,skip_special_tokens=True);pred,method=extract_answer(response)
                r=dict(**q,strategy=strategy,parallelism=8,steps=32,gold_numeric=gold,predicted_numeric=pred,correct=pred==gold,
                    extraction_failed=pred is None,extraction_method=method,position_seed=cfg['inference_seed']+q['question_id'],
                    revision_seed=revision_seed,response=response,raw_response=tokenizer.decode(ids,skip_special_tokens=False),
                    generated_token_ids=ids,eos_token_id=eos,trajectory=trace,forward_passes=32,gpu_seconds=seconds,wall_seconds=wall)
                audit_result(r,q,cfg)
                append_event(journal,dict(event='completed',result=r,**header));records[k]=r
                summarize(out,records,baseline,experiment_target,cfg['mask_id'])
                print(f'[{len(records)}/{len(allowed)}] {strategy} q={q["question_id"]} correct={r["correct"]} GPU={seconds:.2f}s',flush=True)


if __name__=='__main__':main()
