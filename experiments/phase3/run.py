"""Phase 3 runner, planned geometry, and paired offline analysis. No old policy runs."""
from dllm_unmasking.layout import source_matches,enter_repository
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import math
import numpy as np

os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from dllm_unmasking.phase3_schedule import schedule
from dllm_unmasking.phase2_data import sha,atomic_json,append_event,recover_journal
from experiments.phase2.run import cached_model,csv_atomic
from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.analyze_phase25 import load,geometry,outputs,paired


def validate(cfg,source):
    records,lookup,manifest,_=load(source)
    for name,digest in manifest['source_sha256'].items():
        if not source_matches(name,digest):raise ValueError(f'Phase 2 source mismatch: {name}')
    selection=json.loads(Path(cfg['selection_file']).read_text())
    if sha(cfg['selection_file'])!=manifest['selection_sha256']:raise ValueError('Exact Phase 2 selection hash required')
    previous=manifest['config']
    for field in ('model','model_revision','dataset_revision','dataset_test_parquet_sha256','selection_seed','inference_seed','mask_id','dtype','prompt_suffix'):
        if cfg[field]!=previous[field]:raise ValueError(f'Phase 2 setting mismatch: {field}')
    if (cfg['policy'],cfg['gen_length'],cfg['block_length'],cfg['steps'],cfg['parallelism'])!=('ltr_spaced',256,256,32,8):
        raise ValueError('Only ltr_spaced, 256 tokens, parallel-8/32 steps permitted')
    qids=[q['question_id'] for q in selection['questions']]
    if len(qids)!=60 or len(set(qids))!=60 or set(qids)!={r['question_id'] for r in records}:
        raise ValueError('Missing/mismatched question IDs: stop instead of substituting questions')
    for q in selection['questions']:
        for strategy in ('random','left_to_right','right_to_left','confidence','batched_influence'):
            for p in (4,8):
                r=lookup[q['question_id'],strategy,p]
                if any(q[k]!=r[k] for k in ('question','gold_answer','stratum','calculation_count')):
                    raise ValueError('Question content/stratum mismatch')
                if r['position_seed']!=cfg['inference_seed']+q['question_id']:raise ValueError('Inference seed mismatch')
    return selection,lookup,manifest


def plan(out,lookup):
    out.mkdir(parents=True,exist_ok=True)
    planned=schedule();csv_atomic(out/'schedule.csv',[dict(step=i,positions=json.dumps(b),**geometry(b)) for i,b in enumerate(planned)])
    rows=[];curves=[]
    for policy in ('ltr_spaced','left_to_right','right_to_left','confidence'):
        trajectories=[planned] if policy=='ltr_spaced' else [r['trajectory'] for k,r in sorted(lookup.items()) if k[1:]==(policy,8)]
        g=[]
        for trajectory in trajectories:
            for step,batch in enumerate(trajectory):
                positions=batch if policy=='ltr_spaced' else batch['positions']
                g.append(geometry(positions))
        rows.append(dict(policy=policy,n_trajectories=len(trajectories),**{k:mean(x[k] for x in g) for k in g[0]}))
        for step in range(32):
            pos=[p for tr in trajectories for p in (tr[step] if policy=='ltr_spaced' else tr[step]['positions'])]
            curves.append(dict(policy=policy,step=step,n_trajectories=len(trajectories),mean_normalized_position=mean(pos)/255))
    csv_atomic(out/'geometry_comparison.csv',rows);csv_atomic(out/'position_dynamics.csv',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for policy in ('ltr_spaced','left_to_right','right_to_left','confidence'):
        a=[r for r in curves if r['policy']==policy];ax.plot([r['step'] for r in a],[r['mean_normalized_position'] for r in a],label=policy)
    ax.set(xlabel='Zero-based step',ylabel='Mean newly revealed position / 255',title='Planned schedule versus observed Phase 2 P8 timing');ax.legend();fig.tight_layout();fig.savefig(out/'position_dynamics.png',dpi=160);plt.close(fig)
    text=['# Phase 3 planned schedule','',
          'For t=2b+r, select B_t={16b+r+2j : j=0,...,7}; b=0,...,15 and r=0,1. '
          'Tiles advance left to right; even offsets are committed before odd offsets inside each tile.',
          'Every p has the unique decomposition b=floor(p/16), r=p mod 2, j=floor((p mod 16)/2). '
          'Thus all 256 positions are covered once by 32 batches of eight. Minimum sorted gap=2, '
          'spread=14, adjacent pairs=0, max contiguous run=1 in every batch.','',
          'First six batches:','']+[f'- {i}: {b}' for i,b in enumerate(planned[:6])]+['',f'Final two batches: {planned[-2:]}','',
          'LTR/RTL P8 have gap=1, spread=7, adjacent pairs=7, and max contiguous run=8. '
          'Spaced order changes within-tile reveal timing: later even positions may be revealed before '
          'earlier odd positions. It is left-prioritized, not strict monotonically increasing LTR. '
          'At each tile the average positions are 16b+7 then 16b+8, versus contiguous LTR 16b+3.5 then 16b+11.5. '
          'Spacing and contextual direction/timing therefore change together. Geometry does not measure token dependency strength. '
          'This is an intervention plan, not evidence of improved accuracy. Only one planned deterministic '
          'trajectory is shown, compared with 60 observed trajectories per existing policy.']
    (out/'report.md').write_text('\n'.join(text)+'\n')


def summarize(out,records,lookup,mask_id):
    if not records:return
    rows=list(records.values())
    fields=('question_id','stratum','strategy','parallelism','steps','correct','gold_numeric','predicted_numeric','extraction_method','extraction_failed','forward_passes','gpu_seconds','wall_seconds','response','raw_response')
    csv_atomic(out/'results.csv',[{k:r[k] for k in fields} for r in rows])
    csv_atomic(out/'summary.csv',[dict(stratum=s,n=len(a),expected_n=60 if s=='all' else 20,
        correct=sum(r['correct'] for r in a),accuracy=mean(r['correct'] for r in a) if a else None,
        extraction_failures=sum(r['extraction_failed'] for r in a)) for s in ('all','0-2','3-4','5+')
        for a in [[r for r in rows if s=='all' or r['stratum']==s]]])
    csv_atomic(out/'response_diagnostics.csv',outputs(rows,mask_id))
    rng=np.random.default_rng(42);comparisons=[]
    for policy in ('left_to_right','right_to_left','confidence','random','batched_influence'):
        for p in (4,8):
            ids=sorted(r['question_id'] for r in rows)
            old=[int(lookup[q,policy,p]['correct']) for q in ids];new=[int(records[q,'ltr_spaced',8]['correct']) for q in ids]
            lost=sum(a and not b for a,b in zip(old,new));gained=sum(b and not a for a,b in zip(old,new));d=lost+gained
            comparisons.append(dict(phase2_policy=policy,phase2_parallelism=p,**paired(old,new,rng,10000,[lookup[q,policy,p]['stratum'] for q in ids]),
                old_only_correct=lost,spaced_only_correct=gained,
                mcnemar_exact_p=min(1.,2*sum(math.comb(d,j) for j in range(min(lost,gained)+1))/2**d) if d else 1.))
    from dllm_unmasking.analyze_phase15 import holm
    holm(comparisons);csv_atomic(out/'paired_phase2_comparisons.csv',comparisons)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/phase3.yaml')
    parser.add_argument('--phase2-results',help='Read-only completed Phase 2 directory override')
    group=parser.add_mutually_exclusive_group();group.add_argument('--plan',action='store_true');group.add_argument('--check-cache',action='store_true');group.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text());source=Path(args.phase2_results or cfg['phase2_results']).expanduser().resolve()
    out=Path(cfg['output_dir']).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Output/source must not overlap')
    selection,lookup,previous=validate(cfg,source)
    if args.plan:plan(out.parent/'plan',lookup);print('Schedule coverage and geometry plan saved.');return
    if args.check_cache:cached_model();print('Pinned cached model files available; no inference.');return
    if not args.summarize and not os.environ.get('PBS_JOBID'):raise SystemExit('Inference only in a PBS GPU allocation')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    provenance=dict(config=cfg,selection_sha256=sha(cfg['selection_file']),phase2_journal_sha256=sha(source/'journal.jsonl'),
        phase2_manifest_sha256=sha(source/'manifest.json'),schedule=schedule(),n_evaluations=60,
        source_sha256={n:sha(n) for n in ('experiments/phase3/run.py','src/dllm_unmasking/phase3_sampler.py','src/dllm_unmasking/phase3_schedule.py','src/dllm_unmasking/phase2_samplers.py','src/dllm_unmasking/phase2_data.py','experiments/phase2/run.py','src/dllm_unmasking/evaluation.py','src/dllm_unmasking/analyze_phase25.py','src/dllm_unmasking/analyze_phase15.py')})
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=provenance:raise ValueError('Provenance changed; fresh output_dir required')
    elif args.summarize:raise FileNotFoundError('No Phase 3 run manifest')
    else:atomic_json(manifest,provenance)
    allowed={(q['question_id'],'ltr_spaced',8) for q in selection['questions']}
    records,attempts=recover_journal(out/'journal.jsonl',allowed);summarize(out,records,lookup,cfg['mask_id'])
    if args.summarize or len(records)==60:print(f'{len(records)}/60 completed; no inference needed.');return
    import torch,random
    from transformers import AutoModel,AutoTokenizer
    from dllm_unmasking.phase3_sampler import generate
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA GPU required')
    random.seed(cfg['inference_seed']);torch.manual_seed(cfg['inference_seed']);torch.cuda.manual_seed_all(cfg['inference_seed'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=cached_model();loading=dict(local_files_only=True,trust_remote_code=True)
    tokenizer=AutoTokenizer.from_pretrained(str(path),**loading)
    model=AutoModel.from_pretrained(str(path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**loading).to('cuda').eval()
    atomic_json(out/f'environment-{os.environ["PBS_JOBID"]}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),eos_token_id=tokenizer.eos_token_id))
    with (out/'journal.jsonl').open('a') as journal:
        for q in selection['questions']:
            k=(q['question_id'],'ltr_spaced',8)
            if k in records:continue
            text=tokenizer.apply_chat_template([dict(role='user',content=q['question']+cfg['prompt_suffix'])],tokenize=False,add_generation_prompt=True)
            prompt=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if prompt.shape[1]+256>getattr(model.config,'max_position_embeddings',4096):raise ValueError('Context limit exceeded')
            gold,_=extract_answer(q['gold_answer'],gold=True)
            if gold is None:raise ValueError('Gold extraction failure')
            header=dict(question_id=q['question_id'],strategy='ltr_spaced',parallelism=8,attempt=attempts.get(k,0)+1)
            append_event(journal,dict(event='started',**header))
            ids,seconds,wall,trace=generate(model,prompt,mask_id=cfg['mask_id'])
            if [t['positions'] for t in trace]!=schedule() or sum(t['forward_passes'] for t in trace)!=32:raise RuntimeError('Unexpected schedule/passes')
            eos=tokenizer.eos_token_id;visible=ids[:ids.index(eos)] if eos in ids else ids
            response=tokenizer.decode(visible,skip_special_tokens=True);predicted,method=extract_answer(response)
            record=dict(**q,strategy='ltr_spaced',parallelism=8,steps=32,gold_numeric=gold,predicted_numeric=predicted,
                correct=predicted==gold,extraction_failed=predicted is None,extraction_method=method,
                position_seed=cfg['inference_seed']+q['question_id'],response=response,raw_response=tokenizer.decode(ids,skip_special_tokens=False),
                generated_token_ids=ids,eos_token_id=eos,trajectory=trace,forward_passes=32,gpu_seconds=seconds,wall_seconds=wall)
            append_event(journal,dict(event='completed',result=record,**header));records[k]=record
            summarize(out,records,lookup,cfg['mask_id']);print(f'[{len(records)}/60] ltr_spaced q={q["question_id"]} correct={record["correct"]} GPU={seconds:.2f}s',flush=True)


if __name__=='__main__':main()
