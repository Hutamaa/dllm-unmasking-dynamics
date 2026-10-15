"""Matched P1 benchmark; reuse the original sampler without changing earlier phases."""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from phase2_data import sha,atomic_json,append_event,recover_journal
from run_phase2 import cached_model,csv_atomic
from evaluation import extract_answer
from analyze_phase25 import load,audit_trajectories

POLICIES=('left_to_right','confidence')


def validate(cfg,source):
    records,lookup,manifest,_=load(source);audit_trajectories(records)
    selection=json.loads(Path(cfg['selection_file']).read_text())
    if sha(cfg['selection_file'])!=manifest['selection_sha256']:raise ValueError('Exact Phase 2 selection hash required')
    for field in ('model','model_revision','dataset_revision','dataset_test_parquet_sha256','selection_seed','inference_seed','mask_id','dtype','prompt_suffix','gen_length','block_length'):
        if cfg[field]!=manifest['config'][field]:raise ValueError(f'Phase 2 setting mismatch: {field}')
    if (cfg['gen_length'],cfg['block_length'],cfg['steps'],cfg['parallelism'],cfg['policies'])!=(256,256,256,1,list(POLICIES)):
        raise ValueError('Only matched P1/256 steps, LTR and confidence, no revisions')
    for name,digest in manifest['source_sha256'].items():
        if sha(name)!=digest:raise ValueError(f'Phase 2 source mismatch: {name}')
    ids=[q['question_id'] for q in selection['questions']]
    if len(ids)!=60 or len(set(ids))!=60 or set(ids)!={r['question_id'] for r in records}:raise ValueError('Question ID mismatch')
    for q in selection['questions']:
        for policy in POLICIES:
            for p in (4,8):
                r=lookup[q['question_id'],policy,p]
                if any(q[k]!=r[k] for k in ('question','gold_answer','stratum','calculation_count')):raise ValueError('Question metadata mismatch')
                if r['position_seed']!=cfg['inference_seed']+q['question_id']:raise ValueError('Seed mismatch')
    return selection,lookup,manifest


def config_hash(cfg):
    import hashlib
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit_result(r,q,cfg):
    if r['config_hash']!=config_hash(cfg) or r['strategy'] not in POLICIES:raise ValueError('Saved config/policy mismatch')
    if any(q[k]!=r[k] for k in ('question_id','question','gold_answer','stratum','calculation_count')):raise ValueError('Saved question mismatch')
    if (r['parallelism'],r['steps'],r['forward_passes'],r['tokens_initially_committed'])!=(1,256,256,256):raise ValueError('Saved P1 budget mismatch')
    if r['position_seed']!=cfg['inference_seed']+q['question_id']:raise ValueError('Saved seed mismatch')
    if len(r['trajectory'])!=256 or len(r['generated_token_ids'])!=256:raise ValueError('Saved trajectory shape mismatch')
    seen=set();state=[cfg['mask_id']]*256
    for step,t in enumerate(r['trajectory']):
        if t['step']!=step or t['forward_passes']!=1 or len(t['positions'])!=1 or len(t['token_ids'])!=1 or len(t['confidences'])!=1 or t.get('revisions'):
            raise ValueError('Saved commit shape/revision mismatch')
        p=t['positions'][0];token=t['token_ids'][0];c=t['confidences'][0]
        if type(p) is not int or p in seen or not 0<=p<256 or token==cfg['mask_id'] or not math.isfinite(c) or not 0<=c<=1:
            raise ValueError('Invalid saved position/token/confidence')
        if r['strategy']=='left_to_right' and p!=step:raise ValueError('Saved LTR order mismatch')
        seen.add(p);state[p]=token
    if seen!=set(range(256)) or state!=r['generated_token_ids']:raise ValueError('Saved coverage/final IDs mismatch')
    pred,method=extract_answer(r['response']);gold,_=extract_answer(q['gold_answer'],gold=True)
    if (pred,method,gold,pred==gold,pred is None)!=(r['predicted_numeric'],r['extraction_method'],r['gold_numeric'],r['correct'],r['extraction_failed']):
        raise ValueError('Saved extraction mismatch')
    eos=r['eos_token_id'];prefix=r['generated_token_ids'][:r['generated_token_ids'].index(eos)] if eos in r['generated_token_ids'] else r['generated_token_ids']
    if r['tokens_before_first_eos']!=len(prefix):raise ValueError('Saved prefix-token count mismatch')
    if type(r['tokens_final_decoded_response']) is not int or r['tokens_final_decoded_response']<0:raise ValueError('Invalid decoded token count')
    for field in ('gpu_seconds','wall_seconds','evaluation_wall_seconds','model_forward_gpu_stream_seconds','sampler_nonforward_main_thread_cpu_seconds','model_forward_host_enqueue_wall_seconds'):
        if not math.isfinite(r[field]) or r[field]<0:raise ValueError('Invalid saved timing')


class ForwardMeter:
    """One transparent delegate call; CUDA events do not add model forwards or step synchronization."""
    def __init__(self,model,device):
        self.model=model;self.device=device;self.calls=0;self.host_wall=0.;self.host_cpu=0.;self.events=[]
    def __call__(self,*args,**kwargs):
        import torch
        gpu=self.device.type=='cuda'
        if gpu:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        start=time.perf_counter();cpu=time.thread_time()
        output=self.model(*args,**kwargs)
        self.host_cpu+=time.thread_time()-cpu;self.host_wall+=time.perf_counter()-start;self.calls+=1
        if gpu:end.record();self.events.append((begin,end))
        return output
    def forward_gpu_seconds(self):
        if not self.events:return None
        self.events[-1][1].synchronize()
        return sum(begin.elapsed_time(end) for begin,end in self.events)/1000


def generate_p1(model,prompt,policy,mask_id,seed):
    if policy not in POLICIES:raise ValueError('Unsupported P1 policy')
    from samplers import generate
    meter=ForwardMeter(model,prompt.device);cpu=time.thread_time()
    result=generate(meter,prompt,policy,gen_length=256,block_length=256,steps=256,mask_id=mask_id,seed=seed)
    nonforward_cpu=time.thread_time()-cpu-meter.host_cpu
    if meter.calls!=256:raise RuntimeError('Expected exactly 256 forwards')
    timing=dict(model_forward_gpu_stream_seconds=meter.forward_gpu_seconds(),
        model_forward_host_enqueue_wall_seconds=meter.host_wall,sampler_nonforward_main_thread_cpu_seconds=nonforward_cpu)
    return (*result,timing)


def read_results(out,selection,cfg,repair=False):
    allowed={(q['question_id'],policy,1) for q in selection['questions'] for policy in POLICIES}
    path=out/'journal.jsonl'
    if repair:records,attempts=recover_journal(path,allowed)
    else:
        records={};attempts={};pending=set()
        if path.exists():
            with path.open() as f:
                for line in f:
                    if not line.endswith('\n'):raise ValueError('Partial journal: resume runner to repair, analysis never repairs')
                    e=json.loads(line);k=(e['question_id'],e['strategy'],e['parallelism'])
                    if k not in allowed:raise ValueError('Unexpected journal key')
                    if e['event']=='started':
                        if k in records:raise ValueError('Restarted completed key')
                        pending.add(k);attempts[k]=attempts.get(k,0)+1
                    elif e['event']=='completed':
                        if k in records or k not in pending:raise ValueError('Duplicate/unstarted completion')
                        if (e['result']['question_id'],e['result']['strategy'],e['result']['parallelism'])!=k:raise ValueError('Event/result key mismatch')
                        records[k]=e['result'];pending.remove(k)
                    else:raise ValueError('Unknown journal event')
    if path.exists():
        with path.open() as f:
            for line in f:
                if json.loads(line).get('config_hash')!=config_hash(cfg):raise ValueError('Journal event config hash mismatch')
    questions={q['question_id']:q for q in selection['questions']}
    for r in records.values():audit_result(r,questions[r['question_id']],cfg)
    return records,attempts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='phase6_config.yaml');parser.add_argument('--phase2-results')
    parser.add_argument('--policy',choices=(*POLICIES,'both'),default='both')
    parser.add_argument('--check-cache',action='store_true');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text());source=Path(args.phase2_results or cfg['phase2_results']).expanduser().resolve()
    selection,lookup,previous=validate(cfg,source);out=Path(cfg['output_dir']).resolve()
    if not out.is_relative_to(Path('outputs/phase6').resolve()) or source==out or source in out.parents or out in source.parents:
        raise ValueError('Independent outputs/phase6 directory required')
    if args.check_cache:cached_model();print('Pinned cached files available; no inference.');return
    if not args.summarize and not os.environ.get('PBS_JOBID'):raise SystemExit('GPU inference only inside PBS')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=('run_phase6.py','analyze_phase6.py','samplers.py','phase2_samplers.py','phase2_data.py','run_phase2.py','evaluation.py','analyze_phase25.py','analyze_phase15.py')
    manifest=dict(config=cfg,config_hash=config_hash(cfg),question_ids=[q['question_id'] for q in selection['questions']],
        selection_sha256=sha(cfg['selection_file']),phase2_journal_sha256=sha(source/'journal.jsonl'),
        phase2_manifest_sha256=sha(source/'manifest.json'),source_sha256={n:sha(n) for n in sources},
        n_evaluations=120,planned_forwards=30720,timing='decode GPU stream + forward GPU stream intervals + nonforward main-thread CPU + end-to-end evaluation wall')
    path=out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('Run source/config/baseline changed: fresh output required')
    elif args.summarize:raise FileNotFoundError('No Phase 6 manifest')
    else:atomic_json(path,manifest)
    records,attempts=read_results(out,selection,cfg,repair=True)
    from analyze_phase6 import analyze
    analyze(out,records,lookup,selection,cfg)
    policies=POLICIES if args.policy=='both' else (args.policy,)
    work={(q['question_id'],s,1) for q in selection['questions'] for s in policies}
    if args.summarize or work.issubset(records):print(f'{len(records)}/120 total; requested policies complete; no inference.');return
    import random,torch
    from transformers import AutoModel,AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required')
    random.seed(cfg['inference_seed']);torch.manual_seed(cfg['inference_seed']);torch.cuda.manual_seed_all(cfg['inference_seed'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    loading=dict(local_files_only=True,trust_remote_code=True);model_path=cached_model()
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),**loading)
    model=AutoModel.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**loading).to('cuda').eval()
    from importlib.metadata import version
    atomic_json(out/f'environment-{os.environ["PBS_JOBID"]}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),transformers=version('transformers'),timing_instrumentation='per-forward CUDA events; no per-step synchronization'))
    with (out/'journal.jsonl').open('a') as journal:
        for i,q in enumerate(selection['questions']):
            order=policies if i%2==0 else policies[::-1]
            for policy in order:
                k=(q['question_id'],policy,1)
                if k in records:continue
                header=dict(question_id=q['question_id'],strategy=policy,parallelism=1,attempt=attempts.get(k,0)+1)
                append_event(journal,dict(event='started',config_hash=config_hash(cfg),**header))
                torch.cuda.synchronize();evaluation_start=time.perf_counter()
                text=tokenizer.apply_chat_template([dict(role='user',content=q['question']+cfg['prompt_suffix'])],tokenize=False,add_generation_prompt=True)
                prompt=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
                if prompt.shape[1]+256>getattr(model.config,'max_position_embeddings',4096):raise ValueError('Context limit exceeded')
                gold,_=extract_answer(q['gold_answer'],gold=True)
                if gold is None:raise ValueError('Gold extraction failure')
                ids,seconds,wall,trace,timing=generate_p1(model,prompt,policy,cfg['mask_id'],cfg['inference_seed']+q['question_id'])
                eos=tokenizer.eos_token_id;prefix=ids[:ids.index(eos)] if eos in ids else ids
                response=tokenizer.decode(prefix,skip_special_tokens=True);raw=tokenizer.decode(ids,skip_special_tokens=False)
                pred,method=extract_answer(response)
                # Count actual re-encoded visible text separately from prefix/special-token counts.
                decoded_count=len(tokenizer(response,add_special_tokens=False).input_ids)
                torch.cuda.synchronize();evaluation_wall=time.perf_counter()-evaluation_start
                r=dict(**q,strategy=policy,parallelism=1,steps=256,config_hash=config_hash(cfg),forward_passes=256,
                    position_seed=cfg['inference_seed']+q['question_id'],gold_numeric=gold,predicted_numeric=pred,
                    correct=pred==gold,extraction_failed=pred is None,extraction_method=method,response=response,raw_response=raw,
                    generated_token_ids=ids,eos_token_id=eos,trajectory=trace,tokens_initially_committed=256,
                    tokens_before_first_eos=len(prefix),tokens_final_decoded_response=decoded_count,
                    gpu_seconds=seconds,wall_seconds=wall,evaluation_wall_seconds=evaluation_wall,**timing)
                audit_result(r,q,cfg);append_event(journal,dict(event='completed',config_hash=config_hash(cfg),result=r,**header));records[k]=r
                analyze(out,records,lookup,selection,cfg)
                print(f'[{len(records)}/120] {policy} q={q["question_id"]} correct={r["correct"]} decode GPU stream={seconds:.2f}s evaluation wall={evaluation_wall:.2f}s forwards=256',flush=True)


if __name__=='__main__':main()
