"""Phase 2 full experiment or CPU selection/cache preparation. Never downloads."""
from dllm_unmasking.layout import enter_repository
import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
import sys

# Set before importing any Hugging Face library, including during cache preparation.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.phase2_data import (MODEL,MODEL_REVISION,DATASET_REVISION,STRATA,sha,atomic_json,
                         exclusions,offline_dataset,selection_manifest,key,recover_journal,append_event)

STRATEGIES=('random','left_to_right','right_to_left','confidence','batched_influence')


def read_config(path):
    import yaml
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['model'],cfg['model_revision'],cfg['dataset_revision'])!=(MODEL,MODEL_REVISION,DATASET_REVISION):
        raise ValueError('Model/dataset must match pinned Phase 1 revisions')
    if cfg['settings']!=[dict(parallelism=4,steps=64),dict(parallelism=8,steps=32)]:
        raise ValueError('Require precisely parallel-4/64 steps and parallel-8/32 steps')
    if (cfg['gen_length'],cfg['block_length'],cfg['influence_top_k'],cfg['dtype'],cfg['mask_id'])!=(256,256,16,'bfloat16',126336):
        raise ValueError('Require 256 tokens/full block, top-16 influence and BF16')
    return cfg


def cached_model():
    from huggingface_hub import snapshot_download
    try:
        path=Path(snapshot_download(MODEL,revision=MODEL_REVISION,local_files_only=True))
        for name in ('config.json','tokenizer_config.json','tokenizer.json','configuration_llada.py','modeling_llada.py'):
            if not (path/name).is_file():raise FileNotFoundError(name)
        indexes=list(path.glob('*.index.json'))
        if indexes:
            weights={name for index in indexes for name in json.loads(index.read_text())['weight_map'].values()}
            if not weights or any(not (path/name).is_file() for name in weights):
                raise FileNotFoundError('Missing weight shard')
        elif not any(path.glob('*.safetensors')) and not (path/'pytorch_model.bin').is_file():
            raise FileNotFoundError('Missing model weights')
        return path
    except Exception as e:
        raise RuntimeError(f'Pinned model/tokenizer snapshot {MODEL}@{MODEL_REVISION} unavailable or incomplete '
                           'in existing HF cache. Downloads are disabled.') from e


def csv_atomic(path,rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def summarize(out,records):
    if not records:return
    rows=list(records.values())
    fields=('question_id','stratum','calculation_count','strategy','parallelism','steps','correct',
            'gold_numeric','predicted_numeric','extraction_failed','extraction_method',
            'gpu_seconds','wall_seconds','forward_passes','position_seed','response','raw_response')
    csv_atomic(out/'results.csv',[{k:r[k] for k in fields} for r in rows])
    summary=[]
    for p in (4,8):
        for s in STRATEGIES:
            for stratum in ('all',*STRATA):
                selected=[r for r in rows if r['parallelism']==p and r['strategy']==s and (stratum=='all' or r['stratum']==stratum)]
                n=len(selected)
                summary.append(dict(parallelism=p,strategy=s,stratum=stratum,n=n,
                    expected_n=60 if stratum=='all' else 20,correct=sum(r['correct'] for r in selected),
                    accuracy=mean(r['correct'] for r in selected) if n else None,
                    extraction_failures=sum(r['extraction_failed'] for r in selected),
                    mean_gpu_seconds=mean(r['gpu_seconds'] for r in selected) if n else None,
                    total_forward_passes=sum(r['forward_passes'] for r in selected)))
    csv_atomic(out/'summary.csv',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/phase2.yaml')
    parser.add_argument('--dataset-parquet',help='Existing cached pinned test Parquet; verified by SHA256')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-selection',action='store_true',help='CPU only, save/verify selection; no model access')
    modes.add_argument('--check-cache',action='store_true',help='CPU only, verify dataset and model files; no weight loading/inference')
    modes.add_argument('--summarize',action='store_true',help='CPU only, rebuild summaries from own journal')
    args=parser.parse_args();cfg=read_config(args.config)
    excluded=exclusions(cfg['phase1_ids_file']);selection_path=Path(cfg['selection_file'])
    if args.summarize:
        selection=json.loads(selection_path.read_text())
    else:
        rows=offline_dataset(cfg['dataset_test_parquet_sha256'],args.dataset_parquet)
        selection=selection_manifest(rows,excluded,cfg)
        if selection_path.exists():
            if json.loads(selection_path.read_text())!=selection:
                raise ValueError('Saved selection differs from pinned dataset/seed/exclusions; refusing overwrite')
        elif args.prepare_selection:
            atomic_json(selection_path,selection)
        else:raise FileNotFoundError('Prepare selection before inference with --prepare-selection')
    counts={s:sum(q['stratum']==s for q in selection['questions']) for s in STRATA}
    if counts!={s:20 for s in STRATA} or selection['excluded_phase1_ids']!=excluded:
        raise ValueError('Selection/exclusion mismatch')
    print(f'Selection verified: {counts}; no Phase 1 overlap; seed={cfg["selection_seed"]}',flush=True)
    if args.prepare_selection:return
    if args.check_cache:
        cached_model();print('Pinned dataset and complete model snapshot present; no inference performed.',flush=True);return
    if not args.summarize and not os.environ.get('PBS_JOBID'):
        raise SystemExit('Inference requires a PBS GPU allocation; no login-node inference')
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    allowed={(q['question_id'],s,p) for q in selection['questions'] for s in STRATEGIES for p in (4,8)}
    source_names=('experiments/phase2/run.py','src/dllm_unmasking/phase2_data.py','src/dllm_unmasking/phase2_samplers.py','src/dllm_unmasking/evaluation.py')
    manifest=dict(config=cfg,selection_sha256=sha(selection_path),exclusions_sha256=sha(cfg['phase1_ids_file']),
                  source_sha256={n:sha(n) for n in source_names},evaluations=600,
                  policy='one_common_state_per_step_simultaneous_commit_serial_counterfactual_probes',
                  cold_start_note='No model pilot or warm-up is run; first measured evaluation may include initialization costs.')
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:raise ValueError('Run configuration/source changed; use fresh output_dir')
    elif not args.summarize:atomic_json(manifest_path,manifest)
    records,attempts=recover_journal(out/'journal.jsonl',allowed)
    summarize(out,records)
    if args.summarize or len(records)==600:
        print(f'Completed evaluations: {len(records)}/600',flush=True);return
    model_path=cached_model()
    import random
    import torch
    from transformers import AutoModel,AutoTokenizer
    from dllm_unmasking.phase2_samplers import generate,expected_passes
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Allocated CUDA GPU with BF16 required')
    torch.set_num_threads(4);random.seed(cfg['inference_seed']);torch.manual_seed(cfg['inference_seed'])
    torch.cuda.manual_seed_all(cfg['inference_seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    loading=dict(local_files_only=True,trust_remote_code=True)
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),**loading)
    model=AutoModel.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**loading).to('cuda').eval()
    from importlib.metadata import version
    with (out/'environments.jsonl').open('a') as f:
        f.write(json.dumps(dict(pbs_job_id=os.environ['PBS_JOBID'],gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,cuda=torch.version.cuda,transformers=version('transformers'),
            huggingface_hub=version('huggingface-hub')))+'\n')
    print('Pinned cached model loaded; starting full experiment without pilot or sequential decoding.',flush=True)
    with (out/'journal.jsonl').open('a') as journal:
        for qi,q in enumerate(selection['questions']):
            text=tokenizer.apply_chat_template([dict(role='user',content=q['question']+cfg['prompt_suffix'])],
                                                tokenize=False,add_generation_prompt=True)
            prompt=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            limit=getattr(model.config,'max_position_embeddings',4096)
            if prompt.shape[1]+256>limit:raise ValueError('Context limit exceeded; no truncation')
            gold,_=extract_answer(q['gold_answer'],gold=True)
            if gold is None:raise ValueError('Gold answer extraction failed')
            # Rotate all ten condition orders; keep each question paired across both settings.
            conditions=[(s,setting) for setting in cfg['settings'] for s in STRATEGIES]
            shift=qi%10;conditions=conditions[shift:]+conditions[:shift]
            for strategy,setting in conditions:
                p=setting['parallelism'];k=(q['question_id'],strategy,p)
                if k in records:continue
                attempt=attempts.get(k,0)+1;attempts[k]=attempt
                header=dict(question_id=q['question_id'],strategy=strategy,parallelism=p,attempt=attempt)
                append_event(journal,dict(event='started',**header))
                ids,seconds,wall,trace=generate(model,prompt,strategy,steps=setting['steps'],mask_id=cfg['mask_id'],
                                               seed=cfg['inference_seed']+q['question_id'])
                forward_passes=sum(t['forward_passes'] for t in trace)
                if forward_passes!=expected_passes(setting['steps'],strategy):raise RuntimeError('Unexpected forward-pass count')
                eos=tokenizer.eos_token_id
                visible=ids[:ids.index(eos)] if eos in ids else ids
                response=tokenizer.decode(visible,skip_special_tokens=True)
                predicted,method=extract_answer(response)
                record=dict(**q,strategy=strategy,parallelism=p,steps=setting['steps'],gold_numeric=gold,
                    predicted_numeric=predicted,correct=predicted==gold,extraction_failed=predicted is None,
                    extraction_method=method,gpu_seconds=seconds,wall_seconds=wall,forward_passes=forward_passes,
                    position_seed=cfg['inference_seed']+q['question_id'],response=response,
                    raw_response=tokenizer.decode(ids,skip_special_tokens=False),generated_token_ids=ids,
                    eos_token_id=eos,trajectory=trace)
                append_event(journal,dict(event='completed',result=record,**header))
                records[k]=record;summarize(out,records)
                print(f'[{len(records)}/600] q={q["question_id"]} stratum={q["stratum"]} {strategy} '
                      f'parallel-{p}: correct={record["correct"]} GPU={seconds:.2f}s forwards={forward_passes}',flush=True)
    print('Full Phase 2 complete: 600/600. Summary saved.',flush=True)


if __name__=='__main__':main()
