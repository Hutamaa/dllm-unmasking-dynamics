"""Deterministic selection, offline cache checks, and durable Phase 2 journal."""
from dllm_unmasking.layout import source_path
import hashlib
import json
import os
from pathlib import Path
import random
import re

MODEL = 'GSAI-ML/LLaDA-8B-Instruct'
MODEL_REVISION = '852152c60bdfed4632635bfcf3e83408ad3f65bf'
DATASET_REVISION = '740312add88f781978c0658806c59bc2815b9866'
STRATA = ('0-2','3-4','5+')


def sha(path):
    return hashlib.sha256(source_path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def exclusions(path):
    value=json.loads(Path(path).read_text())
    ids=value['question_ids'] if isinstance(value,dict) else [q['question_id'] for q in value]
    if len(ids)!=150 or len(set(ids))!=150 or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('Require exactly 150 unique Phase 1 test question IDs')
    return sorted(ids)


def choose_questions(rows,excluded,seed):
    if len(excluded)!=150 or len(set(excluded))!=150:
        raise ValueError('Invalid Phase 1 exclusions')
    buckets={s:[] for s in STRATA};excluded=set(excluded)
    if any(i>=len(rows) for i in excluded):raise ValueError('Phase 1 IDs outside test split')
    for i,row in enumerate(rows):
        if i in excluded:continue
        count=len(re.findall(r'<<.*?>>',row['answer'],re.S))
        stratum='0-2' if count<=2 else '3-4' if count<=4 else '5+'
        buckets[stratum].append(dict(question_id=i,question=row['question'],gold_answer=row['answer'],
                                    calculation_count=count,stratum=stratum))
    rng=random.Random(seed);selected=[]
    for s in STRATA:
        if len(buckets[s])<20:raise ValueError(f'Insufficient unseen questions in {s}: {len(buckets[s])}')
        selected.extend(sorted(rng.sample(buckets[s],20),key=lambda q:q['question_id']))
    if len({q['question_id'] for q in selected})!=60 or {q['question_id'] for q in selected}&excluded:
        raise ValueError('Selection overlap/duplicate')
    return selected


def parquet_rows(path,expected_sha):
    if sha(path)!=expected_sha:
        raise ValueError('Cached dataset bytes do not match pinned GSM8K test Parquet SHA-256')
    import pyarrow.parquet as pq
    rows=pq.read_table(path).to_pylist()
    if len(rows)!=1319:raise ValueError('Expected 1319 GSM8K main test questions')
    return rows


def offline_dataset(expected_sha,explicit=None):
    """Find existing pinned raw Parquet; never fetch or silently choose another revision."""
    if explicit:
        return parquet_rows(explicit,expected_sha)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    from huggingface_hub import hf_hub_download
    try:
        path=hf_hub_download('openai/gsm8k','main/test-00000-of-00001.parquet',
                            repo_type='dataset',revision=DATASET_REVISION,local_files_only=True)
        return parquet_rows(path,expected_sha)
    except (FileNotFoundError,OSError) as e:
        missing=e
    # datasets 3.x may store downloaded Parquet in its downloads cache instead of hub snapshots.
    home=Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))
    cache=Path(os.environ.get('HF_DATASETS_CACHE',home/'datasets'))/'downloads'
    for meta in sorted(cache.glob('*.json')):
        try:
            url=json.loads(meta.read_text()).get('url','')
        except (ValueError,OSError):continue
        if 'gsm8k' in url and 'main/test-00000-of-00001.parquet' in url:
            file=meta.with_suffix('')
            if file.is_file() and sha(file)==expected_sha:return parquet_rows(file,expected_sha)
    raise FileNotFoundError('Pinned GSM8K cache unavailable. Set --dataset-parquet to an EXISTING cached '
                            'main/test Parquet file with the documented hash. Downloads are disabled.') from missing


def selection_manifest(rows,excluded,cfg):
    return dict(dataset='openai/gsm8k',dataset_revision=DATASET_REVISION,subset='main',split='test',
                dataset_test_parquet_sha256=cfg['dataset_test_parquet_sha256'],selection_seed=cfg['selection_seed'],
                excluded_phase1_ids=excluded,questions=choose_questions(rows,excluded,cfg['selection_seed']))


def key(record):
    return record['question_id'],record['strategy'],record['parallelism']


def recover_journal(path,allowed):
    """Only truncate an incomplete final append; reject corrupt/duplicate completed records."""
    path=Path(path);records={};attempts={}
    if not path.exists():return records,attempts
    with path.open('rb+') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            if not line.endswith(b'\n'):
                f.truncate(offset);f.flush();os.fsync(f.fileno());break
            event=json.loads(line);k=key(event)
            if k not in allowed:raise ValueError('Unexpected journal evaluation key')
            if event['event']=='started':
                if k in records:raise ValueError('Started an already completed evaluation')
                attempts[k]=attempts.get(k,0)+1
            elif event['event']=='completed':
                if k in records or not attempts.get(k):raise ValueError('Duplicate completion or missing write-ahead start')
                records[k]=event['result']
                if key(event['result'])!=k:raise ValueError('Journal key/result mismatch')
            else:raise ValueError('Unknown journal event')
    return records,attempts


def append_event(file,event):
    file.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
    file.flush();os.fsync(file.fileno())
