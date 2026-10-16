"""Strict prospective Phase 9 protocol; opt-in adapter for the existing AR runner."""
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from dllm_unmasking.ar_baseline import (load_config as ar_config,inputs as ar_inputs,digest,object_hash)
from dllm_unmasking.ar_compare import load_run
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.phase2_data import atomic_json
from dllm_unmasking.layout import repository_root

ORIGINAL="Solve this problem step by step. End with '#### <final numerical answer>'."
CONCISE="Solve this problem using only the essential calculation steps. Be concise, avoid unnecessary explanations, and end with '#### <final numerical answer>'."
CONDITION='concise_essential_steps_v1'
OUTPUT_ROOT='outputs/phase9'
DEFAULT_CONFIG='configs/phase9_qwen_concise_256.yaml'
EXTRA_FIELDS={'condition','baseline256_dir','baseline512_dir','phase8_provenance','review_seed'}
SOURCE_NAMES=['experiments/phase9/run.py','src/dllm_unmasking/phase9_protocol.py','src/dllm_unmasking/ar_compare.py']


def validate_config(cfg):
    base=ar_config('configs/phase8_qwen.yaml')
    if set(cfg)!=set(base)|EXTRA_FIELDS:raise ValueError('Unexpected Phase 9 configuration fields')
    projected={k:cfg[k] for k in base}
    if {k for k in base if projected[k]!=base[k]}!={'prompt_suffix','output_dir'}:
        raise ValueError('Only the preselected instruction and output directory may differ from Qwen-256')
    if cfg['prompt_suffix']!='\n'+CONCISE or cfg['condition']!=CONDITION or cfg['review_seed']!=20261015:
        raise ValueError('Fixed prospective instruction/condition/review seed required; no variants')
    out=Path(cfg['output_dir']).resolve();allowed=repository_root()/OUTPUT_ROOT
    if out==allowed or not out.is_relative_to(allowed):raise ValueError('Phase 9 requires its own output subdirectory')
    for name in ('baseline256_dir','baseline512_dir'):
        ref=Path(cfg[name]).resolve()
        if out==ref or ref in out.parents or out in ref.parents:raise ValueError('Output overlaps original reference run')
    return cfg


def load_config(path=None,model=None):
    if model not in (None,'qwen'):raise ValueError('Phase 9 uses only pinned Qwen')
    return validate_config(ar_config(path or DEFAULT_CONFIG,'qwen'))


def references(cfg):
    """All references read-only; hash-match historical Phase 8 receipts."""
    provenance=json.loads(Path(cfg['phase8_provenance']).read_text());runs={};qs=None
    for cap,field in ((256,'baseline256_dir'),(512,'baseline512_dir')):
        path=Path(cfg[field]);c,questions,rs,m=load_run(path)
        for name,sha in provenance['imports'][str(cap)].items():
            if digest(path/name)!=sha:raise ValueError('Historical reference checksum mismatch')
        if c['max_new_tokens']!=cap or c['model_revision']!=cfg['model_revision'] or len(rs)!=60:
            raise ValueError('Incomplete/mismatched historical reference')
        if sum(r['correct'] for r in rs.values())!={256:28,512:50}[cap]:raise ValueError('Historical reference accuracy mismatch')
        if qs is not None and questions!=qs:raise ValueError('Reference question order/content mismatch')
        qs=questions;runs[cap]=(rs,m)
    import csv
    with Path(cfg['diffusion_results']).open() as f:
        rows=[r for r in csv.DictReader(f) if r['method']=='left_to_right P4']
    if len(rows)!=60 or {int(r['question_id']) for r in rows}!={q['question_id'] for q in qs} or sum(r['correct']=='True' for r in rows)!=37:
        raise ValueError('Immutable LLaDA P4 reference mismatch')
    return qs,runs


def inputs(cfg):
    validate_config(cfg)
    # This projection invokes the unchanged historical selection/data validator.
    # It never changes the intervention config passed to prompt_inputs/generate.
    base=ar_config('configs/phase8_qwen.yaml');questions=ar_inputs(base)
    other,runs=references(cfg)
    if questions!=other:raise ValueError('Exact selected question order required')
    return questions


def cache_metadata(cfg,pin,path):
    _,runs=references(cfg);m=runs[256][1]
    if digest(cfg['model_pin'])!=m['model_pin_sha256']:raise ValueError('Historical model pin changed')
    for file,key in [('tokenizer.json','tokenizer_sha256'),('tokenizer_config.json','tokenizer_config_sha256')]:
        if digest(path/file)!=m[key]:raise ValueError('Pinned native tokenizer metadata mismatch')
    for file,key in [('config.json','model_config'),('generation_config.json','native_generation_config')]:
        if json.loads((path/file).read_text())!=m[key]:raise ValueError('Pinned native model configuration mismatch')
    return dict(phase=9,phase8_provenance_sha256=digest(cfg['phase8_provenance']),condition=CONDITION,instruction=CONCISE,instruction_sha256=object_hash(CONCISE),
        reference_manifest_sha256={str(cap):digest(Path(cfg[f'baseline{cap}_dir'])/'manifest.json') for cap in (256,512)},
        reference_journal_sha256={str(cap):digest(Path(cfg[f'baseline{cap}_dir'])/'journal.jsonl') for cap in (256,512)},
        comparison='Prospective single preselected prompt intervention; within Qwen at fixed 256-token cap, not an architecture test')


def validate_environment(cfg,environment):
    original=json.loads((Path(cfg['baseline256_dir'])/'environment.json').read_text())
    if environment!=original:raise ValueError('Require the original Qwen GPU class and software/dtype/attention environment')


def header(cfg):return dict(condition=CONDITION,instruction_sha256=object_hash(CONCISE),configuration_hash=object_hash(cfg))


def validate_records(cfg,records):
    _,runs=references(cfg);base=runs[256][0]
    for qid,r in records.items():
        if r.get('condition')!=CONDITION or r.get('instruction_sha256')!=object_hash(CONCISE) or r.get('configuration_hash')!=object_hash(cfg):
            raise ValueError('Phase 9 condition identity mismatch')
        original=base[qid]['rendered_prompt']
        if original.count('\n'+ORIGINAL)!=1:raise ValueError('Historical instruction placement ambiguous')
        if r['rendered_prompt']!=original.replace('\n'+ORIGINAL,'\n'+CONCISE,1):
            raise ValueError('Only the exact instruction in its original native-template role may change')


def summarize(out,records,questions,cfg):
    out=Path(out);rows=list(records.values())
    if rows:write_csv(out/'results.csv',[{k:v for k,v in r.items() if k not in ('generated_token_ids','rendered_prompt')} for r in rows])
    summary=[]
    for stratum in ('all','0-2','3-4','5+'):
        rs=[r for r in rows if stratum=='all' or r['stratum']==stratum]
        row=dict(condition=CONDITION,stratum=stratum,n=len(rs),expected_n=60 if stratum=='all' else 20,
            correct=sum(r['correct'] for r in rs),accuracy=mean(r['correct'] for r in rs) if rs else None,
            eos=sum(r['termination']=='eos' for r in rs),length_limited=sum(r['termination']=='length_limit' for r in rs),
            final_markers=sum('####' in r['response'] for r in rs),extraction_failures=sum(r['extraction_failed'] for r in rs))
        for field in ('generated_tokens','response_tokens','gpu_prefill_decode_seconds','generation_wall_seconds'):
            values=[r[field] for r in rs]
            for label,value in [('mean',mean(values) if values else None),('median',median(values) if values else None),('p90',float(np.quantile(values,.9,method='linear')) if values else None)]:row[field+'_'+label]=value
        summary.append(row)
    write_csv(out/'summary.csv',summary)
    atomic_json(out/'completion.json',dict(n_completed=len(rows),expected=60,complete=len(rows)==60))
