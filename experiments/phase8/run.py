"""Explicit cache/access/download operations or restart-safe matched AR evaluation."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

from dllm_unmasking.layout import enter_repository, repository_root
from dllm_unmasking.ar_baseline import (model_spec, load_config, inputs, cache_root, cache_check, snapshot, complete_snapshot,
    check_access, digest, object_hash, prompt_inputs, generate_one, validate_result, read_journal)
from dllm_unmasking.ar_analysis import summarize, baseline
from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.phase2_data import atomic_json, append_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Phase 8 YAML (default: Qwen config)')
    parser.add_argument('--model', choices=('qwen','llama'), help='Select the corresponding model config')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-cache', '--preflight', action='store_true', dest='check_cache')
    modes.add_argument('--check-access', action='store_true')
    modes.add_argument('--resolve-revision', action='store_true', help='Explicit online metadata/HEAD check; write immutable pin, never weights')
    modes.add_argument('--pin-cached-revision', metavar='SHA', help='Pin an existing complete local immutable snapshot, offline')
    modes.add_argument('--download-model', action='store_true', help='Explicit USER action only; download the already pinned revision into AR cache')
    modes.add_argument('--summarize', action='store_true', help='CPU rebuild of own derived tables; journal stays read-only')
    args = parser.parse_args()
    cfg = load_config(args.config, args.model)
    model_id = cfg['model']; policy = model_spec(model_id)['policy']
    questions = inputs(cfg); baseline(cfg, questions)
    pin_path = Path(cfg['model_pin']); pin = json.loads(pin_path.read_text()) if pin_path.exists() else None
    if pin and (pin['model'] != model_id or pin['revision'] != pin['tokenizer_revision'] or pin['revision'] != cfg['model_revision']):
        raise ValueError('Model/tokenizer pin mismatch')
    if args.check_cache:
        print(json.dumps(dict(model=model_id, questions=60, strata={'0-2':20, '3-4':20, '5+':20}, **cache_check(cfg, pin)), indent=2)); return
    if args.check_access or args.resolve_revision:
        state = check_access(cfg, pin['revision'] if pin else cfg['model_revision'])
        state['model'] = model_id
        # No credentials or exception strings are emitted.
        print(json.dumps({k:v for k,v in state.items() if k != 'pin'}, indent=2))
        if args.resolve_revision:
            if 'pin' not in state:
                raise SystemExit('Cannot pin model: authorized weight access was not verified')
            if pin and pin != state['pin']:
                raise ValueError('Existing pin differs; use an independent Phase 8 configuration, never overwrite')
            if not pin:
                atomic_json(pin_path, state['pin'])
        return
    if args.pin_cached_revision:
        if args.pin_cached_revision != cfg['model_revision']:
            raise ValueError('Cached revision must match the configured verified revision')
        path = snapshot(cache_root(cfg), args.pin_cached_revision, model_id)
        if not complete_snapshot(path, model=model_id):
            raise SystemExit('Pinned cache snapshot incomplete; no download attempted')
        value = dict(model=model_id, revision=args.pin_cached_revision, tokenizer_revision=args.pin_cached_revision,
                     file_sizes={p.name:p.stat().st_size for p in path.iterdir() if p.is_file()})
        if pin and pin != value:
            raise ValueError('Existing pin differs; refusing overwrite')
        if not pin:
            atomic_json(pin_path, value)
        print('Existing complete immutable cache snapshot pinned; no network or inference.'); return
    if not pin:
        raise SystemExit('Resolve and pin an actual model revision first; no revision is invented or substituted')
    path = snapshot(cache_root(cfg), pin['revision'], model_id)
    if args.download_model:
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(model_id, revision=pin['revision'], cache_dir=str(cache_root(cfg)),
                              allow_patterns=['*.json', '*.safetensors', '*.model', '*.tiktoken', 'merges.txt'],
                              token=None if model_spec(model_id)['gated'] else False)
        except Exception:
            raise SystemExit('Pinned model download failed; verify access/cache privately. No alternative model used.') from None
        if not complete_snapshot(path, pin.get('file_sizes'), model_id):
            raise SystemExit('Downloaded snapshot incomplete; refusing evaluation')
        print('Pinned AR model cached. No inference performed.'); return
    if not complete_snapshot(path, pin.get('file_sizes'), model_id):
        raise SystemExit('Pinned AR model cache incomplete. No automatic download; use explicit user download step.')
    out = Path(cfg['output_dir']).resolve()
    allowed = (repository_root() / 'outputs/phase8').resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Use an independent run subdirectory under outputs/phase8')
    if out in path.resolve().parents or path.resolve() in out.parents or out == path.resolve():
        raise ValueError('Model cache and output directories cannot overlap')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / '.run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_names = ['experiments/phase8/run.py', 'src/dllm_unmasking/ar_baseline.py', 'src/dllm_unmasking/ar_analysis.py',
                    'src/dllm_unmasking/evaluation.py', 'src/dllm_unmasking/phase2_data.py', 'src/dllm_unmasking/analyze_phase7.py', 'src/dllm_unmasking/layout.py']
    manifest = dict(config=cfg, model=model_id, model_revision=pin['revision'], tokenizer_revision=pin['tokenizer_revision'],
                    selection_sha256=digest(cfg['selection_file']), diffusion_sha256=digest(cfg['diffusion_results']),
                    public_selection_sha256=digest(cfg['public_selection']), phase7_provenance_sha256=digest(cfg['phase7_provenance']),
                    model_pin_sha256=digest(pin_path), model_config=json.loads((path/'config.json').read_text()),
                    native_generation_config=json.loads((path/'generation_config.json').read_text()),
                    tokenizer_config_sha256=digest(path/'tokenizer_config.json'), tokenizer_sha256=digest(path/'tokenizer.json'),
                    chat_template=json.loads((path/'tokenizer_config.json').read_text())['chat_template'],
                    source_sha256={n:digest(n) for n in source_names}, n_evaluations=60,
                    comparison='System-level; architecture, training, tokenizer and instruction tuning are confounded')
    run_hash = object_hash(manifest); manifest['run_hash'] = run_hash
    saved = out/'manifest.json'
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError('Run model/config/code/input hashes differ; use a fresh output directory')
    if args.summarize and not saved.exists():
        raise ValueError('No existing manifest to summarize')
    if not args.summarize and not os.environ.get('PBS_JOBID'):
        raise SystemExit('Inference requires a PBS GPU allocation; use --check-cache locally')
    if not saved.exists():
        atomic_json(saved, manifest)
    records = read_journal(out/'journal.jsonl', questions, run_hash, repair=not args.summarize, model=model_id, revision=pin['revision'], cap=cfg['max_new_tokens'])
    summarize(out, records, questions, cfg)
    if args.summarize or len(records) == 60:
        print(f'{len(records)}/60 validated completed evaluations'); return
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_DATASETS_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import random
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from importlib.metadata import version
    if transformers.__version__ != cfg['transformers_version']:
        raise RuntimeError('Use the separate pinned Phase 8 Transformers environment; do not upgrade historical diffusion env')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Allocated BF16-capable CUDA GPU required')
    environment = dict(python=sys.version.split()[0], platform=platform.system(), torch=torch.__version__, cuda=torch.version.cuda,
                       transformers=transformers.__version__, huggingface_hub=version('huggingface-hub'), numpy=version('numpy'), accelerate=version('accelerate'), tokenizers=version('tokenizers'), PyYAML=version('PyYAML'), jinja2=version('jinja2'),
                       gpu=torch.cuda.get_device_name(), attention_implementation=cfg['attention_implementation'], dtype=cfg['dtype'])
    env_path = out/'environment.json'
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError('Resume environment differs; use an independent output directory')
    atomic_json(env_path, environment)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.bfloat16, attn_implementation=cfg['attention_implementation']).to('cuda').eval()
    if model.config.model_type != model_spec(model_id)['model_type']:
        raise ValueError('Loaded backend model type differs from the configured baseline')
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or any(type(t) is not int or t < 0 for t in eos) or not tokenizer.chat_template:
        raise ValueError('Native EOS IDs and official tokenizer chat template required')
    with (out/'journal.jsonl').open('a') as journal:
        for q in questions:
            qid = q['question_id']
            if qid in records:
                continue
            seed = cfg['inference_seed']
            random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            header = dict(question_id=qid, policy=policy, run_hash=run_hash, model=model_id, model_revision=pin['revision'], max_new_tokens=cfg['max_new_tokens'])
            append_event(journal, dict(event='started', **header))
            text, encoded = prompt_inputs(tokenizer, q['question'], cfg['prompt_suffix'])
            encoded = {k:v.to('cuda') for k,v in encoded.items()}
            if encoded['input_ids'].shape[1]+cfg['max_new_tokens'] > model.config.max_position_embeddings:
                raise ValueError('Context limit exceeded; no truncation permitted')
            result = generate_one(model, tokenizer, encoded, eos, cap=cfg['max_new_tokens'])
            predicted, method = extract_answer(result['response']); gold = extract_answer(q['gold_answer'], gold=True)[0]
            result.update(**header, stratum=q['stratum'], calculation_count=q['calculation_count'], inference_seed=seed,
                          gold_numeric=gold, predicted_numeric=predicted, extraction_method=method,
                          extraction_failed=predicted is None, correct=predicted==gold,
                          eos_token_ids=eos, prompt_tokens=encoded['input_ids'].shape[1], rendered_prompt=text)
            validate_result(result, q, run_hash, model=model_id, revision=pin['revision'], cap=cfg['max_new_tokens'])
            append_event(journal, dict(event='completed', result=result, **header))
            records[qid] = result; summarize(out, records, questions, cfg)
            print(f'Phase 8 {policy}: {len(records)}/60 completed; correct={sum(r["correct"] for r in records.values())}/{len(records)}', flush=True)


if __name__ == '__main__':
    enter_repository()
    main()
