"""Phase 1: fixed GSM8K subset, five position orders, one resident LLaDA model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

from evaluation import extract_answer


def read_journal(path):
    """Ignore only an interrupted final append; reject corruption in earlier records."""
    if not path.exists():
        return []
    records = []
    with path.open('rb+') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                f.truncate(offset)
                break
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    # Only model inference is restricted; analyze_results.py works on any CPU.
    if not os.environ.get('PBS_JOBID'):
        raise SystemExit('Inference must run inside a PBS GPU job, never on a login node.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    from samplers import STRATEGIES, generate, transfer_schedule
    from analyze_results import analyze

    cfg = yaml.safe_load(Path(args.config).read_text())
    transfer_schedule(**{k: cfg['decoding'][k] for k in ('gen_length', 'steps', 'block_length')})
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU unavailable in this PBS allocation.')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, cfg['model']['dtype'])
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('dtype must be bfloat16, float16 or float32')
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit('BF16 unavailable; explicitly change config for all strategies.')
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent jobs from appending to the same experiment.
    import fcntl
    lock = (out / '.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in ('run_experiment.py', 'samplers.py', 'evaluation.py', 'analyze_results.py')}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != cfg or manifest['source_hashes'] != source_hashes:
            raise SystemExit('Configuration/code changed: use a new output_dir.')
    else:
        api = HfApi()
        def resolve_revision(name, revision, dataset=False):
            if re.fullmatch(r'[0-9a-f]{40}', revision):
                return revision
            info = api.dataset_info if dataset else api.model_info
            return info(name, revision=revision).sha
        manifest = dict(config=cfg, source_hashes=source_hashes,
                        model_revision=resolve_revision(cfg['model']['name'], cfg['model']['revision']),
                        dataset_revision=resolve_revision(cfg['dataset']['name'], cfg['dataset']['revision'], dataset=True),
                        created_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=2))
    ds_cfg = cfg['dataset']
    ds = load_dataset(ds_cfg['name'], ds_cfg['subset'], split=ds_cfg['split'],
                      revision=manifest['dataset_revision'])
    n = ds_cfg['num_questions']
    if not 0 < n <= len(ds):
        raise ValueError('Invalid num_questions')
    ids = random.Random(cfg['seed']).sample(range(len(ds)), n)
    questions = [dict(question_id=i, question=ds[i]['question'], gold_answer=ds[i]['answer']) for i in ids]
    question_path = out / 'questions.json'
    if question_path.exists() and json.loads(question_path.read_text()) != questions:
        raise SystemExit('Dataset subset differs from saved questions')
    question_path.write_text(json.dumps(questions, indent=2))
    journal = out / 'records.jsonl'
    records = read_journal(journal)
    done = {(r['question_id'], r['strategy']) for r in records}
    if len(done) != len(records):
        raise ValueError('Duplicate journal records')
    if len(done) == n * len(STRATEGIES):
        analyze(out)
        print('All evaluations already complete.', flush=True)
        return
    load_args = dict(revision=manifest['model_revision'], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], **load_args)
    model = AutoModel.from_pretrained(cfg['model']['name'], torch_dtype=dtype,
                                     low_cpu_mem_usage=True, **load_args).to('cuda').eval()
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                   pbs_job_id=os.environ['PBS_JOBID'],
                   packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    with (out / 'environments.jsonl').open('a') as f:
        f.write(json.dumps(runtime) + '\n')

    def encode(question):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question + cfg['prompt_suffix']}],
            tokenize=False, add_generation_prompt=True)
        prompt = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
        limit = getattr(model.config, 'max_position_embeddings', 4096)
        if prompt.shape[1] + cfg['decoding']['gen_length'] > limit:
            raise ValueError('Prompt and generation exceed model context; no silent truncation')
        return prompt

    # Warm up the actual sampler at full shape once, excluded from reported metrics.
    print('Model loaded; warming up sampler.', flush=True)
    generate(model, encode(questions[0]['question']), 'left_to_right',
             **cfg['decoding'], seed=cfg['seed'])
    with journal.open('a') as f:
        for index, item in enumerate(questions):
            prompt = encode(item['question'])
            gold, _ = extract_answer(item['gold_answer'], gold=True)
            if gold is None:
                raise ValueError('Unparseable GSM8K ground truth')
            # Rotate execution order to reduce systematic runtime/order bias.
            shift = index % len(STRATEGIES)
            order = STRATEGIES[shift:] + STRATEGIES[:shift]
            for strategy in order:
                if (item['question_id'], strategy) in done:
                    continue
                position_seed = cfg['seed'] + item['question_id']
                ids_out, seconds, wall, trace = generate(
                    model, prompt, strategy, **cfg['decoding'], seed=position_seed)
                # EOS tokens remain in the trajectory; evaluation sees text before first EOS.
                eos = tokenizer.eos_token_id
                visible_ids = ids_out[:ids_out.index(eos)] if eos in ids_out else ids_out
                response = tokenizer.decode(visible_ids, skip_special_tokens=True)
                predicted, method = extract_answer(response)
                confidences = [c for step in trace for c in step['confidences']]
                record = dict(**item, strategy=strategy, gold_numeric=gold,
                              predicted_numeric=predicted, extraction_method=method,
                              extraction_failed=predicted is None, correct=predicted == gold,
                              gpu_seconds=seconds, wall_seconds=wall,
                              forward_passes=sum(step['forward_passes'] for step in trace),
                              influence_top_k=(cfg['decoding'].get('influence_top_k') if strategy == 'influence' else None),
                              influence_approximation=(trace[0]['influence_approximation'] if strategy == 'influence' else ''),
                              mean_token_confidence=sum(confidences)/len(confidences),
                              response=response, raw_response=tokenizer.decode(ids_out, skip_special_tokens=False),
                              generated_token_ids=ids_out, position_seed=position_seed,
                              trajectory=trace)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                records.append(record)
                done.add((item['question_id'], strategy))
                analyze(out, records, plots=False)
                print(f'[{len(done)}/{n * len(STRATEGIES)}] question={item["question_id"]} '
                      f'{strategy}: correct={record["correct"]}, GPU={seconds:.2f}s, '
                      f'forwards={record["forward_passes"]}', flush=True)
    analyze(out, records)


if __name__ == '__main__':
    main()
