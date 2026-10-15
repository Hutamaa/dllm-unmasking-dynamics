"""Phase 8 cache, exact input validation, native greedy generation and journal audit."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.layout import repository_root
from dllm_unmasking.phase2_data import atomic_json, DATASET_REVISION

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
POLICY = 'llama_greedy'  # Legacy Llama API/schema default.
MODEL_SPECS = {
    MODEL: dict(alias='llama', policy=POLICY, gated=True, model_type='llama'),
    'Qwen/Qwen2.5-7B-Instruct': dict(alias='qwen', policy='qwen_greedy', gated=False, model_type='qwen2'),
}
CONFIGS = {'qwen': 'configs/phase8_qwen.yaml', 'llama': 'configs/phase8_llama.yaml'}


def model_spec(model):
    if model not in MODEL_SPECS:
        raise ValueError('Unsupported AR model; no automatic substitution')
    return MODEL_SPECS[model]


def load_config(config=None, model=None):
    import yaml
    cfg = yaml.safe_load(Path(config or CONFIGS[model or 'qwen']).read_text())
    spec = model_spec(cfg['model'])
    if model is not None and spec['alias'] != model:
        raise ValueError('--model and --config select different models')
    return cfg



def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inputs(cfg):
    model_spec(cfg['model'])
    if (cfg['max_new_tokens'], cfg['do_sample'], cfg['use_cache'], cfg['batch_size'], cfg['dtype']) != (256, False, True, 1, 'bfloat16'):
        raise ValueError('Require the specified model, greedy AR baseline controls, BF16, batch one and 256-token cap')
    if cfg['inference_seed'] != 42 or cfg['prompt_suffix'] != "\nSolve this problem step by step. End with '#### <final numerical answer>'.":
        raise ValueError('Exact historical seed and instruction suffix required')
    if not re.fullmatch('[0-9a-f]{40}', cfg['model_revision']):
        raise ValueError('An immutable model revision is required')
    public = json.loads(Path(cfg['public_selection']).read_text())
    selection = json.loads(Path(cfg['selection_file']).read_text())
    if digest(cfg['selection_file']) != public['selection_sha256'] or selection['dataset_revision'] != DATASET_REVISION:
        raise ValueError('Exact original Phase 2 selection bytes and dataset revision required')
    questions = selection['questions']
    projected = [{k: q[k] for k in ('question_id', 'stratum', 'calculation_count')} for q in questions]
    if projected != public['questions'] or len(questions) != 60 or len({q['question_id'] for q in questions}) != 60:
        raise ValueError('Require all 60 matched questions in original order')
    if any(sum(q['stratum'] == s for q in questions) != 20 for s in ('0-2', '3-4', '5+')):
        raise ValueError('Expected 20 questions per complexity stratum')
    for q in questions:
        if extract_answer(q['gold_answer'], gold=True)[0] is None:
            raise ValueError('Invalid original gold answer')
    provenance = json.loads(Path(cfg['phase7_provenance']).read_text())
    if digest(cfg['public_selection']) != provenance['input_sha256']['benchmark_selection.json']:
        raise ValueError('Public selection index hash mismatch')
    if digest(cfg['diffusion_results']) != provenance['input_sha256']['unified_per_question.csv']:
        raise ValueError('Phase 7 results hash mismatch')
    if (cfg['bootstrap_seed'], cfg['bootstrap_repetitions']) != (provenance['seed'], provenance['bootstrap_repetitions']):
        raise ValueError('Use finalized Phase 7 bootstrap seed and repetitions')
    return questions


def cache_root(cfg):
    value = os.environ.get(cfg['cache_environment'])
    if not value:
        raise ValueError('Set PHASE8_HUB_CACHE to a separate AR Hub cache directory')
    cache = Path(value).expanduser().resolve()
    if cache.is_relative_to(repository_root()):
        raise ValueError('Keep the AR weight cache outside the source checkout, in scratch storage')
    legacy = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser() / 'hub'))).resolve()
    if cache == legacy or cache in legacy.parents or legacy in cache.parents:
        raise ValueError('AR cache must be separate from the existing LLaDA Hub cache')
    return cache


def snapshot(cache, revision, model=MODEL):
    if not re.fullmatch('[0-9a-f]{40}', revision or ''):
        raise ValueError('An actual immutable 40-character Hub revision is required')
    model_spec(model)
    return Path(cache) / ('models--' + model.replace('/', '--')) / 'snapshots' / revision


def complete_snapshot(path, sizes=None, model=None):
    """Read files only; no tokenizer/model instantiation or downloads."""
    try:
        for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'):
            if not (path / name).is_file():
                return False
            json.loads((path / name).read_text())
        if model is not None:
            config = json.loads((path/'config.json').read_text())
            if not isinstance(config, dict) or config.get('model_type') != model_spec(model)['model_type']:
                return False
        index = path / 'model.safetensors.index.json'
        weights = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if not weights or any(not (path / w).is_file() or (path / w).stat().st_size <= 0 for w in weights):
            return False
        if sizes and any((path / w).stat().st_size != sizes[w] for w in weights if w in sizes):
            return False
        # Inspect only safetensors headers, never materialize weights. Reject LFS
        # pointer stubs and tensors whose declared payload is truncated.
        for weight in weights:
            file = path / weight
            with file.open('rb') as handle:
                length = int.from_bytes(handle.read(8), 'little')
                if not 2 <= length <= min(file.stat().st_size - 8, 16 * 1024 * 1024):
                    return False
                header = json.loads(handle.read(length))
            if not isinstance(header, dict):
                return False
            entries = [v for k, v in header.items() if k != '__metadata__']
            body_size = file.stat().st_size - 8 - length
            if not entries or any(not isinstance(v, dict) or len(v.get('data_offsets', [])) != 2
                                  or not 0 <= v['data_offsets'][0] <= v['data_offsets'][1] <= body_size for v in entries):
                return False
        token_config = json.loads((path / 'tokenizer_config.json').read_text())
        return isinstance(token_config, dict) and bool(token_config.get('chat_template'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cache_check(cfg, pin=None):
    cache = cache_root(cfg)
    model = cfg.get('model', MODEL)
    model_spec(model)
    if pin:
        if pin['model'] != model:
            raise ValueError('Pinned model mismatch')
        revision = pin['revision']
        if cfg.get('model_revision') and cfg['model_revision'] != revision:
            raise ValueError('Pinned revision does not match the selected configuration')
    else:
        ref = cache / ('models--' + model.replace('/', '--')) / 'refs/main'
        revision = cfg.get('model_revision') or (ref.read_text().strip() if ref.is_file() else None)
    if revision and complete_snapshot(snapshot(cache, revision, model), (pin or {}).get('file_sizes'), model):
        return dict(status='cached_complete', revision=revision, access='Local availability; online authorization not checked')
    access = ('Offline check cannot distinguish gated denial from authorized-but-not-downloaded; use --check-access'
              if model_spec(model)['gated'] else 'Public model snapshot missing/incomplete; network availability not checked')
    return dict(status='unavailable', revision=revision, access=access)


def check_access(cfg, revision=None):
    """Explicit online metadata/HEAD check, with implicit HF login; never download bytes."""
    model = cfg.get('model', MODEL)
    revision = revision or cfg.get('model_revision', 'main')
    model_spec(model)
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    except ImportError:
        return dict(status='unavailable', reason='Install Phase 8 dependencies before checking Hub access')
    try:
        auth = {} if model_spec(model)['gated'] else {'token': False}
        info = HfApi().model_info(model, revision=revision, files_metadata=True, **auth)
        resolved = info.sha
        if not re.fullmatch('[0-9a-f]{40}', resolved or ''):
            return dict(status='unavailable', reason='Hub did not return an immutable commit')
        weights = [s for s in info.siblings if s.rfilename.startswith('model') and s.rfilename.endswith('.safetensors')]
        if not weights:
            return dict(status='unavailable', reason='Expected official safetensors files unavailable')
        # Repository metadata alone may be public even when weight access is gated.
        get_hf_file_metadata(hf_hub_url(model, weights[0].rfilename, revision=resolved), **auth)
        if revision != 'main' and resolved != revision:
            return dict(status='unavailable', reason='Hub resolved revision differs from the requested pin')
        sizes = {s.rfilename: s.size for s in info.siblings if isinstance(s.size, int)}
        pin = dict(model=model, revision=resolved, tokenizer_revision=resolved, file_sizes=sizes)
        local = cache_check(cfg, pin)
        return dict(status=local['status'] if local['status'] == 'cached_complete' else 'authorized_download_required',
                    revision=resolved, pin=pin)
    except GatedRepoError:
        return dict(status='denied', reason='Gated access denied; request access and authenticate locally')
    except HfHubHTTPError as exc:
        code = getattr(getattr(exc, 'response', None), 'status_code', None)
        return dict(status='denied' if code in (401, 403) else 'unavailable', reason='Hub metadata/HEAD request failed', http_status=code)
    except Exception:
        # Never emit exception strings that may contain authenticated URLs/headers.
        return dict(status='unavailable', reason='Hub access could not be verified; no substitution or download performed')


def prompt_inputs(tokenizer, question, suffix):
    text = tokenizer.apply_chat_template([dict(role='user', content=question + suffix)], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    return text, encoded


def response_record(ids, eos_ids, tokenizer, cap):
    if not ids or len(ids) > cap:
        raise ValueError('Invalid generated length')
    eos = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    if eos is not None and eos != len(ids) - 1:
        raise ValueError('Unexpected tokens after generation terminator')
    if eos is None and len(ids) != cap:
        raise ValueError('Generation stopped without EOS or length limit')
    visible = ids if eos is None else ids[:eos]
    return dict(generated_token_ids=ids, generated_tokens=len(ids), response_tokens=len(visible),
                termination='eos' if eos is not None else 'length_limit', stop_token_id=ids[-1] if eos is not None else None,
                response=tokenizer.decode(visible, skip_special_tokens=True), raw_response=tokenizer.decode(ids, skip_special_tokens=False))


def generate_one(model, tokenizer, encoded, eos_ids, cap=256):
    """Native generate, one sequence, cached greedy decoding. CPU use is fixture-only."""
    import torch
    device = encoded['input_ids'].device
    gpu = device.type == 'cuda'
    events = []; pending = []
    def before(module, args, kwargs):
        if gpu:
            event = torch.cuda.Event(enable_timing=True); event.record(); pending.append(event)
        else:
            pending.append(None)
    def after(module, args, kwargs, output):
        start = pending.pop()
        if gpu:
            end = torch.cuda.Event(enable_timing=True); end.record(); events.append((start, end))
        else:
            events.append((None, None))
    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after, with_kwargs=True)
    try:
        if gpu:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        wall = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=cap,
                                    min_new_tokens=0, use_cache=True, eos_token_id=eos_ids,
                                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
                                    repetition_penalty=1.0, no_repeat_ngram_size=0, return_dict_in_generate=False)
        if gpu:
            end.record(); end.synchronize()
        wall_seconds = time.perf_counter() - wall
        start = encoded['input_ids'].shape[1]
        if output.shape[0] != 1 or not torch.equal(output[0, :start], encoded['input_ids'][0]):
            raise ValueError('Expected prompt-prefixed batch-one generate output')
        ids = output[0, start:].cpu().tolist()  # Never decode prompt IDs as response.
        if not events or len(events) != len(ids):
            raise ValueError('Expected one initial prefill forward plus one per subsequent predicted token')
        intervals = [a.elapsed_time(b) / 1000 for a, b in events] if gpu else []
        total = begin.elapsed_time(end) / 1000 if gpu else None
        timing = dict(gpu_prefill_decode_seconds=total, gpu_prefill_forward_seconds=intervals[0] if gpu else None,
                      gpu_decode_forward_seconds=sum(intervals[1:]) if gpu else None,
                      generation_wall_seconds=wall_seconds, tokens_per_second=len(ids) / total if total else None,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if gpu else None,
                      model_forward_passes=len(events), decode_forward_passes=len(events) - 1)
        return dict(**response_record(ids, eos_ids, tokenizer, cap), **timing)
    finally:
        pre.remove(); post.remove()


def validate_result(result, question, run_hash, require_gpu=True, model=MODEL, revision=None):
    if result.get('model', MODEL) != model or (revision is not None and result.get('model_revision') != revision):
        raise ValueError('Saved result model/revision mismatch')
    if result['question_id'] != question['question_id'] or result['stratum'] != question['stratum'] or result['run_hash'] != run_hash or result['policy'] != model_spec(model)['policy']:
        raise ValueError('Saved result identity/configuration mismatch')
    gold = extract_answer(question['gold_answer'], gold=True)[0]
    predicted, method = extract_answer(result['response'])
    if (result['gold_numeric'], result['predicted_numeric'], result['extraction_method'], result['correct'], result['extraction_failed']) != (gold, predicted, method, predicted == gold, predicted is None):
        raise ValueError('Saved evaluation does not match the unchanged extractor')
    if result['inference_seed'] != 42 or result['calculation_count'] != question['calculation_count']:
        raise ValueError('Saved seed/complexity metadata mismatch')
    ids = result['generated_token_ids']; n = result['generated_tokens']; eos = result['eos_token_ids']
    if type(n) is not int or not 1 <= n <= 256 or n != len(ids) or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('Invalid output token count')
    stop = next((i for i, t in enumerate(ids) if t in eos), None)
    if (stop is not None and stop != n - 1) or (stop is None and n != 256):
        raise ValueError('Invalid stopping semantics')
    if result['termination'] != ('eos' if stop is not None else 'length_limit') or result['response_tokens'] != n - (stop is not None):
        raise ValueError('Invalid termination metadata')
    if result['stop_token_id'] != (ids[-1] if stop is not None else None) or result['model_forward_passes'] != n or result['decode_forward_passes'] != n - 1:
        raise ValueError('Invalid forward/termination counts')
    for field in ('generation_wall_seconds', 'gpu_prefill_decode_seconds', 'gpu_prefill_forward_seconds', 'gpu_decode_forward_seconds', 'tokens_per_second', 'peak_allocated_bytes'):
        value = result[field]
        if value is None and not require_gpu and field != 'generation_wall_seconds':
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid timing/memory schema')
    if require_gpu:
        if result['gpu_prefill_decode_seconds'] <= 0 or not math.isclose(result['tokens_per_second'], n / result['gpu_prefill_decode_seconds'], rel_tol=1e-9):
            raise ValueError('Invalid throughput')
    if type(result['prompt_tokens']) is not int or result['prompt_tokens'] <= 0:
        raise ValueError('Invalid prompt length')


def read_journal(path, questions, run_hash, repair=False, require_gpu=True, model=MODEL, revision=None):
    records = {}; pending = set(); qmap = {q['question_id']: q for q in questions}
    path = Path(path)
    if not path.exists():
        return records
    with path.open('rb+' if repair else 'rb') as file:
        while True:
            offset = file.tell(); line = file.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if not repair:
                    raise ValueError('Partial journal append; resume owning runner to repair')
                file.truncate(offset); file.flush(); os.fsync(file.fileno()); break
            event = json.loads(line); qid = event['question_id']
            if revision is not None and event.get('model_revision') != revision:
                raise ValueError('Journal model revision mismatch')
            if qid not in qmap or event['run_hash'] != run_hash or event['policy'] != model_spec(model)['policy'] or event.get('model', MODEL) != model or qid in records:
                raise ValueError('Unexpected identity/configuration or duplicate journal evaluation')
            if event['event'] == 'started':
                pending.add(qid)
            elif event['event'] == 'completed' and qid in pending:
                validate_result(event['result'], qmap[qid], run_hash, require_gpu, model, revision)
                records[qid] = event['result']; pending.remove(qid)
            else:
                raise ValueError('Corrupt completion or unknown journal event')
    return records
