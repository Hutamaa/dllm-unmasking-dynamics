"""Offline, fail-closed Phase 8 import, raw-journal verification and report generation."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import hashlib
import platform
from statistics import mean
import tempfile

import numpy as np
from dllm_unmasking.ar_baseline import digest, inputs
from dllm_unmasking.ar_compare import load_run
from dllm_unmasking.ar_analysis import baseline, summarize, FRONTIER
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.evaluation import extract_answer
from dllm_unmasking.layout import enter_repository, repository_root
from dllm_unmasking.phase2_data import DATASET_REVISION, atomic_json


def inventory(path):
    return {p.relative_to(path).as_posix(): digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def import_run(source, destination):
    """Idempotent byte-preserving local import; reject conflicting existing files."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Import source and destination overlap')
    required = ('manifest.json', 'journal.jsonl', 'environment.json', 'results.csv', 'summary.csv')
    if any(not (source / name).is_file() for name in required):
        raise ValueError('Missing required Phase 8 artifacts')
    files = {k: v for k, v in inventory(source).items() if k != '.run.lock'}
    if destination.exists() and any(k not in files or v != files[k] for k, v in inventory(destination).items()):
        raise ValueError('Existing imported artifacts differ; never overwrite')
    for name, sha in files.items():
        target = destination / name
        if target.exists() and digest(target) != sha:
            raise ValueError('Import hash conflict')
    for name in files:
        target = destination / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
    if inventory(destination) != files:
        raise ValueError('Imported bytes differ from original artifacts')
    return files


def paired(a, b, strata, name_a, name_b, seed=42, repetitions=10000):
    """B minus A; each contrast starts an independent PCG64(seed) stream."""
    ids = sorted(strata)
    if set(a) != set(ids) or set(b) != set(ids):
        raise ValueError('Paired analysis requires the exact complete matched set')
    d = np.array([int(b[q]['correct']) - int(a[q]['correct']) for q in ids])
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.zeros(repetitions)
    for stratum in ('0-2', '3-4', '5+'):
        values = d[[i for i, q in enumerate(ids) if strata[q] == stratum]]
        draws += rng.choice(values, size=(repetitions, len(values)), replace=True).sum(axis=1)
    low, high = np.quantile(draws * 100 / len(ids), [.025, .975], method='linear')
    recovered, regressed = int((d == 1).sum()), int((d == -1).sum())
    n = recovered + regressed
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(recovered, regressed) + 1)) / 2**n) if n else 1.
    return dict(a=name_a, b=name_b, n=len(ids), both_correct=sum(a[q]['correct'] and b[q]['correct'] for q in ids),
                both_incorrect=sum(not a[q]['correct'] and not b[q]['correct'] for q in ids),
                b_recovers=recovered, b_regresses=regressed, b_minus_a_pp=float(d.mean()*100),
                bootstrap_low_pp=float(low), bootstrap_high_pp=float(high), mcnemar_exact_p=p)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def verify_saved_tables(source, records, questions, cfg):
    """Check saved numerical tables against fresh derivation, never write to source."""
    with tempfile.TemporaryDirectory() as temp:
        summarize(Path(temp), records, questions, cfg)
        for name in ('results.csv', 'summary.csv', 'paired_diffusion.csv', 'latency_comparison.csv'):
            if not (source / name).exists():
                raise ValueError('Missing saved derived table: ' + name)
            old, new = read_csv(source / name), read_csv(Path(temp) / name)
            if len(old) != len(new):
                raise ValueError('Saved table row count mismatch: ' + name)
            for a, b in zip(old, new):
                for key in a.keys() & b.keys():
                    try:
                        equal = math.isclose(float(a[key]), float(b[key]), rel_tol=1e-12, abs_tol=1e-12)
                    except ValueError:
                        equal = a[key] == b[key]
                    if not equal:
                        raise ValueError('Saved table disagrees with journal: ' + name + '/' + key)


def verify_sources(manifest):
    """Locate exact recorded source bytes in the working tree or local Git history."""
    matches = {}
    root = repository_root()
    for name, expected in manifest['source_sha256'].items():
        path = root / name
        if path.is_file() and digest(path) == expected:
            matches[name] = 'current source matches recorded hash'
            continue
        history = subprocess.run(['git', 'log', '--all', '--format=%H', '--', name], cwd=root,
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        for commit in history:
            content = subprocess.run(['git', 'show', commit+':'+name], cwd=root, capture_output=True)
            if content.returncode == 0 and hashlib.sha256(content.stdout).hexdigest() == expected:
                matches[name] = 'local Git commit ' + commit
                break
        else:
            raise ValueError('Exact recorded inference source unavailable: ' + name)
    return matches


def verify_events(path):
    pending, done = set(), set()
    for line in path.open():
        event = json.loads(line); q = event['question_id']
        if event['event'] == 'started' and q not in pending and q not in done:
            pending.add(q)
        elif event['event'] == 'completed' and q in pending and q not in done:
            pending.remove(q); done.add(q)
        else:
            raise ValueError('Duplicate/inconsistent raw run event')
    if pending or len(done) != 60:
        raise ValueError('Incomplete raw run')


def verify_diffusion(paths, cfg, questions):
    groups = baseline(cfg, questions)
    specs = {'confidence P8': (2, 'confidence', 8), 'Confidence + random revision': (5, 'confidence_random_remask', 8),
             'left_to_right P4': (2, 'left_to_right', 4), 'left_to_right P1': (6, 'left_to_right', 1)}
    checks = []
    qmap = {q['question_id']: q for q in questions}
    for name, (phase, policy, parallelism) in specs.items():
        directory = paths[phase]
        manifest = json.loads((directory / 'manifest.json').read_text())
        c = manifest['config']
        if (c['model_revision'], c['dataset_revision'], c['inference_seed'], c['prompt_suffix'], c['dtype']) != (
                '852152c60bdfed4632635bfcf3e83408ad3f65bf', DATASET_REVISION, cfg['inference_seed'], cfg['prompt_suffix'], cfg['dtype']):
            raise ValueError('Diffusion protocol mismatch')
        if manifest['selection_sha256'] != digest(cfg['selection_file']):
            raise ValueError('Diffusion selection mismatch')
        found = {}
        for line in (directory / 'journal.jsonl').open():
            event = json.loads(line)
            if event['event'] != 'completed':
                continue
            r = event['result']
            if r['strategy'] != policy or r['parallelism'] != parallelism:
                continue
            qid = r['question_id']
            if qid in found or qid not in qmap:
                raise ValueError('Duplicate/unknown diffusion evaluation')
            q = qmap[qid]; predicted, method = extract_answer(r['response']); gold = extract_answer(q['gold_answer'], gold=True)[0]
            if (r['question'], r['gold_answer'], r['stratum'], r['calculation_count']) != (q['question'], q['gold_answer'], q['stratum'], q['calculation_count']):
                raise ValueError('Diffusion question content mismatch')
            if (r['gold_numeric'], r['predicted_numeric'], r['correct'], r['extraction_failed'], r['extraction_method']) != (gold, predicted, predicted == gold, predicted is None, method):
                raise ValueError('Diffusion saved extraction mismatch')
            saved = groups[name][qid]
            if r['correct'] != saved['correct'] or any(not math.isclose(r[field], saved[field], rel_tol=1e-12) for field in ('gpu_seconds', 'wall_seconds')):
                raise ValueError('Public Phase 7 reference differs from raw journal')
            if len(r['generated_token_ids']) != 256 or r['forward_passes'] != 256 // parallelism:
                raise ValueError('Diffusion forward/length mismatch')
            found[qid] = r
        if set(found) != set(qmap):
            raise ValueError('Missing matched diffusion results')
        checks.append(dict(check=name, n=60, status='PASS', journal_sha256=digest(directory / 'journal.jsonl'), manifest_sha256=digest(directory / 'manifest.json')))
    return groups, checks


def table(rows, columns):
    return '\n'.join(['| ' + ' | '.join(label for key, label in columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(str(r[key]) for key, label in columns) + ' |' for r in rows])


def figures(out, summary, strata_rows, diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    diffusion = sorted([r for r in summary if r['family'] == 'LLaDA'], key=lambda r:r['mean_gpu_seconds'])
    ax.plot([r['mean_gpu_seconds'] for r in diffusion], [r['accuracy_pct'] for r in diffusion], '--o', label='Original LLaDA empirical frontier (256 positions)')
    for r in summary:
        if r['family'] == 'Qwen':
            ax.scatter(r['mean_gpu_seconds'], r['accuracy_pct'], marker='D' if r['budget'] == 256 else '^', s=80,
                       color='tab:green' if r['budget']==256 else 'tab:orange',
                       label=r['method'] + (' nominal matched budget' if r['budget'] == 256 else ' extended budget'))
        offset = (5,-15) if r['method'] == 'confidence P8' else (5,7)
        ax.annotate(r['method'], (r['mean_gpu_seconds'], r['accuracy_pct']), xytext=offset, textcoords='offset points', fontsize=8)
    ax.margins(x=.15)
    ax.set(xscale='log', xlabel='Mean recorded GPU interval (seconds/question; unequal scopes)', ylabel='Numerical-answer accuracy (%)', ylim=(30,100))
    ax.legend(fontsize=8, loc='upper left')
    fig.text(.5,.02,'Qwen: prefill + cached generation. LLaDA: decode-only full sampler.\nCross-model dominance is not validated; jobs, tokenizers and timing boundaries differ.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'accuracy_latency.png',dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    for r in summary:
        vals=[x for x in strata_rows if x['method']==r['method']]
        ax.plot([x['stratum'] for x in vals], [x['accuracy_pct'] for x in vals], marker='o',label=r['method'])
    ax.set(xlabel='Gold calculation-count proxy (not validated reasoning depth)', ylabel='Numerical-answer accuracy (%)',ylim=(0,105))
    ax.legend(fontsize=8,ncol=2);fig.text(.5,.01,'20 matched questions per stratum; exploratory subgroup point estimates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'complexity.png',dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for cap in (256,512):
        values=sorted(r['generated_tokens'] for r in diagnostics if r['budget']==cap)
        ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'Qwen-{cap} (n={len(values)})')
        ax.axvline(cap,linestyle=':',alpha=.4)
    ax.set(xlabel='Actual generated token count (includes EOS)',ylabel='Empirical cumulative fraction',ylim=(0,1.03))
    ax.legend();fig.tight_layout();fig.savefig(out/'output_lengths.png',dpi=160);plt.close(fig)


def finalize(source256, source512, diffusion_paths, out):
    root = repository_root(); out = Path(out).resolve()
    sources = {256:Path(source256).resolve(),512:Path(source512).resolve()}
    # Safeguard earlier reports, raw runs and arbitrary external output paths.
    if out != root/'reports/phase8':
        if not out.is_relative_to(root/'results'):
            raise ValueError('Use reports/phase8 or a separate results/ directory')
    protected={str(p):inventory(p) for p in sources.values()}
    protected.update({str(p):inventory(p) for p in diffusion_paths.values()})
    protected[str(root/'reports/phase7')] = inventory(root/'reports/phase7')
    imported={}; runs={}; manifests={}; configs={}; import_hashes={}; source_checks={}
    for cap, source in sources.items():
        target=root/f'outputs/phase8/imported/qwen_{cap}'
        import_hashes[str(cap)]=import_run(source,target);imported[cap]=target
        cfg, qs, records, manifest=load_run(target)
        if cfg['max_new_tokens'] != cap or len(records)!=60:
            raise ValueError('Unexpected budget or incomplete completed run')
        verify_events(target/'journal.jsonl')
        source_checks[str(cap)]=verify_sources(manifest)
        verify_saved_tables(target,records,qs,cfg)
        if digest(root/cfg['model_pin']) != manifest['model_pin_sha256']:
            raise ValueError('Model pin file differs from run provenance')
        configs[cap]=cfg;runs[cap]=records;manifests[cap]=manifest
    a,b=configs[256],configs[512]
    if {k for k in a if a[k]!=b[k]} != {'max_new_tokens','output_dir'}:
        raise ValueError('Budgets differ in more than output path and generation cap')
    for key in ('model_revision','tokenizer_revision','selection_sha256','model_pin_sha256','model_config','native_generation_config','tokenizer_config_sha256','tokenizer_sha256','chat_template'):
        if manifests[256][key] != manifests[512][key]:
            raise ValueError('Run model/tokenizer/protocol mismatch: '+key)
    for qid in runs[256]:
        if runs[256][qid]['rendered_prompt'] != runs[512][qid]['rendered_prompt'] or runs[256][qid]['prompt_tokens'] != runs[512][qid]['prompt_tokens']:
            raise ValueError('Budget runs have different tokenized prompt provenance')
    questions=inputs(a);strata={q['question_id']:q['stratum'] for q in questions}
    for q in questions:
        for cap in (256,512):
            r=runs[cap][q['question_id']]
            if q['question']+a['prompt_suffix'] not in r['rendered_prompt']:
                raise ValueError('Recorded prompt does not contain the exact benchmark question and suffix')
            eos=manifests[cap]['native_generation_config']['eos_token_id']
            if r['eos_token_ids'] != (eos if isinstance(eos,list) else [eos]):
                raise ValueError('Result EOS IDs differ from native generation configuration')
    groups, checks=verify_diffusion(diffusion_paths,a,questions)
    out.mkdir(parents=True,exist_ok=True)
    for cap in (256,512):
        checks.append(dict(check=f'Qwen-{cap}',n=60,status='PASS',journal_sha256=digest(imported[cap]/'journal.jsonl'),manifest_sha256=digest(imported[cap]/'manifest.json')))
    write_csv(out/'integrity_checks.csv',checks)
    all_groups=dict(groups,**{f'Qwen-{cap}':runs[cap] for cap in (256,512)})
    summary=[];strata_rows=[];per_question=[];diagnostics=[]
    for name,rs in all_groups.items():
        cap=512 if name=='Qwen-512' else 256;ar=name.startswith('Qwen')
        gpu='gpu_prefill_decode_seconds' if ar else 'gpu_seconds'
        scope='Qwen prefill + native cached decode' if ar else 'LLaDA decode-only full sampler'
        summary.append(dict(method=name,family='Qwen' if ar else 'LLaDA',budget=cap,n=60,correct=sum(r['correct'] for r in rs.values()),
                            accuracy_pct=100*mean(r['correct'] for r in rs.values()),mean_gpu_seconds=mean(r[gpu] for r in rs.values()),timing_scope=scope))
        for stratum in ('0-2','3-4','5+'):
            selected=[r for q,r in rs.items() if strata[q]==stratum]
            strata_rows.append(dict(method=name,stratum=stratum,n=len(selected),correct=sum(r['correct'] for r in selected),accuracy_pct=100*mean(r['correct'] for r in selected)))
        for qid,r in sorted(rs.items()):
            per_question.append(dict(method=name,budget=cap,question_id=qid,stratum=strata[qid],correct=r['correct'],gpu_seconds=r[gpu],timing_scope=scope))
    for cap,rs in runs.items():
        for qid,r in sorted(rs.items()):
            diagnostics.append(dict(budget=cap,question_id=qid,stratum=strata[qid],correct=r['correct'],generated_tokens=r['generated_tokens'],response_tokens=r['response_tokens'],
                eos=r['termination']=='eos',length_limited=r['termination']=='length_limit',final_marker='####' in r['response'],parser_fallback=r['extraction_method']=='last_number',
                extraction_failed=r['extraction_failed'],model_forward_passes=r['model_forward_passes'],gpu_prefill_decode_seconds=r['gpu_prefill_decode_seconds'],
                gpu_prefill_forward_seconds=r['gpu_prefill_forward_seconds'],gpu_decode_forward_seconds=r['gpu_decode_forward_seconds'],generation_wall_seconds=r['generation_wall_seconds']))
    pairs=[]
    for cap in (256,512):
        for name in FRONTIER:pairs.append(paired(groups[name],runs[cap],strata,name,f'Qwen-{cap}'))
    pairs.append(paired(runs[256],runs[512],strata,'Qwen-256','Qwen-512'))
    running=0.
    for rank,r in enumerate(sorted(pairs,key=lambda r:r['mcnemar_exact_p'])):
        running=max(running,min(1.,(len(pairs)-rank)*r['mcnemar_exact_p']));r['holm_nine_tests_p']=running
    transitions=[]
    for qid in sorted(strata):
        x,y=runs[256][qid],runs[512][qid]
        transitions.append(dict(question_id=qid,stratum=strata[qid],correct_256=x['correct'],correct_512=y['correct'],length_limited_256=x['termination']=='length_limit',
            prefix_identical=y['generated_token_ids'][:len(x['generated_token_ids'])]==x['generated_token_ids'],generated_tokens_256=x['generated_tokens'],generated_tokens_512=y['generated_tokens'],
            final_marker_256='####' in x['response'],final_marker_512='####' in y['response'],predicted_numeric_256=x['predicted_numeric'],predicted_numeric_512=y['predicted_numeric']))
    for filename,rows in [('summary.csv',summary),('complexity.csv',strata_rows),('per_question.csv',per_question),('response_diagnostics.csv',diagnostics),('paired_tests.csv',pairs),('budget_transitions.csv',transitions)]:write_csv(out/filename,rows)
    figures(out,summary,strata_rows,diagnostics)
    ds=[]
    for cap in (256,512):
        rs=[r for r in diagnostics if r['budget']==cap]
        for subset in ('all','length_limited','eos'):
            subset_rows=[r for r in rs if subset=='all' or r[subset]]
            ds.append(dict(budget=cap,subset=subset,n=len(subset_rows),correct=sum(r['correct'] for r in subset_rows),
                           mean_generated_tokens=mean(r['generated_tokens'] for r in subset_rows) if subset_rows else None,
                           eos=sum(r['eos'] for r in subset_rows),length_limited=sum(r['length_limited'] for r in subset_rows),
                           final_markers=sum(r['final_marker'] for r in subset_rows),parser_fallbacks=sum(r['parser_fallback'] for r in subset_rows),extraction_failures=sum(r['extraction_failed'] for r in subset_rows)))
    write_csv(out/'termination_summary.csv',ds)
    envs={str(cap):json.loads((imported[cap]/'environment.json').read_text()) for cap in (256,512)}
    environment_fields=('python','torch','cuda','transformers','huggingface_hub','numpy','accelerate','tokenizers','gpu','attention_implementation','dtype')
    provenance=dict(analysis_version=1,model=a['model'],model_revision=a['model_revision'],dataset_revision=DATASET_REVISION,
        selection_sha256=digest(a['selection_file']),diffusion_sha256=digest(a['diffusion_results']),imports=import_hashes,
        diffusion_integrity=checks[:4],seed=42,bootstrap_repetitions=10000,bootstrap_generator='NumPy PCG64; reset seed independently per contrast',
        bootstrap_method='Paired question differences; independently resample 20 questions within each of three strata; linear percentile 2.5/97.5%; pointwise',
        mcnemar='Exact two-sided binomial on discordances; Holm family of nine exploratory report contrasts',
        historical_intervals='Imported paired_diffusion.csv preserved; canonical intervals are a new versioned analysis with independent per-contrast streams',
        environments={cap:{k:e[k] for k in environment_fields} for cap,e in envs.items()},
        analysis_environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=__import__('matplotlib').__version__),
        inference_source_verification=source_checks,source_sha256={p:digest(root/p) for p in ('src/dllm_unmasking/finalize_phase8.py','src/dllm_unmasking/ar_baseline.py','src/dllm_unmasking/ar_compare.py','src/dllm_unmasking/ar_analysis.py','src/dllm_unmasking/evaluation.py')},
        original_inference_source_sha256={str(cap):manifests[cap]['source_sha256'] for cap in (256,512)})
    atomic_json(out/'provenance.json',provenance)
    report(out,summary,strata_rows,pairs,ds,transitions,envs)
    for path,hashes in protected.items():
        if inventory(Path(path))!=hashes:raise RuntimeError('Original artifacts changed')
    for cap,path in imported.items():
        if inventory(path)!=import_hashes[str(cap)]:raise RuntimeError('Imported raw artifacts changed')
    return summary


def report(out, summary, strata, pairs, diagnostics, transitions, environments):
    display=[dict(method=r['method'],budget=r['budget'],correct=f"{r['correct']}/{r['n']}",accuracy=f"{r['accuracy_pct']:.1f}%",gpu=f"{r['mean_gpu_seconds']:.6f}",scope=r['timing_scope']) for r in summary]
    complexity=[]
    for r in summary:
        row=dict(method=r['method'])
        for s in ('0-2','3-4','5+'):
            x=next(x for x in strata if x['method']==r['method'] and x['stratum']==s);row[s]=f"{x['correct']}/{x['n']} ({x['accuracy_pct']:.1f}%)"
        complexity.append(row)
    paired_display=[dict(a=r['a'],b=r['b'],counts=f"{r['b_recovers']}/{r['b_regresses']}",delta=f"{r['b_minus_a_pp']:+.2f}",ci=f"[{r['bootstrap_low_pp']:.2f}, {r['bootstrap_high_pp']:.2f}]",p=f"{r['mcnemar_exact_p']:.6g}",holm=f"{r['holm_nine_tests_p']:.6g}") for r in pairs]
    totals=[r for r in diagnostics if r['subset']=='all']
    diag_display=[dict(budget=r['budget'],n=r['n'],tokens=f"{r['mean_generated_tokens']:.2f}",eos=r['eos'],limited=r['length_limited'],markers=r['final_markers'],fallback=r['parser_fallbacks'],fail=r['extraction_failures']) for r in totals]
    truncated=[r for r in transitions if r['length_limited_256']]
    recoveries=sum(not r['correct_256'] and r['correct_512'] for r in transitions);regressions=sum(r['correct_256'] and not r['correct_512'] for r in transitions)
    trunc_recoveries=sum(not r['correct_256'] and r['correct_512'] for r in truncated)
    trunc_regressions=sum(r['correct_256'] and not r['correct_512'] for r in truncated)
    prefix=sum(r['prefix_identical'] for r in transitions)
    budget_pair=next(r for r in pairs if r['a']=='Qwen-256' and r['b']=='Qwen-512')
    p1_pair=next(r for r in pairs if r['a']=='left_to_right P1' and r['b']=='Qwen-512')
    text=f'''# Phase 8: matched autoregressive baseline and generation-budget sensitivity

## Scope and experimental methodology

This is an exploratory matched benchmark of numerical-answer accuracy and recorded
latency, not a controlled architecture comparison. Two completed Qwen2.5-7B-Instruct
runs use exactly the original 60 Phase 2 GSM8K test question IDs, wording, strata and
instruction suffix. There are 20 questions each in gold calculation-count strata
0–2, 3–4 and 5+; these are proxies, not validated reasoning depth. The subset excludes
all 150 Phase 1 IDs and is deliberately balanced, not a random estimate of the full
test-set distribution. No prompt tuning or baseline reruns were performed.

Model/tokenizer: `Qwen/Qwen2.5-7B-Instruct` at
`a09a35458c702b33eeacc393d103063234e8bc28`.
LLaDA: `GSAI-ML/LLaDA-8B-Instruct` at
`852152c60bdfed4632635bfcf3e83408ad3f65bf`.
Dataset: `openai/gsm8k`, main/test, revision
`740312add88f781978c0658806c59bc2815b9866`.
Selection seed is 20261014; inference seed is 42. Qwen uses its native chat template,
greedy `do_sample=False`, KV cache, batch size one, BF16, SDPA, and EOS-aware stopping.
Its only changed experimental setting is the maximum new-token budget, 256 versus
512; output paths differ for isolation. The exact suffix is:

> Solve this problem step by step. End with '#### <final numerical answer>'.

The 256 comparison is **matched in nominal generated positions**, not necessarily
in compute, tokenization, actual output length, prompt token counts, training,
parameter count, or architecture. LLaDA always fills 256 positions; Qwen may stop
at EOS. Each model uses its own tokenizer/chat template. Numerical extraction and
exact rational comparison use the unchanged shared evaluator: a last `####` marker
takes priority; malformed markers are not rescued by an earlier number, and other
fallbacks are retained. Numerical correctness does not establish faithful reasoning.

## Original LLaDA frontier and Qwen results

The four LLaDA configurations below are the original empirical Phase 7 frontier;
its data, frontier membership and report are unchanged. Qwen-256 is the nominal
matched-budget comparison; Qwen-512 is a separate **extended-budget sensitivity**.

{table(display,[('method','Configuration'),('budget','Token budget / positions'),('correct','Correct / n'),('accuracy','Accuracy'),('gpu','Mean GPU seconds'),('scope','Timing scope')])}

Qwen timings cover synchronized **prefill + native cached decode**, including
stream gaps from native generation scheduling; they exclude loading, tokenization,
and final CPU text decoding. LLaDA timings are **decode-only full sampler intervals**,
including the sampler's scheduling gaps and repeated prompt conditioning, without
an equivalent separate AR prefill. Neither interval is pure summed kernel time.
Per-forward AR prefill/decode intervals and generation wall time are retained in
[response_diagnostics.csv](response_diagnostics.csv). Jobs are separate even though
the Qwen environments report the same GPU/software. No timing repetitions or latency
confidence intervals were collected. Cross-model dominance is **not fully validated**
until timing boundaries and measurement conditions are harmonized; plotted positions
are descriptive observations, not architecture speedup estimates.

![Observed accuracy and unequal-scope latency](accuracy_latency.png)

## Complexity-stratum comparisons

Each cell reports correct/20 and accuracy. Exploratory subgroup differences are
not evidence that complexity causes accuracy changes; question features and
selection effects are uncontrolled. The full-sample paired intervals below do not
supply simultaneous subgroup uncertainty.

{table(complexity,[('method','Configuration'),('0-2','0–2 calculations'),('3-4','3–4 calculations'),('5+','5+ calculations')])}

![Exploratory complexity-stratum accuracies](complexity.png)

## Paired correctness and uncertainty

All contrasts match the same 60 question IDs. Differences are **B minus A** in
percentage points. Recoveries/regressions count A-wrong/B-correct and
A-correct/B-wrong. The canonical Phase 8 analysis uses 10,000 paired bootstrap
replicates, drawing 20 questions with replacement independently within each of the
three strata. Each contrast resets NumPy PCG64 to seed 42; intervals are linear
2.5/97.5 percentile quantiles, pointwise and conditional on this balanced subset.
Exact two-sided McNemar uses a binomial probability of 0.5 on discordant pairs;
Holm adjusts the family of nine displayed contrasts. This exploratory analysis
was specified after observing results and is not a preregistered confirmatory test.

{table(paired_display,[('a','A'),('b','B'),('counts','Recoveries / regressions'),('delta','Δ pp'),('ci','Bootstrap 95% CI pp'),('p','Exact p'),('holm','Holm p')])}

Canonical intervals are a **new versioned offline analysis**: imported per-run
`paired_diffusion.csv` files used a shared RNG stream across four contrasts;
this report resets the RNG per contrast. Those historical files remain byte-for-byte
preserved and were checked by their original regeneration method. Any changed
interval endpoint is an analysis-stream difference, not changed observations.
[paired_tests.csv](paired_tests.csv) contains full precision, and
[provenance.json](provenance.json) records inputs and the exact method.

## Output length, EOS, markers and truncation

Counts below have denominator 60 for each budget. Generated token counts include
the terminating token; visible response lengths exclude it. Parser fallback here
means the evaluator's `last_number` branch, not all alternatives to `####`.

{table(diag_display,[('budget','Qwen cap'),('n','n'),('tokens','Mean generated tokens'),('eos','EOS'),('limited','Length-limited'),('markers','#### present'),('fallback','Last-number fallback'),('fail','Extraction failures')])}

![Actual Qwen output lengths](output_lengths.png)

All {prefix}/60 Qwen-256 token sequences are exact prefixes of their matched
Qwen-512 sequences. There are {recoveries}/60 recoveries and {regressions}/60
regressions at the larger cap. Among the {len(truncated)}/60 length-limited
256-token outputs there are {trunc_recoveries}/{len(truncated)} recoveries and
{trunc_regressions}/{len(truncated)} regressions. This supports an observable
budget/truncation association within these deterministic runs. It does not prove
that extra tokens were faithful reasoning or that every recovered answer needed
all additional text. A wrong fallback answer, malformed marker, or later numeric
revision can change evaluation without isolating arithmetic reasoning.
[budget_transitions.csv](budget_transitions.csv) permits matched inspection;
[termination_summary.csv](termination_summary.csv) separates EOS and cap-limited cases.
All summary counts and saved per-run numerical tables were rechecked against the
raw journals, using the unchanged evaluator. No responses were regraded by a model.

## Confirmed findings versus mechanistic hypotheses

**Confirmed in this subset:** the two reported accuracies and recorded mean
latencies; exact question/model/template matching; the disappearance of cap-limited
outputs at 512; observed correctness transitions; and prefix consistency. Qwen-512
is more accurate here than each original LLaDA frontier point, while using a larger
nominal budget. Qwen-256 is less accurate than LTR P4/P1 on this sample. Sampling
uncertainty and paired significance are shown above rather than inferred from ranks.

**Mechanistic hypotheses, not established conclusions:** additional budget may
allow completing an answer format, extending useful computation, or revising an
earlier candidate. The records do not distinguish these mechanisms automatically.
Different training, model size, tokenizer, chat template, stopping rules and runtime
can contribute to cross-model differences. The diffusion architecture itself is
not shown to cause the observed accuracy or latency differences.

## Statistical interpretation

The larger Qwen budget improves accuracy by {budget_pair['b_minus_a_pp']:.2f} percentage points over Qwen-256
({budget_pair['b_recovers']} recoveries and {budget_pair['b_regresses']} regression). In contrast, Qwen-512 versus LTR P1 has an
observed {p1_pair['b_minus_a_pp']:.2f}-point advantage with {p1_pair['b_recovers']} recoveries and {p1_pair['b_regresses']} regressions, but its
exact McNemar p-value is {p1_pair['mcnemar_exact_p']:.6f} and its pointwise bootstrap interval
is [{p1_pair['bootstrap_low_pp']:.2f}, {p1_pair['bootstrap_high_pp']:.2f}] percentage points. This is not a confirmed population-level advantage over LTR P1. Qwen-256
versus LTR P4 likewise remains uncertain under the exact test. A boundary bootstrap
interval and an exact discordance test can disagree; neither should be selected
post hoc to imply significance.

## Future research directions

1. Analyze reasoning-budget utilization offline: first numerical candidate, first
   valid final marker, answer stability across prefixes, tokens after an answer,
   and human-audited useful computation versus repetition/format completion.
   Report prefix-extraction sensitivity separately from reasoning quality.
2. If later authorized, harmonize prefill/decode and end-to-end timing scopes,
   warmup/repetition, stopping conventions and hardware conditions before drawing
   cross-model latency or dominance conclusions.
3. Evaluate a prespecified broader question sample and budget sweep, with manual
   annotations to distinguish answer-format completion from verified arithmetic
   corrections. This report launches no new experiment or Phase 9.

## Reproducibility, integrity and publication limits

The completed HPC artifacts were already downloaded into separate local Qwen
output directories. This analysis imports hash-verified copies into
`outputs/phase8/imported/qwen_256/` and `outputs/phase8/imported/qwen_512/`,
without replacing original files. Imports reject differing existing bytes;
analysis never resumes inference, repairs journals, or writes to raw inputs.
Every run has 60 distinct completed evaluations with matching starts, no duplicates
or pending evaluations, valid manifest hashes, exact selected IDs, and consistent
saved extraction, stopping, forward counts and timing metadata. Original LLaDA
frontier rows were independently matched to completed Phase 2, 5 and 6 journals.
Input hashes, import receipts, inference-source hashes and analysis-source hashes
are in [provenance.json](provenance.json); checks are in
[integrity_checks.csv](integrity_checks.csv). Saved numerical tables were regenerated
in a temporary directory and compared without overwriting historical tables.

From a clone with analysis dependencies installed and the excluded original
selection, model pin and raw artifacts restored to their documented locations:

```bash
python -m dllm_unmasking.finalize_phase8 --diffusion-root "$DIFFUSION_RESULTS_ROOT"
```

`DIFFUSION_RESULTS_ROOT` must contain downloaded `phase2/`, `phase5/`, `phase6/`
directories. Optional `--qwen256` and `--qwen512` specify already-downloaded local
source directories. Defaults are `outputs/phase8/qwen/run` and
`outputs/phase8/qwen_512/run`. `--output-dir results/phase8_recheck` regenerates a
separate verification copy. No network, model weights, Transformers runtime or GPU
is needed. Public-only verification is:

```bash
python -m dllm_unmasking.finalize_phase8 --verify-public
```

Public per-question tables allow independent recomputation of the
statistics without full responses; raw verification additionally requires excluded
journals and the exact Phase 2 selection. [The run guide](../../experiments/phase8/README.md)
and [512 guide](../../experiments/phase8/QWEN_512.md) document original generation.

Qwen runtime: Python {environments['256']['python']}, Torch {environments['256']['torch']},
Transformers {environments['256']['transformers']}, CUDA {environments['256']['cuda']};
GPU {environments['256']['gpu']}. Offline analysis versions and hashes are recorded
separately in provenance. Historical sources differ from current analysis wrappers;
their exact recorded bytes were located in the current checkout or local Git
history and verified by SHA256. In particular, native generation defaults for
sampling and repetition penalties are explicitly overridden by the audited
runner: greedy decoding, repetition penalty 1.0, and no n-gram repetition ban. Raw job stdout and
scheduler accounting are not part of these result directories, so independent
scheduler-level timing validation is unavailable. Native tokenizer bytes and
weights are intentionally not redistributed. No missing required result artifact
or observed aggregate discrepancy was found. Raw journals, environments and full
responses remain ignored private artifacts. Public outputs contain only compact
metrics, question IDs, sanitized reproducibility metadata and figures.
'''
    (out/'PHASE8_REPORT.md').write_text(text)


def verify_public(source):
    """Recompute released aggregates and paired statistics without private artifacts."""
    source=Path(source)
    rows=read_csv(source/'per_question.csv');groups={};strata={}
    for r in rows:
        q=int(r['question_id']);name=r['method']
        if r['correct'] not in ('True','False') or q in groups.setdefault(name,{}):
            raise ValueError('Duplicate or invalid public evaluation')
        if q in strata and strata[q]!=r['stratum']:raise ValueError('Public stratum mismatch')
        strata[q]=r['stratum'];groups[name][q]=dict(correct=r['correct']=='True',gpu_seconds=float(r['gpu_seconds']))
    public=json.loads((repository_root()/'research/benchmark_selection.json').read_text())
    if strata!={q['question_id']:q['stratum'] for q in public['questions']} or len(groups)!=6:
        raise ValueError('Public benchmark set differs')
    for r in read_csv(source/'summary.csv'):
        rs=groups[r['method']]
        if set(rs)!=set(strata) or sum(x['correct'] for x in rs.values())!=int(r['correct']) or int(r['n'])!=60:
            raise ValueError('Public aggregate mismatch')
        if not math.isclose(mean(x['gpu_seconds'] for x in rs.values()),float(r['mean_gpu_seconds']),rel_tol=1e-12):
            raise ValueError('Public timing mismatch')
        if not math.isclose(100*mean(x['correct'] for x in rs.values()),float(r['accuracy_pct']),abs_tol=1e-12):
            raise ValueError('Public accuracy mismatch')
    for row in read_csv(source/'complexity.csv'):
        values=[r['correct'] for q,r in groups[row['method']].items() if strata[q]==row['stratum']]
        if len(values)!=int(row['n']) or sum(values)!=int(row['correct']):raise ValueError('Public subgroup mismatch')
    recomputed=[]
    saved=read_csv(source/'paired_tests.csv')
    for r in saved:recomputed.append(paired(groups[r['a']],groups[r['b']],strata,r['a'],r['b']))
    running=0.
    for rank,r in enumerate(sorted(recomputed,key=lambda r:r['mcnemar_exact_p'])):
        running=max(running,min(1.,(len(recomputed)-rank)*r['mcnemar_exact_p']));r['holm_nine_tests_p']=running
    for a,b in zip(saved,recomputed):
        for key,value in b.items():
            if isinstance(value,(int,float)):
                if not math.isclose(float(a[key]),value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Public paired mismatch: '+key)
            elif a[key]!=value:raise ValueError('Public paired identity mismatch')
    return len(rows),len(recomputed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen256',default='outputs/phase8/qwen/run')
    p.add_argument('--qwen512',default='outputs/phase8/qwen_512/run')
    p.add_argument('--diffusion-root',type=Path)
    p.add_argument('--verify-public',action='store_true',help='Verify released statistics without private raw artifacts')
    p.add_argument('--output-dir',default='reports/phase8')
    args=p.parse_args()
    if args.verify_public:
        print('Public rows and paired contrasts verified:',verify_public(args.output_dir));return
    if args.diffusion_root is None:p.error('--diffusion-root is required for raw-journal verification')
    rows=finalize(args.qwen256,args.qwen512,{n:args.diffusion_root/f'phase{n}' for n in (2,5,6)},args.output_dir)
    for row in rows:print(f"{row['method']}: {row['correct']}/{row['n']}, {row['mean_gpu_seconds']:.6f} s ({row['timing_scope']})")


if __name__=='__main__':
    enter_repository();main()
