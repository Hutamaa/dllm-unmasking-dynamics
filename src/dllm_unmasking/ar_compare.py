"""Read-only, model-isolated Phase 8 comparison with the original LLaDA frontier."""
import argparse
import json
from pathlib import Path
from statistics import mean

from dllm_unmasking.ar_baseline import (MODEL, model_spec, load_config, inputs, object_hash, digest, read_journal)
from dllm_unmasking.ar_analysis import baseline, summarize, FRONTIER
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.layout import enter_repository, repository_root
from dllm_unmasking.phase2_data import atomic_json


def load_run(path):
    """Accept original Llama schema via manifest identity; never repair a journal."""
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text())
    run_hash = manifest['run_hash']
    if object_hash({k:v for k,v in manifest.items() if k != 'run_hash'}) != run_hash:
        raise ValueError('Run manifest hash is inconsistent')
    cfg = manifest['config']; model_spec(cfg['model']); questions = inputs(cfg)
    if manifest['model'] != cfg['model'] or manifest['model_revision'] != cfg['model_revision'] or manifest['tokenizer_revision'] != cfg['model_revision']:
        raise ValueError('Run manifest model/config identity mismatch')
    for key, file in [('selection_sha256',cfg['selection_file']), ('diffusion_sha256',cfg['diffusion_results'])]:
        if manifest[key] != digest(file):
            raise ValueError('Run inputs differ from the original matched benchmark')
    records = read_journal(path/'journal.jsonl', questions, run_hash, repair=False, model=cfg['model'], revision=cfg['model_revision'] if cfg['model'] != MODEL else None)
    if any(r.get('model_revision',cfg['model_revision']) != cfg['model_revision'] for r in records.values()):
        raise ValueError('Records from different model revisions cannot be combined')
    # Missing model fields are accepted only for the original Llama schema.
    if cfg['model'] != MODEL and any('model_revision' not in r for r in records.values()):
        raise ValueError('Qwen records require explicit model revision')
    return cfg, questions, records, manifest


def mark_dominance(points):
    """Descriptive point-estimate dominance across explicitly different timing scopes."""
    eligible = [p for p in points if p['n'] == 60]
    for point in points:
        point['descriptive_cross_scope_dominated'] = None if point['n'] != 60 else any(
            other['mean_gpu_seconds'] <= point['mean_gpu_seconds'] and other['accuracy'] >= point['accuracy']
            and (other['mean_gpu_seconds'] < point['mean_gpu_seconds'] or other['accuracy'] > point['accuracy'])
            for other in eligible)
    return points


def plot(out, points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 6))
    original = sorted((p for p in points if p['family']=='diffusion'), key=lambda p:p['mean_gpu_seconds'])
    ax.plot([p['mean_gpu_seconds'] for p in original], [100*p['accuracy'] for p in original], '--', color='tab:blue', alpha=.6)
    for p in points:
        if p['n'] != 60:
            continue  # Different denominators cannot define the matched frontier.
        dominated = p['descriptive_cross_scope_dominated']
        ax.scatter(p['mean_gpu_seconds'],100*p['accuracy'],marker='x' if dominated else ('o' if p['family']=='diffusion' else 'D'),
                   color='tab:red' if dominated else ('tab:blue' if p['family']=='diffusion' else 'tab:green'),s=65)
        offset = (5,-16) if p['name']=='confidence P8' else (5,7)
        ax.annotate(p['name'],(p['mean_gpu_seconds'],100*p['accuracy']),xytext=offset,textcoords='offset points',fontsize=8)
    ax.set(xscale='log',xlabel='Mean recorded GPU interval / question (s; different timing scopes)',ylabel='Numerical-answer accuracy (%)',
           ylim=(0,100),title='Matched 60-question system comparison — observed point estimates')
    ax.margins(x=.3)
    ax.legend(handles=[Line2D([],[],color='tab:blue',marker='o',linestyle='--',label='Original LLaDA empirical frontier'),
                       Line2D([],[],color='tab:green',marker='D',linestyle='',label='Completed AR model run'),
                       Line2D([],[],color='tab:red',marker='x',linestyle='',label='Descriptively dominated across recorded scopes')],fontsize=8)
    n_ar=sum(p['family']=='ar' and p['n']==60 for p in points)
    partial=sum(p['family']=='ar' and p['n']!=60 for p in points)
    fig.text(.5,.035,'AR: prefill + native generation. LLaDA: full sampler. Different jobs, models and tokenizers.\n'
             f'No controlled latency/architecture claim. {n_ar} completed AR runs; {partial} partial runs omitted from plot.\n'
             'Dominance ignores sampling/timing uncertainty and is descriptive only.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,1]);fig.savefig(out/'accuracy_latency.png',dpi=160);plt.close(fig)


def compare(out, run_dirs, cfg):
    questions = inputs(cfg); groups = baseline(cfg,questions)
    out = Path(out).resolve()
    if not out.is_relative_to(repository_root()/'results'):
        raise ValueError('Combined outputs must stay in a separate ignored results/ directory')
    paths=[Path(p).resolve() for p in run_dirs]
    if len(paths)!=len(set(paths)) or any(out==p or out in p.parents or p in out.parents for p in paths):
        raise ValueError('Duplicate runs or input/output overlap')
    protected={}
    for p in paths:
        for name in ('manifest.json','journal.jsonl'):protected[str(p/name)]=digest(p/name)
    points=[];seen=set();runs=[]
    for method in FRONTIER:
        rs=list(groups[method].values())
        points.append(dict(family='diffusion',name=method,model='GSAI-ML/LLaDA-8B-Instruct',revision='852152c60bdfed4632635bfcf3e83408ad3f65bf',
                           run_hash='',n=60,correct=sum(r['correct'] for r in rs),accuracy=mean(r['correct'] for r in rs),
                           mean_gpu_seconds=mean(r['gpu_seconds'] for r in rs),scope='Historical full diffusion sampler stream interval'))
    out.mkdir(parents=True,exist_ok=True)
    for path in paths:
        run_cfg,qs,records,manifest=load_run(path)
        identity=(run_cfg['model'],run_cfg['model_revision'])
        if identity in seen:
            raise ValueError('Choose one run per model/revision; replicates are never pooled')
        seen.add(identity);alias=model_spec(run_cfg['model'])['alias']
        run_out=out/'runs'/f'{alias}-{run_cfg["model_revision"][:8]}-{manifest["run_hash"][:8]}'
        run_out.mkdir(parents=True,exist_ok=True)
        summarize(run_out,records,qs,run_cfg)
        runs.append(dict(model=identity[0],revision=identity[1],run_hash=manifest['run_hash'],n_completed=len(records),complete=len(records)==60,
                         manifest_sha256=digest(path/'manifest.json'),journal_sha256=digest(path/'journal.jsonl')))
        if not records:
            continue
        rs=list(records.values())
        points.append(dict(family='ar',name=f'{identity[0].split("/")[-1]} {identity[1][:8]}',model=identity[0],revision=identity[1],run_hash=manifest['run_hash'],
                           n=len(rs),correct=sum(r['correct'] for r in rs),accuracy=mean(r['correct'] for r in rs),
                           mean_gpu_seconds=mean(r['gpu_prefill_decode_seconds'] for r in rs),scope='AR prefill + native cached generation stream interval'))
    mark_dominance(points);write_csv(out/'accuracy_latency.csv',points);plot(out,points)
    atomic_json(out/'comparison_metadata.json',dict(runs=runs,diffusion_sha256=digest(cfg['diffusion_results']),
                selection_sha256=digest(cfg['selection_file']),
                notes=['No pooling across models/revisions/runs','Partial AR runs excluded from dominance and figure',
                       'Dominance is descriptive across unequal timing scopes, not a controlled latency result']))
    if any(digest(p)!=h for p,h in protected.items()):
        raise RuntimeError('Read-only input changed during comparison')
    return points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',action='append',help='Repeat for one run per AR model/revision; defaults to existing configured runs')
    parser.add_argument('--output-dir',default='results/phase8_comparison')
    args=parser.parse_args()
    paths=args.run_dir
    if paths is None:
        paths=[load_config(model=alias)['output_dir'] for alias in ('qwen','llama')
               if (Path(load_config(model=alias)['output_dir'])/'manifest.json').exists()]
    points=compare(args.output_dir,paths,load_config())
    print(f'Comparison saved: {sum(p["family"]=="ar" and p["n"]==60 for p in points)} completed AR overlays; partial runs are not ranked.')


if __name__=='__main__':
    enter_repository()
    main()
