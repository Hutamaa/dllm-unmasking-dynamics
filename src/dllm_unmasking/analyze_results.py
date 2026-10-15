"""Rebuild tables and plots from the durable per-evaluation journal."""
from dllm_unmasking.layout import enter_repository
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

STRATEGIES = ('random', 'left_to_right', 'right_to_left', 'confidence', 'influence')


def write_csv(path, rows, fields):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def analyze(output_dir, records=None, plots=True):
    out = Path(output_dir)
    if records is None:
        # A partial trailing record is expected after interrupted jobs.
        lines = (out / 'records.jsonl').read_bytes().splitlines(keepends=True)
        records = [json.loads(line) for line in lines if line.endswith(b'\n')]
    fields = ['question_id', 'strategy', 'question', 'gold_answer', 'gold_numeric',
              'predicted_numeric', 'extraction_method', 'extraction_failed', 'correct',
              'gpu_seconds', 'wall_seconds', 'mean_token_confidence', 'position_seed',
              'forward_passes', 'influence_top_k', 'influence_approximation',
              'response', 'raw_response']
    write_csv(out / 'results.csv', records, fields)
    tmp = out / 'trajectories.tmp'
    with tmp.open('w') as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in ('question_id', 'strategy', 'position_seed',
                                                 'generated_token_ids', 'trajectory')}) + '\n')
    tmp.replace(out / 'trajectories.jsonl')
    completed = {}
    for r in records:
        completed.setdefault(r['question_id'], set()).add(r['strategy'])
    paired_ids = {i for i, s in completed.items() if s == set(STRATEGIES)}
    summary = []
    for strategy in STRATEGIES:
        rows = [r for r in records if r['strategy'] == strategy]
        paired = [r for r in rows if r['question_id'] in paired_ids]
        if not rows:
            continue
        # Legacy journals used exactly one forward per diffusion step.
        passes = [r.get('forward_passes', len(r['trajectory'])) for r in rows]
        summary.append(dict(strategy=strategy, n=len(rows), accuracy=mean(r['correct'] for r in rows),
                            extraction_failures=sum(r['extraction_failed'] for r in rows),
                            fallback_extractions=sum(r['extraction_method'] == 'last_number' for r in rows),
                            total_gpu_seconds=sum(r['gpu_seconds'] for r in rows),
                            mean_gpu_seconds=mean(r['gpu_seconds'] for r in rows),
                            mean_wall_seconds=mean(r['wall_seconds'] for r in rows),
                            total_forward_passes=sum(passes), mean_forward_passes=mean(passes),
                            influence_top_k=rows[0].get('influence_top_k', ''),
                            influence_approximation=rows[0].get('influence_approximation', ''),
                            mean_token_confidence=mean(r['mean_token_confidence'] for r in rows),
                            paired_n=len(paired), paired_accuracy=mean(r['correct'] for r in paired) if paired else ''))
    if summary:
        write_csv(out / 'summary.csv', summary, list(summary[0]))
    if plots and summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for key, filename, ylabel in [('accuracy', 'accuracy.png', 'Answer accuracy'),
                                       ('mean_gpu_seconds', 'runtime.png', 'Mean GPU generation time (s)')]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([r['strategy'] for r in summary], [r[key] for r in summary])
            ax.set_ylabel(ylabel)
            ax.set_title('Phase 1' + (' (partial run)' if any(r['n'] != len(paired_ids) for r in summary) else ''))
            if key == 'accuracy':
                ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(out / filename, dpi=160)
            plt.close(fig)
    return summary


if __name__ == '__main__':
    enter_repository()
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', nargs='?', default='outputs/phase1')
    args = parser.parse_args()
    print(json.dumps(analyze(args.output_dir), indent=2))
