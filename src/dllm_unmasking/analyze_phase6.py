"""Read-only matched P1/P4/P8 accuracy-latency analysis; writes Phase 6 outputs only."""
from dllm_unmasking.layout import source_matches,enter_repository
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import numpy as np
from experiments.phase2.run import csv_atomic
from dllm_unmasking.phase2_data import atomic_json,sha
from dllm_unmasking.analyze_phase25 import outputs,paired
from dllm_unmasking.analyze_phase15 import holm

POLICIES=('left_to_right','confidence')


def analyze(out,records,lookup,selection,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p1=sorted(records.values(),key=lambda r:(r['question_id'],r['strategy']))
    old=[r for k,r in sorted(lookup.items()) if k[1] in POLICIES and k[2] in (4,8)]
    combined=p1+old;diagnostic={(r['question_id'],r['strategy'],r['parallelism']):r for r in outputs(combined,cfg['mask_id'])}
    rows=[]
    for r in combined:
        d=diagnostic[r['question_id'],r['strategy'],r['parallelism']]
        rows.append(dict(question_id=r['question_id'],stratum=r['stratum'],policy=r['strategy'],parallelism=r['parallelism'],
            source_phase=6 if r['parallelism']==1 else 2,correct=r['correct'],extraction_failed=r['extraction_failed'],
            final_marker_present=d['final_marker_present'],parser_fallback=d['fallback'],max_word_run=d['max_consecutive_word_run'],
            repeated_trigram_fraction=d['repeated_trigram_fraction'],generated_token_repeat_fraction=d['generated_token_repeat_fraction'],
            forward_passes=r['forward_passes'],theoretical_forward_reduction_vs_p1=256/r['forward_passes'],
            observed_decode_gpu_stream_seconds=r.get('gpu_seconds'),observed_decode_wall_seconds=r.get('wall_seconds'),
            observed_evaluation_wall_seconds=r.get('evaluation_wall_seconds'),
            observed_model_forward_gpu_stream_seconds=r.get('model_forward_gpu_stream_seconds'),
            sampler_nonforward_main_thread_cpu_seconds=r.get('sampler_nonforward_main_thread_cpu_seconds'),
            model_forward_host_enqueue_wall_seconds=r.get('model_forward_host_enqueue_wall_seconds'),
            tokens_initially_committed=r.get('tokens_initially_committed',len(r['generated_token_ids'])),
            tokens_before_first_eos=r.get('tokens_before_first_eos'),tokens_final_decoded_response=r.get('tokens_final_decoded_response'),
            latency_scope='legacy-compatible decode stream/wall; instrumented forwards; full evaluation wall' if r['parallelism']==1 else 'legacy decode stream/wall only; E2E/CPU/forward-only unavailable'))
    csv_atomic(out/'accuracy_latency_table.csv',rows)
    if p1:
        fields=('question_id','stratum','strategy','parallelism','correct','gold_numeric','predicted_numeric','extraction_method','extraction_failed',
            'forward_passes','gpu_seconds','wall_seconds','evaluation_wall_seconds','model_forward_gpu_stream_seconds',
            'sampler_nonforward_main_thread_cpu_seconds','model_forward_host_enqueue_wall_seconds','tokens_initially_committed',
            'tokens_before_first_eos','tokens_final_decoded_response','response','raw_response')
        csv_atomic(out/'results.csv',[{k:r[k] for k in fields} for r in p1])
        csv_atomic(out/'response_diagnostics.csv',[diagnostic[r['question_id'],r['strategy'],1] for r in p1])
    summary=[]
    for policy in POLICIES:
        for p in (1,4,8):
            for stratum in ('all','0-2','3-4','5+'):
                a=[r for r in rows if r['policy']==policy and r['parallelism']==p and (stratum=='all' or r['stratum']==stratum)]
                n=len(a);row=dict(policy=policy,parallelism=p,stratum=stratum,n=n,expected_n=60 if stratum=='all' else 20,
                    correct=sum(r['correct'] for r in a),accuracy=mean(r['correct'] for r in a) if n else None,
                    extraction_failures=sum(r['extraction_failed'] for r in a),parser_fallbacks=sum(r['parser_fallback'] for r in a),
                    final_markers=sum(r['final_marker_present'] for r in a),forward_passes=sum(r['forward_passes'] for r in a),
                    theoretical_forward_reduction_vs_p1=p)
                for field in ('observed_decode_gpu_stream_seconds','observed_decode_wall_seconds','observed_evaluation_wall_seconds',
                              'sampler_nonforward_main_thread_cpu_seconds','max_word_run','repeated_trigram_fraction'):
                    values=[r[field] for r in a if r[field] is not None]
                    row[field+'_n']=len(values);row[field+'_mean']=mean(values) if values else None
                summary.append(row)
    csv_atomic(out/'summary.csv',summary)
    all_records={(r['question_id'],r['strategy'],r['parallelism']):r for r in combined};ids=sorted(q['question_id'] for q in selection['questions'])
    conditions=[(s,p) for s in POLICIES for p in (1,4,8)]
    # Compare P1 policies and every P1-versus-recorded-P4/P8 condition. Never infer unmatched answers.
    comparisons=[];rng=np.random.default_rng(42)
    for a,b in itertools.combinations(conditions,2):
        if a[1]!=1 and b[1]!=1:continue
        matched=[q for q in ids if (q,*a) in all_records and (q,*b) in all_records]
        if not matched:continue
        aa=[int(all_records[(q,*a)]['correct']) for q in matched];bb=[int(all_records[(q,*b)]['correct']) for q in matched]
        stats=paired(aa,bb,rng,10000,[all_records[(q,*a)]['stratum'] for q in matched])
        for old,new in (('p4_mean','accuracy_a'),('p8_mean','accuracy_b'),('delta_p8_minus_p4','accuracy_difference_b_minus_a')):stats[new]=stats.pop(old)
        a_only=sum(x and not y for x,y in zip(aa,bb));b_only=sum(y and not x for x,y in zip(aa,bb));discordant=a_only+b_only
        comparisons.append(dict(policy_a=a[0],parallelism_a=a[1],policy_b=b[0],parallelism_b=b[1],**stats,
            a_only_correct=a_only,b_only_correct=b_only,both_correct=sum(x and y for x,y in zip(aa,bb)),both_incorrect=sum(not x and not y for x,y in zip(aa,bb)),
            mcnemar_exact_p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(a_only,b_only)+1))/2**discordant) if discordant else 1.))
    if comparisons:holm(comparisons);csv_atomic(out/'paired_comparisons.csv',comparisons)
    counts={policy:sum(r['strategy']==policy for r in p1) for policy in POLICIES}
    atomic_json(out/'completion.json',dict(per_policy_completed=counts,total_completed=len(p1),expected_total=120,
        complete=all(count==60 for count in counts.values()),all_question_ids=ids))
    report=['# Phase 6 matched accuracy–latency benchmark','',f'P1 completed counts: {counts}; total {len(p1)}/120. Complete only when each policy has all 60.',
        'No P1 outcomes or latency are imputed for missing evaluations. The observed P4/P8 entries reuse Phase 2 exactly.','',
        '## Timing scope','',
        '`gpu_seconds` / observed_decode_gpu_stream_seconds is CUDA-event elapsed time for the full decode stream interval, '
        'including selection GPU work and host launch gaps. It is not a sum of kernel runtimes. Phase 6 uses the original sampler\'s event scope.',
        '`model_forward_gpu_stream_seconds` sums per-forward CUDA stream intervals. It excludes outside-forward selection operations '
        'but can include CPU launch gaps within each forward; it is not isolated hardware kernel execution time. No per-step synchronization is introduced.',
        '`wall_seconds` is the original synchronized decode wall interval, excluding tokenization/text decoding/journal writes.',
        '`evaluation_wall_seconds` spans prompt preparation through synchronized generation, trace transfer, response decoding, '
        'visible-text retokenization and numerical extraction. It excludes model loading, journal fsync, summaries and record audit.',
        '`sampler_nonforward_main_thread_cpu_seconds` is outer sampler thread CPU time minus model-call thread CPU time. '
        'It includes selection/score dispatch, allocation, instrumentation and serialization; it excludes GPU waiting and other CPU threads. '
        'It is measurable host overhead, not a precise pure-selection cost. Host enqueue wall time includes Python model calls and any blocking inside them.',
        'Phase 2 has comparable decode stream/wall scopes, but no E2E, thread-CPU or forward-only measurements. '
        'Those columns are blank, not inferred. Hardware/software/load and additional event instrumentation can affect cross-run latency. '
        'Review private environment metadata before interpreting speedups.',
        'Final decoded-response token count is actual re-encoding with the pinned tokenizer and add_special_tokens=False. '
        'It differs from generated length (256) and token-prefix length before EOS; EOS never shortens generation.',
        '', '## Statistics and interpretation','',
        'Question-matched bootstrap intervals use 10,000 resamples, seed 42, preserving observed calculation-stratum counts. '
        'Exact two-sided McNemar tests use discordant correctness; Holm adjustment covers currently observed comparison rows. '
        'Partial-run results are exploratory; complete-run family has nine comparisons. Numerical correctness is not proof of faithful mathematical reasoning.',
        'The theoretical forward reductions 1×/4×/8× count model calls only. They are not observed wall/GPU speedups. '
        'Observed latency remains in separate columns with valid denominators. All 60 questions are matched across P1/P4/P8; '
        'Phase 1 accuracy on a different 150-question subset is not used as a matched benchmark.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/phase6.yaml');parser.add_argument('--phase2-results');parser.add_argument('--run-dir',help='Copy of Phase 6 run directory; original config remains unchanged')
    args=parser.parse_args()
    import yaml
    from experiments.phase6.run import validate,read_results,config_hash
    cfg=yaml.safe_load(Path(args.config).read_text());source=Path(args.phase2_results or cfg['phase2_results']).expanduser().resolve();out=Path(args.run_dir or cfg['output_dir']).resolve()
    if not out.is_relative_to(Path('outputs/phase6').resolve()) or out==source or out in source.parents or source in out.parents:raise ValueError('Independent Phase 6 output required')
    selection,lookup,_=validate(cfg,source)
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['config_hash']!=config_hash(cfg) or manifest['config']!=cfg or manifest['selection_sha256']!=sha(cfg['selection_file']) or manifest['phase2_journal_sha256']!=sha(source/'journal.jsonl') or manifest['phase2_manifest_sha256']!=sha(source/'manifest.json'):
        raise ValueError('Analysis input/provenance mismatch')
    for name,digest in manifest['source_sha256'].items():
        if not source_matches(name,digest):raise ValueError(f'Analysis source mismatch: {name}')
    protected={str(p):sha(p) for p in (source/'journal.jsonl',source/'manifest.json',out/'journal.jsonl',out/'manifest.json')}
    records,_=read_results(out,selection,cfg,repair=False);analyze(out,records,lookup,selection,cfg)
    if any(sha(Path(p))!=digest for p,digest in protected.items()):raise RuntimeError('Original journal/manifest changed')
    print(f'Phase 6 analysis updated: {len(records)}/120 complete, original journals unchanged.')


if __name__=='__main__':main()
