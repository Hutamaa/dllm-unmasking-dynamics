"""Finalize observed Phase 9 results from immutable local journals; no inference."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
from statistics import mean,median
import tempfile

import numpy as np
from dllm_unmasking.analyze_phase9 import analyze,load_completed
from dllm_unmasking.analyze_phase8_5 import load_inputs
from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.ar_baseline import digest
from dllm_unmasking.finalize_phase8 import inventory,paired,table,verify_events,verify_sources,verify_saved_tables
from dllm_unmasking.prepare_ltr_audit import candidates
from dllm_unmasking import phase9_protocol as protocol
from dllm_unmasking.phase2_data import atomic_json
from dllm_unmasking.layout import enter_repository,repository_root


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def compare_tables(a,b):
    x,y=read_csv(a),read_csv(b)
    if len(x)!=len(y):raise ValueError('Saved table row count differs')
    for u,v in zip(x,y):
        if set(u)!=set(v):raise ValueError('Saved table fields differ')
        for key in u:
            try:ok=math.isclose(float(u[key]),float(v[key]),abs_tol=1e-12,rel_tol=1e-12)
            except ValueError:ok=u[key]==v[key]
            if not ok:raise ValueError('Saved table disagrees with raw journal: '+key)


def audit_errors(records,annotation_path):
    spec=json.loads(Path(annotation_path).read_text());rows=[]
    expected={q for q,r in records.items() if not r['correct']}
    if len(spec['entries'])!=len(expected) or {e['question_id'] for e in spec['entries']}!=expected:
        raise ValueError('Text audit must cover every incorrect case exactly once')
    for e in spec['entries']:
        r=records[e['question_id']]
        for quote in e['evidence_quotes']:
            if not quote or quote not in r['response']:raise ValueError('Audit evidence not present verbatim in response')
        mentions=candidates(r['response']);last=mentions[-1]['normalized'] if mentions else None
        if 'empty_final_marker' in e['tags'] and (not r['extraction_failed'] or not r['response'].rstrip().endswith('#### <final numerical answer>')):
            raise ValueError('Empty-marker evidence mismatch')
        if 'gold_valued_result_before_empty_marker' in e['tags'] and last!=r['gold_numeric']:raise ValueError('Pre-marker result mismatch')
        rows.append(dict(question_id=e['question_id'],stratum=r['stratum'],official_extracted=r['predicted_numeric'],gold_numeric=r['gold_numeric'],
            tags=';'.join(e['tags']),evidence_json=json.dumps(e['evidence_quotes'],ensure_ascii=False),assessment=e['assessment'],annotation_type=spec['annotation_type']))
    counts=Counter(tag for e in spec['entries'] for tag in e['tags'])
    summary=[dict(category=k,n=v,incorrect_denominator=len(expected),all_denominator=60,overlapping=True,annotation_type=spec['annotation_type']) for k,v in sorted(counts.items())]
    return rows,summary


def last_mention_diagnostic(records):
    result={}
    for q,r in records.items():
        cs=candidates(r['response']);value=cs[-1]['normalized'] if cs else None
        result[q]=dict(correct=value==r['gold_numeric'],value=value)
    return result


def finalize(run_dir,phase2,out,private,tokenizer,llada_tokenizer):
    root=repository_root();out=Path(out).resolve();private=Path(private).resolve();run_dir=Path(run_dir).resolve()
    if out!=root/'reports/phase9' and not out.is_relative_to(root/'results/phase9'):raise ValueError('Use reports/phase9 or separate results/phase9 regeneration directory')
    if not private.is_relative_to(root/'results/phase9'):raise ValueError('Review materials must remain private')
    if private==out or private in out.parents or out in private.parents:raise ValueError('Public/private outputs overlap')
    cfg,questions,concise,refs,manifest=load_completed(run_dir)
    for path in (run_dir,Path(phase2).resolve(),Path(cfg['baseline256_dir']).resolve(),Path(cfg['baseline512_dir']).resolve(),Path(tokenizer).resolve(),Path(llada_tokenizer).resolve()):
        if any(o==path or o in path.parents or path in o.parents for o in (out,private)):raise ValueError('Input/output overlap')
    references,views,qmap,reference_manifests,protected=load_inputs(phase2,cfg['baseline256_dir'],cfg['baseline512_dir'],tokenizer,llada_tokenizer)
    protected[str(run_dir)]=inventory(run_dir)
    protected[str(root/'reports/phase8')]=inventory(root/'reports/phase8')
    protected[str(root/'reports/phase8_5')]=inventory(root/'reports/phase8_5')
    verify_events(run_dir/'journal.jsonl');source_verification=verify_sources(manifest)
    if json.loads((Path(cfg['baseline512_dir'])/'environment.json').read_text())!=json.loads((run_dir/'environment.json').read_text()):raise ValueError('Original512/concise environment mismatch')
    for path in (run_dir,Path(cfg['baseline256_dir']),Path(cfg['baseline512_dir'])):
        completed=[json.loads(l)['question_id'] for l in (path/'journal.jsonl').open() if json.loads(l)['event']=='completed']
        if completed!=[q['question_id'] for q in questions]:raise ValueError('Question execution order differs')
    with tempfile.TemporaryDirectory() as temp:
        protocol.summarize(Path(temp),concise,questions,cfg)
        for name in ('results.csv','summary.csv'):compare_tables(run_dir/name,Path(temp)/name)
    for cap in (256,512):verify_saved_tables(Path(cfg[f'baseline{cap}_dir']),refs[cap][0],questions,refs[cap][1]['config'])
    # Preserve existing review forms and any ratings: never regenerate on top of them.
    if not private.exists():analyze(run_dir,private,tokenizer)
    if json.loads((private/'analysis_manifest.json').read_text())['run_hash']!=manifest['run_hash']:raise ValueError('Private analysis belongs to another run')
    with tempfile.TemporaryDirectory(dir=root/'results/phase9') as temp:
        check=Path(temp)/'analysis';analyze(run_dir,check,tokenizer)
        for name in ('per_question.csv','summary.csv','paired_correctness.csv','paired_outcomes.csv','recoveries_regressions.csv'):
            compare_tables(private/name,check/name)
        for name in ('review_packet/comparison.html','review_packet/README.md','unblinding/key.json'):
            if (private/name).read_bytes()!=(check/name).read_bytes():raise ValueError('Review randomization/content changed')
    ratings=read_csv(private/'review_packet/ratings.csv')
    rating_fields=('arithmetic_correctness','essential_step_completeness','logical_coherence','unsupported_jumps','final_answer_correctness')
    reviewed_rows=sum(any(r[k].strip() for k in rating_fields) for r in ratings)
    groups={'Qwen original-256':refs[256][0],'Qwen original-512':refs[512][0],'Qwen concise-256':concise,'LLaDA LTR P4':references['LLaDA LTR P4']}
    summaries=[]
    for name,rs in groups.items():
        gpu='gpu_seconds' if name=='LLaDA LTR P4' else 'gpu_prefill_decode_seconds'
        times=[r[gpu] for r in rs.values()]
        summaries.append(dict(method=name,n=len(rs),correct=sum(r['correct'] for r in rs.values()),accuracy_pct=100*mean(r['correct'] for r in rs.values()),
            gpu_mean=mean(times),gpu_median=median(times),gpu_p90=float(np.quantile(times,.9,method='linear')),timing_scope='decode-only full sampler' if name=='LLaDA LTR P4' else 'prefill + native cached decode'))
    strata={q['question_id']:q['stratum'] for q in questions}
    comparisons=[]
    for name in ('Qwen original-256','Qwen original-512','LLaDA LTR P4'):
        c=paired(groups[name],concise,strata,name,'Qwen concise-256')
        c['role']='primary' if name=='Qwen original-256' else 'exploratory_secondary';c['holm_two_secondary_p']=None;comparisons.append(c)
    running=0.
    for i,c in enumerate(sorted(comparisons[1:],key=lambda r:r['mcnemar_exact_p'])):
        running=max(running,min(1.,(2-i)*c['mcnemar_exact_p']));c['holm_two_secondary_p']=running
    strata_comparisons=[]
    for s in ('0-2','3-4','5+'):
        ids={q:v for q,v in strata.items() if v==s}
        c=paired({q:groups['Qwen original-256'][q] for q in ids},{q:concise[q] for q in ids},ids,'original-256','concise-256')
        c.update(stratum=s,original_correct=sum(groups['Qwen original-256'][q]['correct'] for q in ids),concise_correct=sum(concise[q]['correct'] for q in ids),exploratory=True);strata_comparisons.append(c)
    running=0.
    for i,c in enumerate(sorted(strata_comparisons,key=lambda r:r['mcnemar_exact_p'])):
        running=max(running,min(1.,(3-i)*c['mcnemar_exact_p']));c['holm_three_strata_p']=running
    audits,audit_summary=audit_errors(concise,'research/phase9_text_audit.json')
    diagnostic={name:last_mention_diagnostic(rs) for name,rs in groups.items() if name.startswith('Qwen')}
    sensitivity=paired(diagnostic['Qwen original-256'],diagnostic['Qwen concise-256'],strata,'original last-mention diagnostic','concise last-mention diagnostic')
    sensitivity['posthoc_not_official']=True
    sensitivity_rows=[]
    for q in sorted(strata):
        row=dict(question_id=q,stratum=strata[q])
        for condition,name in [('original','Qwen original-256'),('concise','Qwen concise-256')]:
            row[condition+'_official_correct']=groups[name][q]['correct'];row[condition+'_diagnostic_correct']=diagnostic[name][q]['correct'];row[condition+'_diagnostic_value']=diagnostic[name][q]['value']
        sensitivity_rows.append(row)
    audit_lookup={r['question_id']:r for r in audits}
    transition_lookup={int(r['question_id']):r for r in read_csv(private/'recoveries_regressions.csv')}
    budget_error_transitions=[]
    for qid in sorted(strata):
        budget_error_transitions.append(dict(question_id=qid,stratum=strata[qid],transition=transition_lookup[qid]['category'],original_capped=refs[256][0][qid]['termination']=='length_limit',concise_empty_marker=concise[qid]['response'].rstrip().endswith('#### <final numerical answer>'),incorrect_case_audited=qid in audit_lookup,evidence_tags=audit_lookup[qid]['tags'] if qid in audit_lookup else ''))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'budget_error_transitions.csv',budget_error_transitions)
    for name in ('per_question.csv','summary.csv','paired_outcomes.csv','recoveries_regressions.csv'):(out/name).write_bytes((private/name).read_bytes())
    for filename,rows in [('reference_summary.csv',summaries),('paired_correctness.csv',comparisons),('stratum_paired.csv',strata_comparisons),('text_audit.csv',audits),('error_taxonomy.csv',audit_summary),('parser_sensitivity.csv',sensitivity_rows),('parser_sensitivity_paired.csv',[sensitivity])]:write_csv(out/filename,rows)
    for name in ('accuracy_latency.png','output_lengths.png','termination.png'):(out/name).write_bytes((private/name).read_bytes())
    public_perq=read_csv(out/'per_question.csv');summary=read_csv(out/'summary.csv')
    provenance=dict(phase=9,analysis_version='finalization-v1',condition=protocol.CONDITION,instruction=protocol.CONCISE,model_revision=cfg['model_revision'],run_hash=manifest['run_hash'],configuration_hash=manifest['configuration_hash'],
        selection_sha256=manifest['selection_sha256'],input_sha256={name:{file:digest(path/file) for file in ('manifest.json','journal.jsonl','environment.json')} for name,path in [('concise',run_dir),('original256',Path(cfg['baseline256_dir'])),('original512',Path(cfg['baseline512_dir']))]},
        llada_input_sha256={n:digest(Path(phase2)/n) for n in ('manifest.json','journal.jsonl')},tokenizer_sha256=manifest['tokenizer_sha256'],tokenizer_config_sha256=manifest['tokenizer_config_sha256'],
        recorded_inference_source_sha256=manifest['source_sha256'],inference_source_verification=source_verification,
        analysis_environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=__import__('matplotlib').__version__),
        bootstrap=dict(seed=42,repetitions=10000,generator='NumPy PCG64 reset independently per contrast',method='Paired within-stratum resampling at observed stratum sizes; linear percentile 2.5/97.5; pointwise'),
        testing_families=dict(primary='One original256 vs concise256 exact two-sided McNemar test',secondary='Two comparisons, Holm adjusted separately',strata='Three exploratory stratum comparisons, Holm adjusted separately',parser_sensitivity='Posthoc diagnostic, not a new benchmark score'),
        review=dict(seed=cfg['review_seed'],n_pairs=60,n_rating_rows=len(ratings),rows_with_any_rating=reviewed_rows,reasoning_quality_verified=False,packet_relative_path='results/phase9/finalization/review_packet',unblinding_not_public=True),
        annotation_type='Unblinded assistant-assisted textual evidence audit; not human blinded review',annotation_sha256=digest('research/phase9_text_audit.json'),
        analysis_source_sha256={p:digest(p) for p in ('src/dllm_unmasking/finalize_phase9.py','src/dllm_unmasking/analyze_phase9.py','src/dllm_unmasking/phase9_review.py','src/dllm_unmasking/reasoning_budget.py')})
    atomic_json(out/'provenance.json',provenance)
    report(out,summaries,summary,comparisons,strata_comparisons,audit_summary,sensitivity,diagnostic,reviewed_rows,public_perq)
    for path,hashes in protected.items():
        if inventory(Path(path))!=hashes:raise RuntimeError('Historical artifacts changed')
    return comparisons


def report(out,refs,summary,comparisons,strata,audits,sensitivity,diagnostic,reviewed,perq):
    count={r['category']:r['n'] for r in audits}
    a=next(r for r in summary if r['condition']=='original' and r['stratum']=='all')
    b=next(r for r in summary if r['condition']=='concise' and r['stratum']=='all')
    token_reduction=100*(1-float(b['native_generated_tokens_mean'])/float(a['native_generated_tokens_mean']))
    ratio=float(a['gpu_prefill_decode_seconds_mean'])/float(b['gpu_prefill_decode_seconds_mean'])
    reduction=100*(1-1/ratio);primary=comparisons[0]
    reference_rows=[dict(method=r['method'],correct=f"{r['correct']}/{r['n']}",accuracy=f"{r['accuracy_pct']:.1f}%",mean=f"{r['gpu_mean']:.4f}",median=f"{r['gpu_median']:.4f}",p90=f"{r['gpu_p90']:.4f}",scope=r['timing_scope']) for r in refs]
    paired_rows=[dict(a=r['a'],role=r['role'],both=r['both_correct'],neither=r['both_incorrect'],recover=r['b_recovers'],regress=r['b_regresses'],delta=f"{r['b_minus_a_pp']:.2f}",ci=f"[{r['bootstrap_low_pp']:.2f}, {r['bootstrap_high_pp']:.2f}]",p=f"{r['mcnemar_exact_p']:.6g}",holm='—' if r['holm_two_secondary_p'] is None else f"{r['holm_two_secondary_p']:.6g}") for r in comparisons]
    budget=[]
    for r in (a,b):
        budget.append(dict(condition=r['condition'],n=r['n'],tokens=f"{float(r['native_generated_tokens_mean']):.2f}",words=f"{float(r['words_mean']):.2f}",chars=f"{float(r['chars_mean']):.2f}",eos=r['eos'],cap=r['length_limited'],marker=r['final_markers'],failed=r['extraction_failures'],answer_n=r['explicit_answer_available_n'],answer_mean=f"{float(r['first_explicit_prefix_tokens_mean']):.2f}"))
    stratum_rows=[dict(stratum=r['stratum'],original=f"{r['original_correct']}/20",concise=f"{r['concise_correct']}/20",recover=r['b_recovers'],regress=r['b_regresses'],delta=f"{r['b_minus_a_pp']:.1f}",ci=f"[{r['bootstrap_low_pp']:.1f}, {r['bootstrap_high_pp']:.1f}]",p=f"{r['mcnemar_exact_p']:.6g}",holm=f"{r['holm_three_strata_p']:.6g}") for r in strata]
    taxonomy=[dict(category=r['category'],n=r['n'],denom=r['incorrect_denominator']) for r in audits]
    diag_original=sum(r['correct'] for r in diagnostic['Qwen original-256'].values());diag_concise=sum(r['correct'] for r in diagnostic['Qwen concise-256'].values())
    text=f'''# Phase 9 — Concise Reasoning Intervention

## Abstract

A single preselected concise-reasoning instruction was tested on pinned
Qwen2.5-7B-Instruct with the same 60 GSM8K questions and a 256-token cap.
Official numerical accuracy was 22/60 (36.7%), versus 28/60 (46.7%) under the
original instruction: a −10.0 percentage-point change, with paired bootstrap
95% interval [{primary['bootstrap_low_pp']:.1f}, {primary['bootstrap_high_pp']:.1f}] and exact McNemar p={primary['mcnemar_exact_p']:.3f}.
The paired evidence does not establish either a population-level accuracy loss or
non-inferiority. Mean recorded GPU prefill+decode time fell from 5.3244 to 1.4791 s,
and every concise output ended at EOS rather than the token cap. However,
19/60 concise outputs ended with an empty final-answer placeholder, and visible
constraint/arithmetic mistakes coexist with extraction effects. Faster, shorter
outputs did not improve the preselected official accuracy metric in this subset.
This is a prompt intervention, not an architecture comparison. Blinded human
reasoning-quality review remains outstanding.

## Research question and hypotheses

Does the fixed instruction below improve numerical accuracy within Qwen at the
same nominal 256-token generation budget, by reducing budget spent on explanation?
Phase 8.5 motivated the prospective hypothesis: original Qwen-256 had 32 capped
outputs, whereas its 512-token continuation recovered many late final answers.
The directional accuracy hypothesis and brevity/termination outcomes preceded this
run. The detailed error taxonomy, parser sensitivity and stratum mechanisms below
are **posthoc exploratory analyses**, not preregistered mechanistic tests.
No wording variants were tuned on these outcomes.

## Experimental setup and integrity

Model/tokenizer: `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`; dataset `openai/gsm8k` main/test,
revision `740312add88f781978c0658806c59bc2815b9866`. Exactly the original 60 Phase 2
IDs, order, questions, gold answers and balanced 20/20/20 calculation-count strata
were reused. Selection seed 20261014; inference seed 42. Greedy decoding, KV cache,
BF16, SDPA, batch one, native chat template and EOS, at most 256 new tokens,
unchanged numerical evaluator, original GPU class and recorded software environment
were retained. The instruction occupies the same user-role position after a newline.

Original instruction:

```text
{protocol.ORIGINAL}
```

The one preselected intervention:

```text
{protocol.CONCISE}
```

Rendered prompts were checked to differ only by this exact substitution. Gold
answers were used for evaluation, not generation. All 60 concise completions match
valid starts; there are no pending or duplicate evaluations. Raw manifest/config,
model/tokenizer metadata, cached-source hashes, exact native response decoding,
question execution order and saved tables were checked. All ten recorded inference
source hashes match available source bytes. Original Qwen-256/512 reference hashes
match Phase 8 receipts. LLaDA LTR P4's Phase 2 raw journal, questions, labels and
timing fields match the immutable Phase 8.5/Phase 8 provenance. No journal or
historical result was altered.

## Main results and accuracy–latency comparison

{table(reference_rows,[('method','Configuration'),('correct','Correct/n'),('accuracy','Accuracy'),('mean','Mean GPU s'),('median','Median GPU s'),('p90','p90 GPU s'),('scope','Timing scope')])}

Concise versus original-256 reduces the recorded mean GPU interval by {reduction:.1f}%
(a {ratio:.2f}× ratio of means). This is an observed across-job latency comparison,
not a controlled hardware speedup. Qwen intervals include synchronized prefill plus
native cached decode and stream scheduling gaps; they exclude loading, tokenization
and final CPU text decoding. LLaDA intervals are decode-only full sampler intervals,
including repeated prompt conditioning, without a separately equivalent AR prefill.
Same GPU class/software does not eliminate scheduling or thermal variation. No
latency repetitions were collected. Cross-model dominance is not validated by
these unequal timing scopes. Original Qwen-512 and LLaDA are references, not reruns
or additional intervention arms.

![Recorded accuracy and latency](accuracy_latency.png)

## Paired correctness analysis

B is concise-256 in every row; Δ is B minus A. Counts match question IDs,
not independent strategy observations. Pointwise 95% bootstrap intervals use
10,000 paired within-stratum samples, 20 questions per stratum, NumPy PCG64 seed
42 reset per contrast, linear percentile quantiles. Exact two-sided McNemar uses
the binomial discordance test. There is one primary comparison; the two secondary
comparisons use a separate Holm family and do not change the primary p-value.

{table(paired_rows,[('a','A'),('role','Role'),('both','Both correct'),('neither','Both incorrect'),('regress','A correct → concise wrong'),('recover','A wrong → concise correct'),('delta','Δ pp'),('ci','95% CI pp'),('p','Exact p'),('holm','Secondary Holm p')])}

The primary has {primary['both_correct']} both-correct, {primary['both_incorrect']}
both-incorrect, {primary['b_regresses']} regressions and {primary['b_recovers']} recoveries.
A non-significant exact test is not evidence of equivalent accuracy. The interval
allows practically relevant loss and improvement. [Paired tables](paired_correctness.csv)
and [per-question transitions](recoveries_regressions.csv) preserve full counts and precision.

## Output length, EOS, answer placement and latency

{table(budget,[('condition','Condition'),('n','n'),('tokens','Mean generated tokens'),('words','Mean words'),('chars','Mean chars'),('eos','EOS'),('cap','Capped'),('marker','#### present'),('failed','Extraction failures'),('answer_n','Explicit answer available n/60'),('answer_mean','Mean first explicit answer prefix tokens')])}

Generated tokens decrease by {token_reduction:.1f}%; recorded latency decreases by
{reduction:.1f}%. These aggregate changes track one another, but do not identify a
kernel-level speedup or a pure causal latency mechanism.

Generated-token counts include EOS; visible-token counts exclude it. Words are
whitespace-delimited and characters are Unicode code points. Native tokenizer JSON
and config hashes match the pinned historical manifests; decoded raw and visible
text match all saved responses. Answer starts are zero-based, ends exclusive.
First complete explicit-answer availability is parser-defined, excludes fallback
and partial digit matches using full-output numerical spans, and does not certify
an intended correct answer or faithful reasoning. Missing explicit answers remain
missing, not token zero. Placement means are conditional on available answers;
this selection changes across conditions. The concise availability count includes
43/60 outputs with an explicit answer at some prefix, versus 41/60 that ultimately
parse successfully: questions 158 and 603 have an earlier marked/boxed number
subsequently overridden by an empty final marker. Availability is not a guarantee
of a valid final response. Full character/word/native placements,
mean/median/p90 lengths and timings are in [summary.csv](summary.csv) and
[per_question.csv](per_question.csv). Paired EOS/marker/length/latency intervals
are [exploratory secondary outcomes](paired_outcomes.csv), not simultaneous coverage.

![Output lengths](output_lengths.png)
![EOS and token-cap termination](termination.png)

## Error taxonomy: evidence rather than hidden-mechanism labels

All 38 officially incorrect concise responses were inspected in an **unblinded
assistant-assisted textual audit**, separate from the blinded human review below.
Every quote is verified verbatim against its stored response. Categories overlap
and are not independent events or population prevalence estimates. They describe
visible text relative to the problem/reference, not internal model computations.
This audit is not an independent human rating or evidence that reasoning quality
is preserved. Incorrect numbers alone do not establish arithmetic errors.

{table(taxonomy,[('category','Evidence category'),('n','Count'),('denom','Incorrect-case denominator')])}

- **Empty marked answers:** 19 responses terminate at EOS with the literal
  `#### <final numerical answer>` and no following number. Thirteen contain a
  preceding reported result equal to gold. These are output-contract failures
  under the specified marker-priority parser, not implementation bugs in that
  parser. Short output and EOS do not guarantee a valid final-answer format.
- **Parser selection effects:** questions 521 and 811 mark a final equation/rate,
  but the first-number rule selects the left operand rather than the stated result.
  The correct result is visible; official correctness remains unchanged.
- **Arithmetic/constraint evidence:** false displayed equations are checked locally;
  omitted terms or unit conversions are compared with explicit problem constraints.
  These can coexist with a malformed marker. A locally correct calculation on a
  mistaken setup is not mislabeled an arithmetic error.
- **Unsupported or unexposed work:** the audit distinguishes explicit unsupported
  transitions from 13 wrong answer-only outputs. The latter lack a visible
  derivation, so arithmetic, omitted steps and constraint interpretations remain
  ambiguous. Concise valid reasoning need not display every trivial operation.

[text_audit.csv](text_audit.csv) covers every incorrect question with evidence,
notes and annotation origin; [error_taxonomy.csv](error_taxonomy.csv) gives counts.

### Exploratory parser sensitivity, without changing official results

A uniform diagnostic selects the **last numerical mention in the entire visible
response**, independent of gold, for both original and concise outputs. It ignores
marker priority and can accidentally select intermediate quantities; it is neither
a recommended replacement evaluator nor a corrected benchmark score. It was
examined after seeing these outputs and is explicitly posthoc.

Under this diagnostic only, original-256 matches gold on {diag_original}/60 and
concise-256 on {diag_concise}/60 (instead of official 28/60 and 22/60). Its paired
Δ is {sensitivity['b_minus_a_pp']:.2f} pp, CI [{sensitivity['bootstrap_low_pp']:.2f}, {sensitivity['bootstrap_high_pp']:.2f}],
exact p={sensitivity['mcnemar_exact_p']:.6g}. Thirteen concise empty-marker cases and
the two equation-selection cases account for the 15 extra diagnostic matches.
That sensitivity materially changes the descriptive comparison but does not
establish faithful solutions, validate a parser change, or overturn the prespecified
primary endpoint. Per-question values and paired counts are published separately
in [parser_sensitivity.csv](parser_sensitivity.csv) and
[parser_sensitivity_paired.csv](parser_sensitivity_paired.csv).

## Representative paired examples

These are deliberately selected explanatory cases, not a random sample of errors.
The full two-condition outputs are in the local blinded packet, with the key kept
separate. Brief quotations below belong to the unblinded evidence audit, not
blinded ratings.

| Question | Original → concise official outcome | Textual evidence and bounded interpretation |
| --- | --- | --- |
| 118 | wrong → wrong | Concise explicitly computes 4 containers, then leaves an empty final marker. Original was capped. No claim that additional arithmetic is needed. |
| 211 | correct → wrong | Concise derives `r = 4` but ends with an empty marker. This regression is sensitive to output formatting, not proof of failed algebra. |
| 521 | correct → wrong | Concise ends `#### 52 - 13 = 39`; gold is 39, but the parser selects 52. |
| 975 | correct → wrong | Concise states Jerry has 30 cherries from `60 - 30 - 20`, which is 10; final 10 differs from requested difference 20. This is explicit arithmetic evidence. |
| 157 | wrong → wrong | Concise omits Kelly's initial full-field trip and gives 280 rather than 80. `3000 - 2720 = 280` is locally valid; the setup omits a term. |
| 268 | correct → wrong | Concise gives only `#### $505` versus gold 1375. The wrong value is observable; its internal cause is not. |
| 218 | wrong → correct | Original is capped; concise shows ages 66, 24 and 45 and their mean 45, then supplies a parseable answer. Brevity itself is not treated as worse reasoning. |

## Complexity strata: exploratory crossover

{table(stratum_rows,[('stratum','Calculation-count proxy'),('original','Original correct'),('concise','Concise correct'),('recover','Recoveries'),('regress','Regressions'),('delta','Δ pp'),('ci','95% CI pp'),('p','Exact p'),('holm','Holm p (3 strata)')])}

The 5+ group changes from 3/20 to 7/20, while 0–2 changes from 12/20 to 9/20 and
3–4 from 13/20 to 6/20. This pattern was not an independently powered hypothesis;
no subgroup establishes a significant difference under its exact test. Original
5+ outputs were capped on 17/20 questions, versus 7/20 and 8/20 in easier strata.
Concision removes that cap exposure, offering one plausible explanation for the
hard-group recoveries. But concise empty markers occur on 8/20 hard cases, 4/20
0–2 cases and 7/20 middle cases; error types coexist. The 5+ gain is five recoveries
and one regression, not evidence that the instruction improves deep reasoning.
Calculation count is a gold-solution proxy, not validated difficulty or reasoning
depth. The middle-stratum pointwise percentile interval excludes zero, while its
exact p-value is 0.0654 (Holm 0.1963); the hard-stratum interval touches zero. These
small-sample procedures need not agree. Do not select a favorable procedure after
seeing the subgroup pattern. [budget_error_transitions.csv](budget_error_transitions.csv)
links each recovery/regression to observed cap and marker status, with evidence
labels only for the audited incorrect cases. No mediation or internal-reasoning
mechanism is identified by these records.

## Blinded reasoning-quality review

A complete 60-pair packet has anonymous case IDs, A/B condition ordering and 120
rating rows. Question order uses Python random seed 20261015 and independent A/B
allocation seed 20261016. Review criteria cover arithmetic correctness, essential
intermediate-step completeness, coherence, unsupported jumps and final-answer
correctness. Shorter solutions are not automatically penalized. Only
`results/phase9/finalization/review_packet/` should be shared with reviewers; keep
`unblinding/` separate. Condition labels, original question IDs, timing/length
summaries and automated correctness labels are absent from the review packet.
Reviewers may infer conditions from style; this is label concealment, not guaranteed
successful blinding. Reviewers should avoid the unblinded report, evidence audit,
baseline outputs and key until their ratings are locked; matching a published
example could otherwise unblind a case. There are {reviewed}/120 rows with any supplied quality rating.
**No completed independent human reasoning-quality review is supplied; global
quality preservation or degradation remains unverified.** The unblinded textual
audit above does not substitute for it. Existing forms/keys are never overwritten.

## Limitations, conclusions and future directions

Verified: exact matched artifacts and prompt substitution; 22/60 official accuracy;
11 recoveries and 17 regressions; shorter outputs, all-EOS termination and lower
recorded latency; explicit empty-marker and parser-selection effects; and bounded
textual errors. Exploratory: subgroup crossover, parser sensitivity, evidence-tag
counts, and interpretations of individual changes. Untested: latent calculation
strategies, quality preservation, and whether omitted work caused a particular
wrong answer without visible evidence. Gold-matching numerical extraction is not
proof of faithful reasoning, and omitted prose is not proof of worse reasoning.

The preselected instruction did not improve the official accuracy point estimate,
but the primary paired uncertainty is too wide to claim a general accuracy loss
or equivalent performance. The recorded latency–accuracy trade-off is substantial;
output-format failures make a simple "shorter reasoning is worse" account unsupported.
These findings concern this instruction and balanced 60-question sample, with
no attribution to diffusion architecture. Questions, baseline outcomes and Phase 8.5
findings were already known; this is not a blinded holdout benchmark. Separate
GPU jobs and no timing repetitions limit latency inference.

First complete the existing blinded review. If a further intervention is later
authorized, test a separately prespecified **format-only instruction control**
that requests a bare numeric suffix without the literal placeholder, holding the
concise instruction otherwise fixed. Keep its results exploratory relative to this
locked Phase 9 instruction; do not tune/regrade the existing run or silently change
the evaluator. No new intervention, inference or prompt variant is implemented here.

## Reproducibility and publication checks

The raw concise run remains under `outputs/phase9/qwen_concise_256/run/`; original
Qwen and LLaDA journals, manifests, environments, logs and historical reports stay
unchanged. Public CSVs contain derived values and bounded evidence quotes, not
full responses, runtime logs, private allocation/host details or review keys.
[provenance.json](provenance.json) records input/source hashes, native-tokenizer
verification, exact statistics families and the distinction between textual audit
and unperformed human review. The original runtime environment matches both Qwen
baselines; analysis versions are recorded separately.

```bash
python -m dllm_unmasking.finalize_phase9 --phase2-results "$PHASE2_RESULTS"
```

Optional `--output-dir results/phase9/recheck_public` creates an independent derived
copy. The finalizer first regenerates/compares all saved numerical tables, verifies
recorded source bytes and exact question order, and checks every text-audit quote.
It preserves existing review forms. To regenerate the complete packet independently,
use a fresh `--private-dir results/phase9/recheck_private` too; share only its
review_packet/ directory. Local tokenizer assets remain ignored. No network,
weights, inference, submission, commit, push or history rewrite is used.
'''
    (out/'PHASE9_REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',default='outputs/phase9/qwen_concise_256/run')
    p.add_argument('--phase2-results',required=True)
    p.add_argument('--output-dir',default='reports/phase9')
    p.add_argument('--private-dir',default='results/phase9/finalization')
    p.add_argument('--tokenizer-dir',default='local_tokenizers/qwen')
    p.add_argument('--llada-tokenizer-dir',default='local_tokenizers/llada')
    a=p.parse_args();print(finalize(a.input_dir,a.phase2_results,a.output_dir,a.private_dir,a.tokenizer_dir,a.llada_tokenizer_dir))


if __name__=='__main__':enter_repository();main()
