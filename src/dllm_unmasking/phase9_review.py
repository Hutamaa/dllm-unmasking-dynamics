"""Deterministic blinded paired review packet; decoding condition key kept separately."""
import html
import json
from pathlib import Path
import random

from dllm_unmasking.analyze_phase7 import write_csv
from dllm_unmasking.phase2_data import atomic_json

CRITERIA='''# Blinded reasoning-quality review

Share only this review_packet directory. The separate unblinding directory is for
the study custodian and must stay hidden until independent ratings are locked.
No condition labels, original question IDs, timing/length summaries, automated
correctness labels, or randomization key are included here. Output style may make
condition guessable; blinding is not guaranteed by label concealment alone.

Rate A and B independently, using the question and reference final numerical
answer. Do not require either output to match a gold solution's exact approach.
Shorter reasoning is not automatically worse; omitted trivial steps are allowed.

- Arithmetic correctness: correct / incorrect / unclear. Check explicit numerical
  operations; quote any false calculation and its correction. A wrong final value
  alone is not sufficient evidence of an arithmetic error.
- Essential-step completeness: complete / missing_necessary / unclear. Determine
  whether the necessary transformations are justified, allowing concise valid
  derivations and implicit trivial operations.
- Logical coherence: coherent / incoherent / unclear. Check that quantities,
  assumptions and operations address the question without contradictions.
- Unsupported jumps: none / consequential / unclear. Identify conclusions that
  do not follow from available premises; do not penalize brevity itself.
- Final-answer correctness: correct / incorrect / missing / unclear. Assess the
  intended final answer against the question/reference, separately from whether
  a numerical parser would extract it correctly. Record ambiguities explicitly.

Provide evidence spans/quotes and notes for each rating. Fields start blank:
blank means not reviewed. Use at least two independent reviewers where feasible;
retain originals, reconcile disagreements transparently, and lock a dated copy
before consulting the key. Do not claim reasoning quality was preserved merely
because numerical accuracy improved. No human evidence is generated automatically.
'''


def make_review(out,questions,original,concise,seed):
    out=Path(out);packet=out/'review_packet';keys=out/'unblinding'
    if packet.exists() or keys.exists():raise ValueError('Never overwrite review forms or keys; select a fresh analysis directory')
    packet.mkdir(parents=True);keys.mkdir(parents=True)
    ids=[q['question_id'] for q in questions];random.Random(seed).shuffle(ids)
    rng=random.Random(seed+1);qmap={q['question_id']:q for q in questions}
    cards=[];forms=[];key=[]
    for index,qid in enumerate(ids,1):
        case=f'C{index:03d}';q=qmap[qid]
        order=['original','concise'];rng.shuffle(order);panels=[]
        for slot,condition in zip(('A','B'),order):
            record=original[qid] if condition=='original' else concise[qid]
            panels.append(f'<section><h3>{slot}</h3><pre>{html.escape(record["response"])}</pre></section>')
            forms.append(dict(case_id=case,slot=slot,reviewer_id='',arithmetic_correctness='',essential_step_completeness='',logical_coherence='',unsupported_jumps='',final_answer_correctness='',evidence='',notes=''))
            key.append(dict(case_id=case,slot=slot,question_id=qid,condition=condition))
        gold=original[qid]['gold_numeric']
        cards.append(f'<article id="{case}"><h2>{case}</h2><p>{html.escape(q["question"])}</p><p>Reference final numerical answer: {html.escape(gold)}</p><div class="pair">'+''.join(panels)+'</div></article>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired mathematical solution review</title><style>body{font-family:system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}section{min-width:0;padding:12px;border:1px solid #bbb}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:3px solid #333;margin-top:30px}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>Paired mathematical solution review</h1><p>Read README.md before rating. Assess A and B independently; numerical correctness and reasoning quality are separate.</p>'+''.join(cards)+'</html>'
    (packet/'comparison.html').write_text(page);(packet/'README.md').write_text(CRITERIA)
    write_csv(packet/'ratings.csv',forms)
    atomic_json(keys/'key.json',dict(seed=seed,question_order_stream=f'Python random.Random({seed})',condition_order_stream=f'Python random.Random({seed+1})',key=key))
    return dict(n_questions=len(ids),n_rating_rows=len(forms),seed=seed,human_reviews_received=0,reasoning_quality_assessed=False)
