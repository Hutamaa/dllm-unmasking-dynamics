"""Create a manual-audit packet from stored Phase 2 outputs; no inference."""
import argparse
import csv
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import re
from collections import Counter

from analyze_phase25 import load,outputs
from evaluation import NUMBER,extract_answer


def numeric(value):
    try:
        value=value.replace(',','').replace(' ','')
        if '/' in value:
            a,b=value.split('/');return str(Fraction(a)/Fraction(b))
        return str(Fraction(value))
    except (ValueError,ZeroDivisionError):return None


def candidates(text):
    return [dict(index=i+1,start=m.start(),end=m.end(),text=m.group(),normalized=numeric(m.group()),
                 context=text[max(0,m.start()-50):min(len(text),m.end()+50)])
            for i,m in enumerate(re.finditer(NUMBER,text))]


def selected_span(text):
    """Trace the unchanged parser's chosen numeric substring, including parse failure."""
    start=0;mode='last_number';end=len(text)
    if '####' in text:
        start=text.rfind('####')+4;mode='marker'
    else:
        boxed=list(re.finditer(r'\\boxed\{([^{}]*)\}',text))
        if boxed:start,end=boxed[-1].span(1);mode='boxed'
        else:
            phrases=list(re.finditer(r'(?:final answer|answer is|answer:)\s*[:=]?\s*',text,re.I))
            if phrases:start=phrases[-1].end();mode='answer_phrase'
    matches=list(re.finditer(NUMBER,text[start:end]))
    if not matches:return None
    match=matches[-1] if mode=='last_number' else matches[0]
    return start+match.start(),start+match.end()


def highlight(text,selected=None):
    parts=[];previous=0
    for c in candidates(text):
        parts.append(html.escape(text[previous:c['start']]))
        cls='selected' if selected==(c['start'],c['end']) else 'number'
        parts.append(f'<mark class="{cls}" title="Numerical mention #{c["index"]}; normalized {html.escape(str(c["normalized"]))}">{html.escape(c["text"])}</mark>')
        previous=c['end']
    parts.append(html.escape(text[previous:]))
    return ''.join(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',default='~/Documents/dllm-unmasking-results/phase2')
    parser.add_argument('--output-dir',default='results/phase25/ltr_manual_audit')
    args=parser.parse_args();source=Path(args.input_dir).expanduser().resolve();out=Path(args.output_dir).resolve()
    if source==out or source in out.parents or out in source.parents:parser.error('Input and output must not overlap')
    files=[p for p in source.iterdir() if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    records,lookup,manifest,events=load(source)
    ids=sorted(q for q,s,p in lookup if s=='left_to_right' and p==4 and lookup[q,s,4]['correct'] and not lookup[q,s,8]['correct'])
    if len(ids)!=33:raise ValueError(f'Expected the 33 requested LTR losses, found {len(ids)}')
    diagnostic={ (r['question_id'],r['parallelism']):r for r in outputs([r for r in records if r['strategy']=='left_to_right'],manifest['config']['mask_id']) }
    annotations=[];mentions=[];cards=[]
    for qid in ids:
        q=lookup[qid,'left_to_right',4]
        panels=[]
        for p in (4,8):
            r=lookup[qid,'left_to_right',p];text=r['response'];span=selected_span(text);cs=candidates(text)
            extracted,method=extract_answer(text)
            assert (extracted,method)==(r['predicted_numeric'],r['extraction_method'])
            chosen=next((c for c in cs if (c['start'],c['end'])==span),None)
            if chosen:assert chosen['normalized']==extracted
            d=diagnostic[qid,p]
            for view,t in [('visible',text),('raw',r['raw_response'])]:
                for c in candidates(t):
                    mentions.append(dict(question_id=qid,parallelism=p,view=view,**c,
                        parser_selected=view=='visible' and (c['start'],c['end'])==span,
                        matches_gold=c['normalized']==r['gold_numeric'],
                        # Same value occurring in calculations is not automatically a final answer.
                        role_human='',notes_human=''))
            annotations.append(dict(question_id=qid,parallelism=p,stratum=r['stratum'],
                stored_correct=r['correct'],gold_numeric=r['gold_numeric'],extracted_numeric=extracted,
                extraction_method=method,parser_selected_text=chosen['text'] if chosen else '',
                numerical_mentions_visible=len(cs),gold_value_mentions_visible=sum(c['normalized']==r['gold_numeric'] for c in cs),
                repeated_trigram_fraction=d['repeated_trigram_fraction'],max_word_run=d['max_consecutive_word_run'],
                max_character_run=d['max_identical_character_run'],first_eos_position=d['first_eos_position'],
                intended_final_answer_human='',parser_mistake_human='',arithmetic_error_human='',
                text_degeneration_human='',reasoning_quality_human='',evidence_spans_human='',notes_human=''))
            candidate_list='; '.join(f'#{c["index"]}: <code>{html.escape(c["text"])}</code> → {html.escape(str(c["normalized"]))}'+
                                    (' <strong>[PARSER]</strong>' if (c['start'],c['end'])==span else '')+
                                    (' [same value as gold]' if c['normalized']==r['gold_numeric'] else '') for c in cs)
            panels.append(f'''<section><h3>Parallel-{p}</h3><p><strong>Extracted: {html.escape(str(extracted))}</strong> · gold: {html.escape(r['gold_numeric'])} · stored correct: {r['correct']} · method: {html.escape(method)}</p>
<p>Longest identical-word run: {d['max_consecutive_word_run']}; literal character run: {d['max_identical_character_run']}; first EOS token position: {d['first_eos_position']}.</p>
<pre>{highlight(text,span)}</pre><details><summary>All visible numerical mentions ({len(cs)})</summary><p>{candidate_list or 'None'}</p></details>
<details><summary>Full raw response, including special tokens and post-EOS text</summary><pre>{highlight(r['raw_response'])}</pre></details></section>''')
        cards.append(f'<article id="q{qid}"><h2>Question {qid} · stratum {q["stratum"]}</h2><p>{html.escape(q["question"])}</p><details><summary>Gold solution (reference, not automatically an error annotation)</summary><pre>{highlight(q["gold_answer"])}</pre></details><div class="pair">'+''.join(panels)+'</div></article>')
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('annotations.csv',annotations),('numerical_candidates.csv',mentions)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>LTR P4 versus P8 manual audit</title><style>
body{font-family:system-ui;margin:24px;max-width:1600px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{min-width:0;border:1px solid #ccc;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 monospace}.number{background:#fff2a8}.selected{background:#aee8ff;border-bottom:3px solid #147ca1}article{border-top:3px solid #333;margin-top:30px}summary{cursor:pointer}@media(max-width:800px){.pair{grid-template-columns:1fr}}@media print{article{break-before:page}}
</style><h1>LTR P4-correct / P8-incorrect: all 33 matched cases</h1><p>Yellow highlights every numerical mention; blue identifies the unchanged parser's selected substring. Numerical mentions include intermediate quantities, not just asserted final answers. Hover for normalized values. Inspect both outputs: P4 numerical correctness does not establish sound reasoning.</p><p>Annotate parser mistakes, arithmetic errors, and text degeneration separately in annotations.csv. They may co-occur. Human fields are deliberately blank; no error type is automatically assigned.</p><nav>'''+ ' · '.join(f'<a href="#q{q}">{q}</a>' for q in ids)+'</nav>'+''.join(cards)+'</html>'
    (out/'comparison.html').write_text(page)
    guide='''# Manual LTR audit

Open `comparison.html` in a browser. It includes all 33 LTR P4-correct/P8-incorrect questions, full side-by-side visible responses, expandable full raw responses, and gold solutions. All numerical mentions are highlighted; the primary parser-selected substring is blue. The complete mention inventory, offsets, normalized values and context are in `numerical_candidates.csv` (visible and raw views are separate).

Work on a COPY of `annotations.csv` (66 rows, one per question/setting). Review P4 as well as P8. Stored correctness measures numerical extraction only.

Use each error field independently: `yes`, `no`, or `uncertain`, with quoted evidence and character offsets. Blank means not reviewed. Multiple errors can coexist.

- **Parser mistake:** the extractor selects or fails to parse a clearly intended final answer. Write that intended answer and cite the exact final-answer wording. Finding a gold-valued intermediate number somewhere is insufficient. A malformed or missing final answer is not automatically a parser bug.
- **Arithmetic error:** an explicit, locally checkable calculation is false. Quote the equation, provide the corrected calculation, and distinguish it from invalid premises or logic. A wrong extracted number alone is insufficient.
- **Text degeneration:** describe observed repetitive, garbled, or syntactically broken spans. The recorded run lengths and repetition fractions are descriptive aids, not ground-truth thresholds.
- **Reasoning quality:** independently choose `supported`, `unsupported`, or `uncertain` for the solution's derivation and cite evidence. Check valid premises, coherent operations, and agreement between reasoning and final answer; numerical correctness by coincidence does not establish successful reasoning.

`role_human` in numerical_candidates.csv may be annotated `given_quantity`, `intermediate`, `asserted_final`, or `uncertain`. A mention's match to gold is a value comparison, not an answer-role or reasoning label.

Raw text can include special tokens and text beyond the runner's first EOS. Do not silently substitute a raw/post-EOS answer for the stored visible answer. Note termination issues separately. Gold solutions are references and may themselves require checking. Prefer a second blinded reviewer and reconcile disagreements before calculating error-category frequencies.

No automatic parser/arithmetic/degeneration classifications or corrected accuracy estimates are produced. No model calls or original result edits occur.
'''
    (out/'README.md').write_text(guide)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    (out/'metadata.json').write_text(json.dumps(dict(question_ids=ids,n_questions=33,n_annotations=66,
        input_sha256=hashes,primary_parser='unchanged evaluation.extract_answer',inputs_unchanged=True),indent=2)+'\n')
    print(f'Prepared 33-case manual audit: {out/"comparison.html"}; 66 blank human annotation rows.')


if __name__=='__main__':main()
