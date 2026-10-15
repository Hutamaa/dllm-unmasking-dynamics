"""Pure offline text/token observations, not reasoning-quality classifications."""
import json
import hashlib
from pathlib import Path
import re

import numpy as np
from dllm_unmasking.evaluation import NUMBER, extract_answer
from dllm_unmasking.prepare_ltr_audit import selected_span, numeric


class ByteLevelDecoder:
    """Decode stored BPE IDs directly; never encode text or load model weights.

    Only the explicit ByteLevel decoder contract is supported. Offsets are used
    only when complete decoding exactly reproduces the stored raw/visible text.
    UTF-8 bytes can span multiple tokens; span ends count the last covering token.
    """
    def __init__(self, path, expected_sha256=None):
        if expected_sha256 is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError('Native tokenizer checksum mismatch; no approximate offsets')
        data=json.loads(Path(path).read_text())
        if data['model']['type']!='BPE' or data['decoder']['type']!='ByteLevel':
            raise ValueError('Unsupported native tokenizer decoder; no offset approximation')
        bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
        for b in range(256):
            if b not in bs:bs.append(b);cs.append(256+n);n+=1
        inverse={chr(c):b for b,c in zip(bs,cs)}
        self.pieces={i:bytes(inverse[c] for c in text) for text,i in data['model']['vocab'].items()}
        self.special=set()
        for token in data['added_tokens']:
            self.pieces[token['id']]=token['content'].encode('utf-8')
            if token.get('special'):self.special.add(token['id'])

    def raw_bytes(self, ids, skip_special=False):
        return b''.join(self.pieces[i] for i in ids if not skip_special or i not in self.special)

    def decode(self, ids, skip_special=False):
        return self.raw_bytes(ids,skip_special).decode('utf-8',errors='replace')

    def view(self, record):
        ids=record['generated_token_ids']
        eos=record.get('eos_token_ids',[record.get('eos_token_id')])
        stop=next((i for i,t in enumerate(ids) if t in eos),len(ids))
        if self.decode(ids)!=record['raw_response'] or self.decode(ids[:stop],True)!=record['response']:
            raise ValueError('Native decoder does not reproduce saved text; exact offsets unavailable')
        return TokenView(self,ids[:stop],record['response'])


class TokenView:
    def __init__(self, decoder, ids, text):
        self.ids=ids;self.text=text;self.decoder=decoder
        self.spans=[];self.ends=[];offset=0
        for i,token in enumerate(ids):
            piece=b'' if token in decoder.special else decoder.pieces[token]
            if piece:self.spans.append((offset,offset+len(piece),i))
            offset+=len(piece);self.ends.append(offset)
        self.bytes=decoder.raw_bytes(ids,True)
        # Replacement decoding cannot supply an exact character/byte alignment.
        try:self.offsets_available=self.bytes.decode('utf-8',errors='strict')==text
        except UnicodeDecodeError:self.offsets_available=False

    def position(self, start, end):
        if not self.offsets_available or not 0<=start<end<=len(self.text):return None,None
        lo=len(self.text[:start].encode('utf-8'));hi=len(self.text[:end].encode('utf-8'))
        covering=[i for a,b,i in self.spans if b>lo and a<hi]
        return (covering[0],covering[-1]+1) if covering else (None,None)

    def prefixes(self):
        buffer=b''
        for i,t in enumerate(self.ids):
            if t not in self.decoder.special:buffer+=self.decoder.pieces[t]
            # This is the native decoder's replacement policy at a token boundary.
            yield i+1,buffer.decode('utf-8',errors='replace')


def word_spans(text):return [m.span() for m in re.finditer(r'\S+',text)]


def placement(text, span, view=None):
    """All starts zero-based; ends exclusive. Native span uses original output IDs."""
    if span is None:
        return dict(char_start=None,char_end=None,word_start=None,word_end=None,native_start=None,native_end=None)
    lo,hi=span;words=word_spans(text)
    covering=[i for i,(a,b) in enumerate(words) if b>lo and a<hi]
    ns,ne=view.position(lo,hi) if view else (None,None)
    return dict(char_start=lo,char_end=hi,word_start=covering[0] if covering else None,
                word_end=covering[-1]+1 if covering else None,native_start=ns,native_end=ne)


def explicit_answer_candidates(text):
    """First complete numerical span per explicit marker/phrase/box; not reasoning labels."""
    spans=[]
    for match in re.finditer(r'####|(?:final answer|answer is|answer:)\s*[:=]?\s*',text,re.I):
        later=re.search(NUMBER,text[match.end():])
        # Never cross a subsequent explicit #### marker to invent a marker answer.
        if later and '####' not in text[match.end():match.end()+later.start()]:
            spans.append((match.end()+later.start(),match.end()+later.end()))
    for match in re.finditer(r'\\boxed\{([^{}]*)\}',text):
        number=re.search(NUMBER,match.group(1))
        if number:spans.append((match.start(1)+number.start(),match.start(1)+number.end()))
    return sorted(set(spans))


def first_complete_gold(text,gold,explicit=False):
    spans=explicit_answer_candidates(text) if explicit else [m.span() for m in re.finditer(NUMBER,text)]
    for span in spans:
        if numeric(text[slice(*span)])==gold:return span
    return None


def prefix_events(record,view,gold):
    """Observe official parser on stored native prefixes, with no inference.

    'Complete' excludes a substring that is just the first digits of a larger
    full-output numerical mention. This uses future full-output text as an audit
    reference and is not an online stopping rule. Correct prefixes may be transient.
    """
    full_spans={m.span() for m in re.finditer(NUMBER,record['response'])}
    first_any=first_explicit=first_raw=None
    correctness=[]
    for count,text in view.prefixes():
        value,method=extract_answer(text);span=selected_span(text)
        ok=value==gold;correctness.append(ok)
        if ok and first_raw is None:first_raw=count
        if ok and span in full_spans:
            if first_any is None:first_any=count
            if method!='last_number' and first_explicit is None:first_explicit=count
    suffix_stable=None
    if correctness and correctness[-1]:
        suffix_stable=len(correctness)
        while suffix_stable>1 and correctness[suffix_stable-2]:suffix_stable-=1
    return dict(first_any_parser_gold_native=first_raw,first_complete_parser_gold_native=first_any,
                first_complete_explicit_parser_gold_native=first_explicit,stable_parser_gold_from_native=suffix_stable,
                parser_correct_native_prefixes=sum(correctness),native_prefixes_observed=len(correctness))


def metrics(record,view):
    text=record['response'];gold=record['gold_numeric'];span=selected_span(text)
    marker=list(re.finditer('####',text));marker_span=marker[-1].span() if marker else None
    explicit=first_complete_gold(text,gold,True);any_gold=first_complete_gold(text,gold,False)
    native=len(view.ids);words=len(word_spans(text))
    row=dict(chars=len(text),words=words,native_visible_tokens=native,native_generated_tokens=len(record['generated_token_ids']),
             offset_mapping_exact=view.offsets_available,nonempty_lines=sum(bool(x.strip()) for x in text.splitlines()),
             numbered_step_lines=len(re.findall(r'^\s*(?:step\s+\d+|\d+[.)]\s)',text,re.I|re.M)),
             sentence_units=sum(bool(x.strip()) for x in re.split(r'[.!?](?=\s|$)',text)),
             arithmetic_symbol_lines=sum(bool(re.search(r'[=+*/×÷]|\d\s*-\s*\d',x)) for x in text.splitlines() if x.strip()),
             final_marker_present=bool(marker),marker_count=len(marker),predicted_numeric=record['predicted_numeric'],
             correct=record['correct'],extraction_failed=record['extraction_failed'],extraction_method=record['extraction_method'],
             any_gold_mention=any_gold is not None,explicit_gold_mention=explicit is not None,
             selected_number_touches_output_end=bool(span and span[1]==len(text)),
             max_repeated_word_run=max_word_run(text))
    for label,location in [('last_marker',marker_span),('selected_answer',span),('first_explicit_gold',explicit),('first_any_gold',any_gold)]:
        row.update({label+'_'+k:v for k,v in placement(text,location,view).items()})
    row.update(prefix_events(record,view,gold))
    return row


def max_word_run(text):
    longest=0;last=None;run=0
    for word in re.findall(r'\w+',text.lower()):
        run=run+1 if word==last else 1;last=word;longest=max(longest,run)
    return longest


def error_flags(record):
    """Non-exclusive observations; never auto-assign an arithmetic/reasoning error."""
    wrong=not record['correct'];text=record['response'];sp=selected_span(text)
    marker='####' in text;explicit=bool(explicit_answer_candidates(text))
    return dict(incorrect=wrong,cap_limited=record['termination']=='length_limit',
        incorrect_and_cap_limited=wrong and record['termination']=='length_limit',
        incorrect_missing_marker=wrong and not marker,
        incorrect_with_explicit_wrong_number=wrong and record['predicted_numeric'] is not None and record['extraction_method']!='last_number',
        incorrect_with_marker_wrong_number=wrong and marker and record['predicted_numeric'] is not None,
        incorrect_extraction_failure=wrong and record['extraction_failed'],
        incorrect_ambiguous_answer_role=wrong and not explicit,
        incorrect_with_gold_elsewhere=wrong and first_complete_gold(text,record['gold_numeric']) is not None,
        incorrect_with_explicit_gold_elsewhere=wrong and first_complete_gold(text,record['gold_numeric'],True) is not None,
        incomplete_numeric_candidate_at_boundary=wrong and record['termination']=='length_limit' and sp is not None and sp[1]==len(text))


def paired_continuous(rows,a,b,field,seed=42,repetitions=10000,correct_both=False):
    ar={r['question_id']:r for r in rows if r['method']==a}
    br={r['question_id']:r for r in rows if r['method']==b}
    if set(ar)!=set(br):raise ValueError('Unmatched question set')
    ids=sorted(q for q in ar if not correct_both or ar[q]['correct'] and br[q]['correct'])
    differences=np.array([br[q][field]-ar[q][field] for q in ids],dtype=float)
    if not ids:raise ValueError('Empty paired subset')
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.zeros(repetitions)
    for stratum in ('0-2','3-4','5+'):
        ds=differences[[i for i,q in enumerate(ids) if ar[q]['stratum']==stratum]]
        if len(ds):draws+=rng.choice(ds,(repetitions,len(ds)),replace=True).sum(axis=1)
    lo,hi=np.quantile(draws/len(ids),[.025,.975],method='linear')
    return dict(a=a,b=b,metric=field,subset='both_numerically_correct' if correct_both else 'all',n=len(ids),
                mean_b_minus_a=float(differences.mean()),bootstrap_low=float(lo),bootstrap_high=float(hi))
