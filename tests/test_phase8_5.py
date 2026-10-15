"""Offline Phase 8.5 token/text and publication tests; no private assets required."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from dllm_unmasking.reasoning_budget import (ByteLevelDecoder,placement,metrics,first_complete_gold,
    prefix_events,error_flags,paired_continuous,explicit_answer_candidates)
from dllm_unmasking.ar_baseline import digest
from dllm_unmasking.evaluation import extract_answer


def fixture(tmp):
    # Exact synthetic ByteLevel UTF-8 pieces: euro's three bytes span two tokens.
    vocab={'A':0,'Ġ':1,'#':2,'1':3,'5':4,'0':5,'âĤ':6,'¬':7,'!':8}
    data=dict(model=dict(type='BPE',vocab=vocab),decoder=dict(type='ByteLevel'),added_tokens=[dict(id=9,content='<eos>',special=True)])
    path=Path(tmp)/'tokenizer.json';path.write_text(json.dumps(data));return ByteLevelDecoder(path)


def record(text,ids,raw,gold='1'):
    value,mode=extract_answer(text)
    return dict(response=text,raw_response=raw,generated_token_ids=ids,eos_token_ids=[9],gold_numeric=gold,
        predicted_numeric=value,extraction_method=mode,correct=value==gold,extraction_failed=value is None,termination='eos' if ids[-1]==9 else 'length_limit')


class BudgetAnalysisTests(unittest.TestCase):
    def test_exact_native_offsets_and_multibyte_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=fixture(tmp);r=record('A € #### 15',[0,1,6,7,1,2,2,2,2,1,3,4,9],'A € #### 15<eos>','15')
            v=d.view(r)
            self.assertTrue(v.offsets_available)
            self.assertEqual(v.position(2,3),(2,4))
            self.assertEqual(v.position(9,11),(10,12))
            self.assertEqual(placement(r['response'],(9,11),v)['word_start'],3)
            self.assertEqual(v.position(4,8),(5,9))
            # Partial euro bytes decode with replacement until completed.
            self.assertEqual(list(v.prefixes())[2][1],'A �')

    def test_saved_text_mismatch_fails_without_offset_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=fixture(tmp)
            with self.assertRaises(ValueError):ByteLevelDecoder(Path(tmp)/'tokenizer.json','0'*64)
            with self.assertRaises(ValueError):d.view(record('wrong',[0,9],'A<eos>'))
            p=Path(tmp)/'unsupported.json';p.write_text(json.dumps(dict(model=dict(type='BPE'),decoder=dict(type='Other'))))
            with self.assertRaises(ValueError):ByteLevelDecoder(p)

    def test_invalid_utf8_offsets_are_missing_not_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=fixture(tmp);r=record('�',[6,9],'�<eos>');v=d.view(r)
            self.assertFalse(v.offsets_available);self.assertEqual(v.position(0,1),(None,None))

    def test_complete_number_rule_rejects_partial_correct_digits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=fixture(tmp);r=record('#### 150',[2,2,2,2,1,3,4,5,9],'#### 150<eos>','1');v=d.view(r)
            events=prefix_events(r,v,'1')
            self.assertEqual(events['first_any_parser_gold_native'],6)
            self.assertIsNone(events['first_complete_explicit_parser_gold_native'])
            self.assertIsNone(first_complete_gold(r['response'],'1',True))

    def test_marker_missing_is_not_zero_and_flags_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=fixture(tmp);r=record('A 15',[0,1,3,4],'A 15','1');v=d.view(r);m=metrics(r,v)
            self.assertIsNone(m['last_marker_native_start']);self.assertFalse(m['final_marker_present'])
            flags=error_flags(r)
            self.assertTrue(flags['incorrect_and_cap_limited']);self.assertTrue(flags['incorrect_missing_marker'])
            self.assertTrue(flags['incorrect_ambiguous_answer_role']);self.assertFalse(flags['incorrect_with_explicit_wrong_number'])

    def test_gold_intermediate_is_not_explicit_answer_or_parser_bug(self):
        self.assertEqual(first_complete_gold('We have 15 apples. #### 10','15'),(8,10))
        self.assertIsNone(first_complete_gold('We have 15 apples. #### 10','15',True))
        self.assertEqual(explicit_answer_candidates('#### \n#### 10'),[(11,13)])

    def test_continuous_paired_mean_preserves_fractional_differences(self):
        rows=[dict(method=m,question_id=q,stratum=('0-2','3-4','5+')[q//2],correct=True,words=(q%2 if m=='B' else 0)) for m in ('A','B') for q in range(6)]
        result=paired_continuous(rows,'A','B','words',repetitions=100)
        self.assertEqual(result['mean_b_minus_a'],.5)
        self.assertEqual(result,paired_continuous(rows,'A','B','words',repetitions=100))
        with self.assertRaises(ValueError):paired_continuous(rows[:-1],'A','B','words')

    def test_public_native_metrics_and_recovery_claims(self):
        source=Path('reports/phase8_5')
        with (source/'per_question.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),180)
        self.assertEqual(len({(r['method'],r['question_id']) for r in rows}),180)
        for method,correct in [('LLaDA LTR P4',37),('Qwen-256',28),('Qwen-512',50)]:
            rs=[r for r in rows if r['method']==method]
            self.assertEqual(len(rs),60);self.assertEqual(sum(r['correct']=='True' for r in rs),correct)
            self.assertTrue(all(r['offset_mapping_exact']=='True' for r in rs))
        with (source/'budget_transitions.csv').open() as f:trans=list(csv.DictReader(f))
        recoveries=[r for r in trans if r['transition']=='recovery']
        self.assertEqual(len(recoveries),23)
        self.assertEqual(sum(r['transition']=='regression' for r in trans),1)
        self.assertTrue(all(int(r['first_complete_explicit_parser_gold_native_512'])>256 for r in recoveries))
        self.assertEqual(sum(r['llada_correct']=='True' and r['qwen256_correct']=='False' for r in trans),16)
        provenance=json.loads((source/'provenance.json').read_text())
        self.assertIn('not independently authenticated',provenance['tokenizers']['llada']['verification'])
        self.assertEqual(provenance['bootstrap']['seed'],42)

    def test_public_paired_means_recompute(self):
        with Path('reports/phase8_5/per_question.csv').open() as f:rows=list(csv.DictReader(f))
        rs=[dict(r,question_id=int(r['question_id']),correct=r['correct']=='True',**{k:float(r[k]) for k in ('chars','words','native_visible_tokens','nonempty_lines','numbered_step_lines')}) for r in rows]
        with Path('reports/phase8_5/paired_lengths.csv').open() as f:contrasts=list(csv.DictReader(f))
        for c in contrasts:
            derived=paired_continuous(rs,c['a'],c['b'],c['metric'],correct_both=c['subset']=='both_numerically_correct')
            for key in ('n','mean_b_minus_a','bootstrap_low','bootstrap_high'):self.assertAlmostEqual(float(c[key]),derived[key],places=10)

    def test_new_source_provenance_and_report_caveats(self):
        p=Path('reports/phase8_5');provenance=json.loads((p/'provenance.json').read_text())
        for source,sha in provenance['source_sha256'].items():self.assertEqual(digest(source),sha)
        report=(p/'PHASE8_5_REPORT.md').read_text()
        for phrase in ('zero-based','not a GPU experiment','not automatically parser mistakes','revision identity','one targeted follow-up'):
            self.assertIn(phrase,report)


    def test_reclassification_preserves_scientific_artifacts_and_conclusions(self):
        import hashlib
        p=Path('reports/phase8_5');receipt=json.loads((p/'renaming_provenance.json').read_text())
        for name,sha in receipt['scientific_artifacts_sha256'].items():self.assertEqual(digest(p/name),sha)
        original=json.dumps(receipt['original_provenance'],indent=2)+'\n'
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),receipt['original_provenance_sha256'])
        report=(p/'PHASE8_5_REPORT.md').read_text()
        conclusions=report.split('## Does compactness explain the gap?')[1].split('## Reproduction and preservation')[0]
        self.assertEqual(hashlib.sha256(conclusions.encode()).hexdigest(),receipt['scientific_conclusions_sha256'])
        provenance=json.loads((p/'provenance.json').read_text())
        self.assertEqual(provenance['phase'],'8.5')
        self.assertEqual(provenance['naming_cleanup']['receipt_sha256'],digest(p/'renaming_provenance.json'))
        self.assertIn('Phase 9 is reserved for a future Qwen-256',report)
