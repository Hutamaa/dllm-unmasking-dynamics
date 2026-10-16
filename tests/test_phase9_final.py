"""Completed Phase 9 publication checks from released metrics; no inference/raw dependency."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from dllm_unmasking.ar_baseline import digest
from dllm_unmasking.finalize_phase8 import paired
from dllm_unmasking.finalize_phase9 import audit_errors,compare_tables,finalize


def csv_rows(name):
    with Path(name).open() as f:return list(csv.DictReader(f))


class FinalPhase9Tests(unittest.TestCase):
    def test_primary_and_secondary_counts_recompute(self):
        rows=csv_rows('reports/phase9/per_question.csv');groups={};strata={}
        for r in rows:
            name='Qwen original-256' if r['condition']=='original' else 'Qwen concise-256'
            q=int(r['question_id']);strata[q]=r['stratum'];groups.setdefault(name,{})[q]=dict(correct=r['correct']=='True')
        for r in csv_rows('reports/phase8/per_question.csv'):
            mapping={'Qwen-512':'Qwen original-512','left_to_right P4':'LLaDA LTR P4'}
            if r['method'] in mapping:groups.setdefault(mapping[r['method']],{})[int(r['question_id'])]=dict(correct=r['correct']=='True')
        for row in csv_rows('reports/phase9/paired_correctness.csv'):
            result=paired(groups[row['a']],groups[row['b']],strata,row['a'],row['b'])
            for key in ('n','both_correct','both_incorrect','b_recovers','b_regresses','b_minus_a_pp','bootstrap_low_pp','bootstrap_high_pp','mcnemar_exact_p'):
                self.assertAlmostEqual(float(row[key]),result[key],places=12)
        primary=csv_rows('reports/phase9/paired_correctness.csv')[0]
        self.assertEqual([int(primary[k]) for k in ('both_correct','both_incorrect','b_regresses','b_recovers')],[11,21,17,11])

    def test_error_audit_covers_all_failures_and_origin_is_explicit(self):
        rows=csv_rows('reports/phase9/per_question.csv');wrong={int(r['question_id']) for r in rows if r['condition']=='concise' and r['correct']=='False'}
        audit=csv_rows('reports/phase9/text_audit.csv')
        self.assertEqual(len(audit),38);self.assertEqual({int(r['question_id']) for r in audit},wrong)
        self.assertTrue(all('not independent human review' in r['annotation_type'] for r in audit))
        counts={r['category']:int(r['n']) for r in csv_rows('reports/phase9/error_taxonomy.csv')}
        self.assertEqual(counts['empty_final_marker'],19);self.assertEqual(counts['gold_valued_result_before_empty_marker'],13)
        self.assertEqual(counts['parser_selection_effect'],2);self.assertEqual(counts['no_visible_derivation'],13)
        source=json.loads(Path('research/phase9_text_audit.json').read_text())
        self.assertTrue(source['categories_overlap'])

    def test_audit_rejects_fabricated_quote(self):
        r=dict(correct=False,response='#### 3',extraction_failed=False,gold_numeric='2',predicted_numeric='3',stratum='0-2')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'audit.json';p.write_text(json.dumps(dict(annotation_type='fixture',entries=[dict(question_id=1,tags=['wrong_value'],evidence_quotes=['NOT PRESENT'],assessment='fixture')])))
            with self.assertRaises(ValueError):audit_errors({1:r},p)

    def test_parser_sensitivity_is_separate_from_official_scores(self):
        rows=csv_rows('reports/phase9/parser_sensitivity.csv')
        self.assertEqual(sum(r['original_official_correct']=='True' for r in rows),28)
        self.assertEqual(sum(r['concise_official_correct']=='True' for r in rows),22)
        self.assertEqual(sum(r['original_diagnostic_correct']=='True' for r in rows),33)
        self.assertEqual(sum(r['concise_diagnostic_correct']=='True' for r in rows),37)
        report=Path('reports/phase9/PHASE9_REPORT.md').read_text()
        for phrase in ('posthoc','not an independent human rating','quality preservation or degradation remains unverified','unequal timing scopes'):
            self.assertIn(phrase,report)

    def test_source_provenance_and_no_fabricated_human_review(self):
        p=json.loads(Path('reports/phase9/provenance.json').read_text())
        for name,sha in p['analysis_source_sha256'].items():self.assertEqual(digest(name),sha)
        self.assertEqual(p['annotation_sha256'],digest('research/phase9_text_audit.json'))
        self.assertEqual(p['review']['n_pairs'],60);self.assertEqual(p['review']['rows_with_any_rating'],0)
        self.assertFalse(p['review']['reasoning_quality_verified'])
        self.assertTrue(p['review']['unblinding_not_public'])

    def test_saved_table_corruption_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/'a.csv';b=Path(temp)/'b.csv';a.write_text('count\n22\n');b.write_text('count\n23\n')
            with self.assertRaises(ValueError):compare_tables(a,b)

    def test_old_public_results_are_not_allowed_finalization_targets(self):
        with self.assertRaises(ValueError):finalize('unused','unused','reports/phase8','results/phase9/private','unused','unused')
