"""CPU publication contracts; public tables and temporary synthetic artifacts only."""
import json
from pathlib import Path
import tempfile
import unittest

from dllm_unmasking.finalize_phase8 import import_run, inventory, paired, verify_events, verify_public


class FinalPhase8Tests(unittest.TestCase):
    def test_import_is_idempotent_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();dest=Path(tmp)/'imported'
            for name in ('manifest.json','journal.jsonl','environment.json','results.csv','summary.csv'):
                (source/name).write_text('synthetic fixture\n')
            (source/'.run.lock').touch()
            before=inventory(source)
            receipt=import_run(source,dest)
            self.assertNotIn('.run.lock',receipt)
            self.assertEqual(receipt,import_run(source,dest))
            self.assertEqual(before,inventory(source))
            copied=inventory(dest)
            (source/'journal.jsonl').write_text('different fixture\n')
            with self.assertRaises(ValueError):import_run(source,dest)
            self.assertEqual(copied,inventory(dest))
            with self.assertRaises(ValueError):import_run(source,source/'nested')

    def test_missing_required_import_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir()
            with self.assertRaises(ValueError):import_run(source,Path(tmp)/'dest')
            self.assertFalse((Path(tmp)/'dest').exists())

    def test_event_audit_rejects_duplicates_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl'
            events=[dict(event=kind,question_id=q) for q in range(60) for kind in ('started','completed')]
            path.write_text(''.join(json.dumps(e)+'\n' for e in events));verify_events(path)
            path.write_text(''.join(json.dumps(e)+'\n' for e in events[:-1]))
            with self.assertRaises(ValueError):verify_events(path)
            path.write_text(''.join(json.dumps(e)+'\n' for e in events+[events[-1]]))
            with self.assertRaises(ValueError):verify_events(path)

    def test_paired_stream_reproducibility_and_known_discordances(self):
        strata={q:('0-2','3-4','5+')[q//20] for q in range(60)}
        a={q:dict(correct=False) for q in strata};b={q:dict(correct=False) for q in strata}
        a[0]['correct']=True;b[1]['correct']=True
        first=paired(a,b,strata,'A','B')
        self.assertEqual(first,paired(a,b,strata,'A','B'))
        self.assertEqual(first['b_recovers'],1);self.assertEqual(first['b_regresses'],1)
        self.assertEqual(first['mcnemar_exact_p'],1.);self.assertEqual(first['b_minus_a_pp'],0.)
        b[2]['correct']=True
        self.assertNotEqual(first,paired(a,b,strata,'A','B'))
        with self.assertRaises(ValueError):paired(a,{0:b[0]},strata,'A','B')

    def test_published_statistics_recompute_without_raw_journals(self):
        self.assertEqual(verify_public('reports/phase8'),(360,9))

    def test_public_corruption_is_detected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'tables';shutil.copytree('reports/phase8',target)
            path=target/'per_question.csv'
            text=path.read_text();text=text.replace(',True,',',False,',1);path.write_text(text)
            with self.assertRaises(ValueError):verify_public(target)

    def test_canonical_report_labels_and_provenance(self):
        source=Path('reports/phase8');text=(source/'PHASE8_REPORT.md').read_text()
        self.assertIn('matched in nominal generated positions',text)
        self.assertIn('not fully validated',text)
        self.assertIn('Confirmed findings versus mechanistic hypotheses',text)
        self.assertIn('new versioned offline analysis',text)
        provenance=json.loads((source/'provenance.json').read_text())
        self.assertEqual(provenance['seed'],42);self.assertEqual(provenance['bootstrap_repetitions'],10000)
        self.assertEqual(set(provenance['imports']),{'256','512'})
        self.assertEqual(set(provenance['inference_source_verification']),{'256','512'})
