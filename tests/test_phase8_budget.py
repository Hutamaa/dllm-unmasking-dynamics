"""Token-budget isolation contracts; synthetic CPU data only."""
import json
from pathlib import Path
import tempfile
import unittest
from dllm_unmasking.ar_baseline import load_config, validate_budget, generate_one, read_journal, validate_result, digest
from dllm_unmasking.ar_compare import mark_dominance
from dllm_unmasking.phase2_data import append_event
from tests.test_phase8 import Toy, Tokenizer, fixture


class BudgetTests(unittest.TestCase):
    def test_materialized_config_changes_only_budget_and_outputs(self):
        old=load_config();new=load_config('configs/phase8_qwen_512.yaml')
        self.assertEqual({k for k in old if old[k]!=new[k]}, {'max_new_tokens','output_dir'})
        self.assertEqual(validate_budget(new),512)
        self.assertEqual(load_config(model='llama')['max_new_tokens'],256)
        for output in (old['output_dir'], 'outputs/phase8', 'outputs/phase8/qwen/run/child'):
            with self.assertRaises(ValueError):validate_budget(dict(new,output_dir=output))
        with self.assertRaises(ValueError):validate_budget(dict(new,model=load_config(model='llama')['model']))

    def test_generation_uses_selected_budget_and_eos(self):
        import torch
        encoded={'input_ids':torch.tensor([[7,8]])}
        model=Toy(False);r=generate_one(model,Tokenizer(),encoded,[9],cap=512)
        self.assertEqual(model.kwargs['max_new_tokens'],512)
        self.assertEqual(r['generated_tokens'],512)
        self.assertEqual(r['model_forward_passes'],512)
        self.assertEqual(r['termination'],'length_limit')
        self.assertEqual(generate_one(Toy(),Tokenizer(),encoded,[9],cap=512)['generated_token_ids'],[2,9])

    def test_budget_journal_resume_isolation_and_legacy_compatibility(self):
        q,r=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl'
            header={k:r[k] for k in ('question_id','policy','run_hash')}
            with p.open('w') as f:
                append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=r,**header))
            before=digest(p)
            self.assertEqual(len(read_journal(p,[q],'fixture',require_gpu=False)),1)
            with self.assertRaises(ValueError):read_journal(p,[q],'fixture',repair=True,require_gpu=False,cap=512)
            self.assertEqual(digest(p),before)
            # Even early-EOS outputs must carry the correct budget identity.
            r['max_new_tokens']=512;header['max_new_tokens']=512
            with p.open('w') as f:
                append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=r,**header))
            self.assertEqual(len(read_journal(p,[q],'fixture',require_gpu=False,cap=512)),1)
            with self.assertRaises(ValueError):validate_result(r,q,'fixture',require_gpu=False)
            with self.assertRaises(ValueError):read_journal(p,[q],'other',require_gpu=False,cap=512)
            with p.open('a') as f:append_event(f,dict(event='completed',result=r,**header))
            with self.assertRaises(ValueError):read_journal(p,[q],'fixture',require_gpu=False,cap=512)

    def test_dominance_keeps_unequal_budgets_explicit(self):
        points=[dict(n=60,mean_gpu_seconds=1.,accuracy=.5,max_new_tokens=256),
                dict(n=60,mean_gpu_seconds=.5,accuracy=.9,max_new_tokens=512)]
        mark_dominance(points)
        self.assertFalse(points[0]['descriptive_cross_scope_dominated'])
        self.assertTrue(points[0]['descriptive_cross_budget_dominated'])
        self.assertIsNone(points[1]['descriptive_cross_scope_dominated'])

    def test_analysis_accepts_both_budgets_without_changing_inputs(self):
        from unittest.mock import patch
        from dllm_unmasking.ar_compare import compare
        from dllm_unmasking.ar_baseline import object_hash
        public=json.loads(Path('research/benchmark_selection.json').read_text())
        qs=[dict(q,question='Fixture',gold_answer='#### 2') for q in public['questions']]
        with tempfile.TemporaryDirectory(dir='results') as tmp:
            root=Path(tmp);paths=[]
            selection=root/'selection.json';selection.write_text(json.dumps(qs))
            for cap in (256,512):
                cfg=load_config('configs/phase8_qwen_512.yaml' if cap==512 else None)
                cfg['selection_file']=str(selection)
                p=root/str(cap);p.mkdir();paths.append(p)
                manifest=dict(config=cfg,model=cfg['model'],model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],
                    selection_sha256=digest(cfg['selection_file']),diffusion_sha256=digest(cfg['diffusion_results']))
                h=object_hash(manifest);manifest['run_hash']=h;(p/'manifest.json').write_text(json.dumps(manifest))
                with (p/'journal.jsonl').open('w') as f:
                    for q in qs:
                        _,r=fixture();r.update(question_id=q['question_id'],stratum=q['stratum'],calculation_count=q['calculation_count'],run_hash=h,
                            model=cfg['model'],model_revision=cfg['model_revision'],policy='qwen_greedy',max_new_tokens=cap,
                            gpu_prefill_decode_seconds=.2,gpu_prefill_forward_seconds=.1,gpu_decode_forward_seconds=.05,tokens_per_second=10.,peak_allocated_bytes=100)
                        header={k:r[k] for k in ('question_id','policy','run_hash','model','model_revision','max_new_tokens')}
                        append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=r,**header))
            hashes={p/n:digest(p/n) for p in paths for n in ('manifest.json','journal.jsonl')}
            with patch('dllm_unmasking.ar_compare.inputs',return_value=qs):points=compare(root/'comparison',paths,load_config())
            ar=[p for p in points if p['family']=='ar']
            self.assertEqual([p['max_new_tokens'] for p in ar],[256,512])
            self.assertEqual([p['comparison_kind'] for p in ar],['matched_256','extended_budget_512'])
            self.assertTrue(all(digest(p)==h for p,h in hashes.items()))
