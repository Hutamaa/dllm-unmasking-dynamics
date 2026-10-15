"""CPU-only P1 equivalence/resume tests; no real model inference."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from phase2_samplers import predictions
from phase2_data import append_event
from run_phase6 import generate_p1,validate,config_hash,audit_result,read_results
from analyze_phase6 import analyze


class Toy:
    def __init__(self,uniform=False):self.inputs=[];self.uniform=uniform
    def __call__(self,x,use_cache=False):
        self.inputs.append(x.clone());logits=torch.zeros((*x.shape,10));logits[...,9]=100
        step=len(self.inputs)-1;chosen=1+step%3
        logits[0,2:,chosen]=3 if self.uniform else 1+((torch.arange(256)+step*7)%256)/128
        return SimpleNamespace(logits=logits)


def phase2_reference_p1(model,prompt,policy):
    # Mathematical extrapolation of Phase 2 selection, only in tests: its production guard is unchanged.
    x=torch.full((1,258),9,dtype=torch.long);x[:,:2]=prompt;seen=torch.zeros(256,dtype=torch.bool);trace=[]
    for step in range(256):
        tokens,c,_=predictions(model,x,2,9)
        p=step if policy=='left_to_right' else int(torch.argsort(c.masked_fill(seen,-torch.inf),descending=True,stable=True)[0])
        x[0,2+p]=tokens[p];seen[p]=True;trace.append((p,int(tokens[p]),float(c[p])))
    return x[0,2:].tolist(),trace


class Tests(unittest.TestCase):
    def test_exact_phase2_selection_equivalence(self):
        for policy in ('left_to_right','confidence'):
            for uniform in (False,True):
                toy=Toy(uniform);reference=Toy(uniform);prompt=torch.tensor([[4,5]])
                ids,gpu,wall,trace,timing=generate_p1(toy,prompt,policy,9,42)
                expected,expected_trace=phase2_reference_p1(reference,prompt,policy)
                self.assertEqual(ids,expected);self.assertEqual(len(toy.inputs),256)
                self.assertEqual([(t['positions'][0],t['token_ids'][0],t['confidences'][0]) for t in trace],expected_trace)
                self.assertEqual(sorted(t['positions'][0] for t in trace),list(range(256)))
                self.assertIsNone(gpu);self.assertIsNone(timing['model_forward_gpu_stream_seconds'])
                self.assertGreaterEqual(timing['sampler_nonforward_main_thread_cpu_seconds'],0)
                for step,(x,y) in enumerate(zip(toy.inputs,reference.inputs)):
                    self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(x[:,:2],prompt))
                    self.assertEqual(int((x[0,2:]!=9).sum()),step)

    def fixture(self,policy='left_to_right'):
        cfg=dict(mask_id=9,inference_seed=42);q=dict(question_id=1,question='Toy',gold_answer='#### 2',stratum='0-2',calculation_count=0)
        ids,gpu,wall,trace,timing=generate_p1(Toy(),torch.tensor([[4,5]]),policy,9,43)
        timing['model_forward_gpu_stream_seconds']=.08  # synthetic timing fixture, not an experimental observation
        r=dict(**q,strategy=policy,parallelism=1,steps=256,config_hash=config_hash(cfg),position_seed=43,
            forward_passes=256,tokens_initially_committed=256,tokens_before_first_eos=256,tokens_final_decoded_response=3,
            generated_token_ids=ids,trajectory=trace,eos_token_id=8,response='#### 2',raw_response='#### 2',
            predicted_numeric='2',gold_numeric='2',correct=True,extraction_failed=False,extraction_method='marker',
            gpu_seconds=.1,wall_seconds=wall,evaluation_wall_seconds=wall+.1,**timing)
        return cfg,q,r

    def test_resume_corruption_duplicate_and_config_validation(self):
        cfg,q,r=self.fixture();audit_result(r,q,cfg)
        for mutation in ('duplicate','config','id','final'):
            bad=copy.deepcopy(r)
            if mutation=='duplicate':bad['trajectory'][1]['positions']=[bad['trajectory'][0]['positions'][0]]
            elif mutation=='config':bad['config_hash']='wrong'
            elif mutation=='id':bad['question_id']=2
            else:bad['generated_token_ids'][0]=7
            with self.assertRaises(ValueError):audit_result(bad,q,cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);path=out/'journal.jsonl';head=dict(question_id=1,strategy='left_to_right',parallelism=1,config_hash=config_hash(cfg))
            with path.open('a') as f:append_event(f,dict(event='started',**head));f.write('partial')
            with self.assertRaises(ValueError):read_results(out,{'questions':[q]},cfg,repair=False)
            results,_=read_results(out,{'questions':[q]},cfg,repair=True);self.assertFalse(results)
            with path.open('a') as f:append_event(f,dict(event='completed',result=r,**head))
            results,_=read_results(out,{'questions':[q]},cfg);self.assertEqual(len(results),1)
            # Confidence can be added independently, while LTR's completion remains intact.
            cfg2,q2,r2=self.fixture('confidence')
            h2=dict(question_id=1,strategy='confidence',parallelism=1,config_hash=config_hash(cfg))
            with path.open('a') as f:
                append_event(f,dict(event='started',**h2));append_event(f,dict(event='completed',result=r2,**h2))
            results,_=read_results(out,{'questions':[q]},cfg);self.assertEqual(len(results),2)
            with path.open('a') as f:append_event(f,dict(event='completed',result=r,**head))
            with self.assertRaises(ValueError):read_results(out,{'questions':[q]},cfg)

    def test_actual_60_question_matching(self):
        import yaml
        cfg=yaml.safe_load(Path('phase6_config.yaml').read_text());source=Path(os.environ.get('PHASE2_RESULTS',cfg['phase2_results'])).expanduser()
        selection,lookup,_=validate(cfg,source);self.assertEqual(len(selection['questions']),60)
        self.assertEqual(sum(lookup[q['question_id'],'confidence',8]['correct'] for q in selection['questions']),26)
        for s in ('0-2','3-4','5+'):self.assertEqual(sum(q['stratum']==s for q in selection['questions']),20)
        with self.assertRaises(ValueError):validate(dict(cfg,steps=32),source)

    def test_preservation_of_earlier_sources_and_baseline_artifacts(self):
        paths=[Path(n) for n in ('samplers.py','run_experiment.py','config.yaml','evaluation.py',
            'analyze_phase15.py','phase2_samplers.py','phase2_data.py','run_phase2.py','phase2_config.yaml',
            'run_phase3.py','phase3_sampler.py','phase3_schedule.py','phase3_config.yaml',
            'phase4_sampler.py','run_phase4.py','phase4_config.yaml','phase5_sampler.py','run_phase5.py','phase5_config.yaml')]
        import yaml
        cfg=yaml.safe_load(Path('phase6_config.yaml').read_text())
        source=Path(os.environ.get('PHASE2_RESULTS',cfg['phase2_results'])).expanduser()
        paths += [p for p in source.iterdir() if p.is_file()]
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        generate_p1(Toy(),torch.tensor([[4,5]]),'confidence',9,42)
        validate(cfg,source)
        self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in before.items()))

    def test_analysis_labels_and_missing_latency(self):
        cfg,q,r=self.fixture();records={(1,'left_to_right',1):r}
        # Temporary fixture baseline rows only: no fabricated results are saved as research outputs.
        old={}
        for policy in ('left_to_right','confidence'):
            for p in (4,8):
                fixture=copy.deepcopy(r);fixture.update(strategy=policy,parallelism=p,steps=256//p,forward_passes=256//p,correct=False)
                for field in ('evaluation_wall_seconds','model_forward_gpu_stream_seconds','sampler_nonforward_main_thread_cpu_seconds',
                              'model_forward_host_enqueue_wall_seconds','tokens_initially_committed','tokens_before_first_eos','tokens_final_decoded_response'):
                    fixture.pop(field)
                old[(1,policy,p)]=fixture
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);analyze(out,records,old,{'questions':[q]},cfg)
            import csv
            rows=list(csv.DictReader((out/'accuracy_latency_table.csv').read_text().splitlines()))
            self.assertEqual(len(rows),5)
            self.assertTrue(all(x['observed_evaluation_wall_seconds']=='' for x in rows if x['source_phase']=='2'))
            self.assertFalse(json.loads((out/'completion.json').read_text())['complete'])
            self.assertTrue((out/'paired_comparisons.csv').exists())


if __name__=='__main__':unittest.main()
