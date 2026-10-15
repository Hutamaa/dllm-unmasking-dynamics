"""CPU deterministic fixtures; no pretrained model loading or GPU inference."""
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import torch
from phase4_sampler import generate,STRATEGIES
from phase3_schedule import schedule
from phase2_data import recover_journal,append_event
from run_phase4 import validate,summarize,workload


class Toy:
    def __init__(self):self.inputs=[]
    def __call__(self,x,use_cache=False):
        step=len(self.inputs);self.inputs.append(x.clone())
        logits=torch.zeros((*x.shape,10));logits[...,1+step%3]=3;logits[...,9]=100
        return SimpleNamespace(logits=logits)


class Tests(unittest.TestCase):
    def test_remasking_and_single_common_forward(self):
        for strategy in STRATEGIES:
            toy=Toy();prompt=torch.tensor([[4,5]]);rng_before=random.getstate();torch_before=torch.random.get_rng_state().clone()
            ids,seconds,wall,trace=generate(toy,prompt,strategy,mask_id=9,revision_seed=777)
            self.assertEqual(len(toy.inputs),32);self.assertEqual([t['positions'] for t in trace],schedule())
            self.assertEqual(sum(t['forward_passes'] for t in trace),32);self.assertNotIn(9,ids)
            self.assertEqual(random.getstate(),rng_before);self.assertTrue(torch.equal(torch_before,torch.random.get_rng_state()))
            seen=set();revised=set();state=[9]*256;private=random.Random(777)
            for step,t in enumerate(trace):
                self.assertTrue(torch.equal(toy.inputs[step][:,:2],prompt))
                rr=t['revisions'];picked=[r['position'] for r in rr]
                eligible=sorted(seen-revised);budget=0 if step==0 else 2
                self.assertEqual(t['revision_budget'],budget);self.assertEqual(len(rr),budget)
                self.assertEqual(t['eligible_revision_count'],len(eligible))
                self.assertTrue(set(picked).issubset(eligible));self.assertFalse(set(picked)&set(t['positions']))
                if strategy=='random_remask':self.assertEqual(picked,private.sample(eligible,budget))
                else:self.assertEqual(picked,eligible[:budget]) # uniform confidence: position tie-break
                expected=state.copy()
                for r in rr:
                    self.assertEqual(r['original_token'],state[r['position']]);expected[r['position']]=9
                    self.assertEqual(r['replacement_token'],1+step%3)
                    self.assertEqual(r['changed'],r['original_token']!=r['replacement_token'])
                    self.assertEqual(r['decoding_half'],'first' if step<16 else 'second')
                    self.assertGreater(r['original_confidence'],0);self.assertGreater(r['replacement_confidence'],0)
                self.assertEqual(toy.inputs[step][0,2:].tolist(),expected)
                for p in t['positions']+picked:state[p]=1+step%3
                seen.update(t['positions']);revised.update(picked)
                self.assertEqual(t['remaining_masks_before']-t['remaining_masks_after'],8)
                self.assertEqual(t['remaining_masks_after'],256-8*(step+1))
            self.assertEqual(state,ids);self.assertEqual(len(revised),62)
            self.assertEqual(sum(bool(r['changed']) for t in trace for r in t['revisions'])>0,True)
            self.assertIsNone(seconds);self.assertGreater(wall,0)

    def test_low_confidence_uses_original_commit_probability(self):
        class Graded:
            def __init__(self):self.calls=0
            def __call__(self,x,use_cache=False):
                step=self.calls;self.calls+=1
                logits=torch.zeros((*x.shape,10));logits[...,9]=100
                # Reverse confidence ranking after first commit; ranking must still
                # use stored first-step probabilities when selecting step-1 revisions.
                margins=2+(-1 if step==0 else 1)*torch.arange(256)/512
                logits[0,2:,1+step%3]=margins
                return SimpleNamespace(logits=logits)
        _,_,_,trace=generate(Graded(),torch.tensor([[4,5]]),'low_confidence_remask',mask_id=9)
        rr=trace[1]['revisions'];self.assertEqual([r['position'] for r in rr],[14,12])
        stored=dict(zip(trace[0]['positions'],trace[0]['confidences']))
        self.assertEqual([r['original_confidence'] for r in rr],[stored[14],stored[12]])
        self.assertTrue(all(r['changed'] for r in rr))

    def test_repeatability(self):
        for s in STRATEGIES:
            a=generate(Toy(),torch.tensor([[4,5]]),s,mask_id=9,revision_seed=5)
            b=generate(Toy(),torch.tensor([[4,5]]),s,mask_id=9,revision_seed=5)
            self.assertEqual(a[0],b[0]);self.assertEqual(a[3],b[3])

    def test_nonfinite_confidence_stops(self):
        class Bad:
            def __call__(self,x,use_cache=False):return SimpleNamespace(logits=torch.full((*x.shape,10),float('nan')))
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            generate(Bad(),torch.tensor([[4,5]]),'random_remask',mask_id=9)

    def test_actual_baseline_config(self):
        import yaml
        cfg=yaml.safe_load(Path('phase4_config.yaml').read_text())
        source=Path(os.environ.get('PHASE3_RESULTS',cfg['phase3_results'])).expanduser()
        selection,baseline,_=validate(cfg,source);self.assertEqual(len(baseline),60)
        for s in ('0-2','3-4','5+'):self.assertEqual(sum(q['stratum']==s for q in selection['questions']),20)
        with self.assertRaises(ValueError):validate(dict(cfg,revision_budget=3),source)

    def test_pilot_extends_full_without_duplicate_keys(self):
        selection=json.loads(Path('outputs/phase2/selection.json').read_text())
        pilot,scope=workload(selection,20,False);full,full_scope=workload(selection,60,False)
        self.assertEqual(scope,full_scope);self.assertEqual(pilot,full[:20])
        self.assertEqual([sum(q['stratum']==s for q in pilot) for s in ('0-2','3-4','5+')],[7,7,6])
        completed={(q['question_id'],s,8) for q in pilot for s in STRATEGIES}
        remaining={(q['question_id'],s,8) for q in full for s in STRATEGIES}-completed
        self.assertEqual(len(completed),40);self.assertEqual(len(remaining),80)
        smoke,smoke_scope=workload(selection,2,True);self.assertEqual(smoke,smoke_scope)
        self.assertEqual(len(smoke),2)

    def test_journal_resume_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl';header=dict(question_id=1,strategy='random_remask',parallelism=8)
            allowed={(1,'random_remask',8)}
            with path.open('a') as f:append_event(f,dict(event='started',**header));f.write('partial')
            records,attempts=recover_journal(path,allowed);self.assertFalse(records)
            with path.open('a') as f:append_event(f,dict(event='completed',result=header,**header))
            records,attempts=recover_journal(path,allowed);self.assertEqual(len(records),1)
            with path.open('a') as f:append_event(f,dict(event='completed',result=header,**header))
            with self.assertRaises(ValueError):recover_journal(path,allowed)

    def test_summary_schema_with_fixture_only(self):
        ids,_,_,trace=generate(Toy(),torch.tensor([[4,5]]),'random_remask',mask_id=9)
        r=dict(question_id=1,question='Toy fixture',gold_answer='#### 2',calculation_count=0,stratum='0-2',strategy='random_remask',
            parallelism=8,steps=32,correct=True,predicted_numeric='2',gold_numeric='2',extraction_method='marker',extraction_failed=False,
            forward_passes=32,gpu_seconds=.1,wall_seconds=.1,revision_seed=42,response='#### 2',raw_response='#### 2',
            generated_token_ids=ids,eos_token_id=8,trajectory=trace)
        baseline={1:dict(correct=False,predicted_numeric='3',response='Toy fixture baseline')}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);summarize(out,{(1,'random_remask',8):r},baseline,[r],9)
            for name in ('summary.csv','revisions.csv','paired_phase3.csv','revision_confidence.csv','matched_report.md'):
                self.assertTrue((out/name).exists())
            self.assertIn('improved',(out/'matched_report.md').read_text())


if __name__=='__main__':unittest.main()
