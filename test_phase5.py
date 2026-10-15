"""CPU-only fixtures. No GPU, baseline inference, or pretrained model loading."""
import copy
import csv
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import torch
from phase5_sampler import generate,STRATEGIES
from phase2_samplers import predictions
from phase2_data import append_event,recover_journal
from run_phase5 import validate,audit_result,summarize


class Toy:
    """Confidence rank changes each step; revised old positions have overwhelming confidence."""
    def __init__(self):self.inputs=[]
    def __call__(self,x,use_cache=False):
        step=len(self.inputs);self.inputs.append(x.clone())
        logits=torch.zeros((*x.shape,10));logits[...,9]=100
        # Rotating position preference proves new positions are dynamic, not a fixed LTR schedule.
        margins=1+((torch.arange(256)+step*7)%256)/128
        known=[]
        if step:
            ever_known=torch.zeros(256,dtype=torch.bool)
            for previous in self.inputs:ever_known|=previous[0,2:]!=9
            known=torch.nonzero(ever_known & (x[0,2:]==9)).flatten().tolist()
            margins[ever_known]=20
        logits[0,2:,1+step%3]=margins
        return SimpleNamespace(logits=logits)


class Uniform:
    def __init__(self):self.inputs=[]
    def __call__(self,x,use_cache=False):
        step=len(self.inputs);self.inputs.append(x.clone())
        logits=torch.zeros((*x.shape,10));logits[...,1+step%3]=3;logits[...,9]=100
        return SimpleNamespace(logits=logits)


class Tests(unittest.TestCase):
    def test_common_state_dynamic_rank_and_revision_exclusion(self):
        for strategy in STRATEGIES:
            toy=Toy();prompt=torch.tensor([[4,5]])
            random_state=random.getstate();torch_state=torch.random.get_rng_state().clone()
            ids,_,_,trace=generate(toy,prompt,strategy,mask_id=9,revision_seed=777)
            self.assertEqual(len(toy.inputs),32);self.assertEqual(len(trace),32)
            self.assertEqual(random.getstate(),random_state);self.assertTrue(torch.equal(torch_state,torch.random.get_rng_state()))
            seen=set();revised=set();state=[9]*256;confidence={};rng=random.Random(777)
            for step,t in enumerate(trace):
                pos=t['positions'];rev=t['revisions'];chosen=[v['position'] for v in rev]
                self.assertTrue(torch.equal(toy.inputs[step][:,:2],prompt))
                eligible=sorted(seen-revised);budget=0 if not step else 2
                expected_rev=rng.sample(eligible,budget) if strategy=='confidence_random_remask' else sorted(eligible,key=lambda p:(confidence[p],p))[:budget]
                self.assertEqual(chosen,expected_rev)
                before=state.copy()
                for p in chosen:before[p]=9
                self.assertEqual(toy.inputs[step][0,2:].tolist(),before)
                # Independently reproduce full baseline score ranking at this already captured input.
                class Replay:
                    def __call__(self,x,use_cache=False):
                        margins=1+((torch.arange(256)+step*7)%256)/128
                        if step:
                            ever=torch.zeros(256,dtype=torch.bool)
                            for previous in toy.inputs[:step]:ever|=previous[0,2:]!=9
                            margins[ever]=20
                        logits=torch.zeros((*x.shape,10));logits[...,9]=100;logits[0,2:,1+step%3]=margins
                        return SimpleNamespace(logits=logits)
                _,scores,_=predictions(Replay(),toy.inputs[step],2,9)
                expected_new=sorted(set(range(256))-seen,key=lambda p:(-float(scores[p]),p))[:8]
                self.assertEqual(pos,expected_new);self.assertFalse(set(pos)&seen);self.assertFalse(set(pos)&set(chosen))
                self.assertEqual(len(pos),8);self.assertEqual(t['forward_passes'],1)
                for v in rev:
                    p=v['position'];self.assertEqual(v['original_token'],state[p]);self.assertEqual(v['original_confidence'],confidence[p])
                    self.assertEqual(v['replacement_token'],1+step%3)
                    self.assertEqual(v['changed'],v['original_token']!=v['replacement_token'])
                    state[p]=v['replacement_token']
                for p,token,c in zip(pos,t['token_ids'],t['confidences']):state[p]=token;confidence[p]=c
                seen.update(pos);revised.update(chosen)
                self.assertEqual(t['remaining_masks_before']-t['remaining_masks_after'],8)
            self.assertEqual(len(seen),256);self.assertEqual(len(revised),62);self.assertEqual(ids,state)
            self.assertNotEqual(trace[0]['positions'],list(range(8)))

    def test_adversarial_remasked_positions_rank_highest_but_not_new(self):
        class OldHigh:
            def __init__(self):self.inputs=[]
            def __call__(self,x,use_cache=False):
                self.inputs.append(x.clone());logits=torch.zeros((*x.shape,10));logits[...,1]=2;logits[...,9]=100
                # Uniform step zero commits 0..7. In subsequent steps those positions
                # get max confidence, even if MASK. They must never be counted as new again.
                if len(self.inputs)>1:logits[0,2:10,1]=30
                return SimpleNamespace(logits=logits)
        for s in STRATEGIES:
            toy=OldHigh();ids,_,_,trace=generate(toy,torch.tensor([[4,5]]),s,mask_id=9,revision_seed=7)
            self.assertEqual(trace[0]['positions'],list(range(8)))
            self.assertEqual(trace[1]['positions'],list(range(8,16)))
            self.assertEqual(len([p for t in trace for p in t['positions']]),256)
            self.assertEqual(len(set(p for t in trace for p in t['positions'])),256)
            rev=[r['position'] for r in trace[1]['revisions']]
            self.assertTrue(all(toy.inputs[1][0,2+p]==9 for p in rev))

    def test_uniform_confidence_ties_and_repeatability(self):
        for s in STRATEGIES:
            a=generate(Uniform(),torch.tensor([[4,5]]),s,mask_id=9,revision_seed=4)
            b=generate(Uniform(),torch.tensor([[4,5]]),s,mask_id=9,revision_seed=4)
            self.assertEqual(a[0],b[0]);self.assertEqual(a[3],b[3])
            self.assertEqual([p for t in a[3] for p in t['positions']],list(range(256)))

    def fixture(self):
        cfg=dict(mask_id=9,inference_seed=42,revision_seed=314159)
        q=dict(question_id=1,question='Toy fixture',gold_answer='#### 2',stratum='0-2',calculation_count=0)
        ids,_,_,trace=generate(Uniform(),torch.tensor([[4,5]]),STRATEGIES[0],mask_id=9,revision_seed=314160)
        r=dict(**q,strategy=STRATEGIES[0],parallelism=8,steps=32,forward_passes=32,position_seed=43,revision_seed=314160,
            generated_token_ids=ids,trajectory=trace,response='#### 2',raw_response='#### 2',eos_token_id=8,
            predicted_numeric='2',gold_numeric='2',correct=True,extraction_failed=False,extraction_method='marker',gpu_seconds=.1,wall_seconds=.1)
        return cfg,q,r

    def test_resume_record_replay_and_corruption_rejection(self):
        cfg,q,r=self.fixture();audit_result(r,q,cfg)
        for mutation in ('duplicate','original','final','seed'):
            bad=copy.deepcopy(r)
            if mutation=='duplicate':bad['trajectory'][1]['positions'][0]=bad['trajectory'][0]['positions'][0]
            elif mutation=='original':bad['trajectory'][1]['revisions'][0]['original_token']=5
            elif mutation=='final':bad['generated_token_ids'][0]=7
            else:bad['revision_seed']=0
            with self.assertRaises(ValueError):audit_result(bad,q,cfg)

    def test_journal_and_outputs(self):
        cfg,q,r=self.fixture();header=dict(question_id=1,strategy=STRATEGIES[0],parallelism=8);allowed={(1,STRATEGIES[0],8)}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);path=out/'journal.jsonl'
            with path.open('a') as f:append_event(f,dict(event='started',**header));f.write('partial')
            records,_=recover_journal(path,allowed);self.assertFalse(records)
            with path.open('a') as f:append_event(f,dict(event='completed',result=r,**header))
            records,_=recover_journal(path,allowed);audit_result(records[(1,STRATEGIES[0],8)],q,cfg)
            summarize(out,records,{1:dict(correct=False,predicted_numeric='3',response='Toy baseline')},[q],9)
            for name in ('results.csv','summary.csv','revisions.csv','revision_confidence.csv','response_diagnostics.csv','paired_phase2.csv','matched_cases.csv','matched_report.md'):
                self.assertTrue((out/name).exists(),name)
            with (out/'paired_phase2.csv').open() as file:
                paired_rows=list(csv.DictReader(file))
            self.assertIn('phase2_confidence_p8_accuracy',paired_rows[0]);self.assertNotIn('p4_mean',paired_rows[0])
            with path.open('a') as f:append_event(f,dict(event='completed',result=r,**header))
            with self.assertRaises(ValueError):recover_journal(path,allowed)

    def test_actual_baseline_selection(self):
        import yaml
        cfg=yaml.safe_load(Path('phase5_config.yaml').read_text());source=Path(os.environ.get('PHASE2_RESULTS',cfg['phase2_results'])).expanduser()
        selection,baseline,_=validate(cfg,source);self.assertEqual(len(selection['questions']),60)
        self.assertEqual(sum(r['correct'] for r in baseline.values()),26)
        with self.assertRaises(ValueError):validate(dict(cfg,revision_budget=3),source)

    def test_nonfinite_predictions_stop(self):
        class Bad:
            def __call__(self,x,use_cache=False):return SimpleNamespace(logits=torch.full((*x.shape,10),float('nan')))
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            generate(Bad(),torch.tensor([[4,5]]),STRATEGIES[0],mask_id=9)


if __name__=='__main__':unittest.main()
