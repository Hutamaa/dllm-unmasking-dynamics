"""CPU-only tests with deterministic fixture logits; no pretrained model use."""
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import torch

from dllm_unmasking.phase2_samplers import (STRATEGIES,generate,predictions,influence_candidates,
                             expected_passes,validate_schedule)
from dllm_unmasking.phase2_data import choose_questions,recover_journal,append_event,key,parquet_rows
from experiments.phase2.run import read_config,cached_model


class Toy:
    def __init__(self):self.inputs=[]
    def __call__(self,x,use_cache=False):
        self.inputs.append(x.clone())
        logits=torch.zeros((*x.shape,8))
        # Baseline top-1 changes with number of reveals. Detects greedy predictions within a step.
        count=int((x[:,1:]!=7).sum())
        logits[...,2 if count%2==0 else 3]=2
        logits[...,7]=100
        return SimpleNamespace(logits=logits)


class InfluenceToy:
    def __init__(self):self.inputs=[]
    def __call__(self,x,use_cache=False):
        self.inputs.append(x.clone());logits=torch.zeros((*x.shape,8))
        logits[...,2]=1;logits[...,7]=100
        # Revealing position 1 reduces uncertainty elsewhere. Revealing position 0
        # changes only its own entropy and must score exactly zero.
        if x[0,1]!=7:logits[0,1,2]=10
        if x[0,2]!=7:logits[0,1,2]=10;logits[0,3,2]=10
        return SimpleNamespace(logits=logits)


class ParallelTests(unittest.TestCase):
    def test_all_schedules_and_no_global_rng_changes(self):
        for steps in (64,32):
            for strategy in STRATEGIES:
                model=Toy();state=random.getstate();ts=torch.random.get_rng_state().clone()
                ids,gpu,wall,trace=generate(model,torch.tensor([[1]]),strategy,steps=steps,mask_id=7,seed=17)
                self.assertEqual(len(trace),steps)
                self.assertTrue(all(len(t['positions'])==256//steps for t in trace))
                positions=[p for t in trace for p in t['positions']]
                self.assertEqual(sorted(positions),list(range(256)))
                self.assertEqual(sum(t['forward_passes'] for t in trace),len(model.inputs))
                self.assertEqual(len(model.inputs),expected_passes(steps,strategy))
                self.assertEqual(random.getstate(),state);self.assertTrue(torch.equal(ts,torch.random.get_rng_state()))
                self.assertIsNone(gpu);self.assertGreater(wall,0)
                self.assertNotIn(7,ids)
                if strategy in ('confidence','left_to_right','batched_influence'):
                    self.assertEqual(positions,list(range(256)))
                if strategy=='right_to_left':self.assertEqual(positions,list(range(255,-1,-1)))
                # Even 4/8-sized commits keep baseline prediction 2 across all steps;
                # a sequential greedy implementation would alternate predicted tokens.
                self.assertTrue(all(t==2 for t in ids))
                call=0
                for step in trace:
                    baseline=model.inputs[call]
                    self.assertEqual(int((baseline[:,1:]!=7).sum()),step['step']*(256//steps))
                    if strategy=='batched_influence':
                        for probe in model.inputs[call+1:call+step['forward_passes']]:
                            self.assertEqual(int((probe!=baseline).sum()),1)
                            self.assertEqual(int((probe[:,1:]!=7).sum()),step['step']*(256//steps)+1)
                    call+=step['forward_passes']

    def test_counterfactual_entropy_excludes_self_and_restores(self):
        model=InfluenceToy();x=torch.tensor([[1,7,7,7]])
        before=x.clone();revealed=torch.zeros(3,dtype=torch.bool)
        tokens,confidence,entropy=predictions(model,x,1,7,True)
        selected,scores,pool,passes=influence_candidates(model,x,1,revealed,tokens,confidence,entropy,7,2)
        self.assertTrue(torch.equal(x,before));self.assertEqual(passes,3)
        self.assertEqual(selected.tolist(),[1,0])
        mapping=dict(zip(pool['candidate_positions'].tolist(),pool['candidate_scores'].tolist()))
        self.assertEqual(mapping[0],0);self.assertGreater(mapping[1],0);self.assertEqual(mapping[2],0)
        for probe,position in zip(model.inputs[1:],[0,1,2]):
            expected=before.clone();expected[0,position+1]=tokens[position]
            self.assertTrue(torch.equal(probe,expected))
        probe=x.clone();probe[0,2]=tokens[1]
        _,_,after=predictions(model,probe,1,7,True)
        self.assertAlmostEqual(mapping[1],float((entropy[[0,2]]-after[[0,2]]).sum()),places=6)

    def test_prefilter_and_revealed_exclusion(self):
        x=torch.full((1,21),7,dtype=torch.long);x[0,0]=1
        revealed=torch.zeros(20,dtype=torch.bool);revealed[19]=True;x[0,20]=2
        model=Toy();tokens,confidence,entropy=predictions(model,x,1,7,True)
        confidence=torch.arange(20,dtype=torch.float)
        selected,_,pool,count=influence_candidates(model,x,1,revealed,tokens,confidence,entropy,7,8)
        self.assertEqual(count,16);self.assertEqual(pool['candidate_positions'].tolist(),list(range(3,19)))
        self.assertNotIn(19,selected.tolist());self.assertEqual(len(set(selected.tolist())),8)

    def test_invalid_sequential_setting_and_counts(self):
        for steps in (256,128,0):
            with self.assertRaises(ValueError):validate_schedule(256,256,steps)
        self.assertEqual(expected_passes(64,'batched_influence'),1064)
        self.assertEqual(expected_passes(32,'batched_influence'),536)

    def test_random_repeatability(self):
        t1=generate(Toy(),torch.tensor([[1]]),'random',steps=32,mask_id=7,seed=21)[3]
        t2=generate(Toy(),torch.tensor([[1]]),'random',steps=32,mask_id=7,seed=21)[3]
        self.assertEqual(t1,t2)


class SelectionAndJournalTests(unittest.TestCase):
    def test_pinned_config_and_missing_cache_failure(self):
        config=read_config('configs/phase2.yaml')
        self.assertEqual(config['settings'],[dict(parallelism=4,steps=64),dict(parallelism=8,steps=32)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong.parquet';p.write_bytes(b'not-the-pinned-dataset')
            with self.assertRaisesRegex(ValueError,'pinned GSM8K'):
                parquet_rows(p,config['dataset_test_parquet_sha256'])
            def snapshot(*args,**kwargs):
                self.assertTrue(kwargs['local_files_only'])
                self.assertEqual(kwargs['revision'],config['model_revision'])
                return tmp
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=snapshot)}):
                with self.assertRaisesRegex(RuntimeError,'Downloads are disabled'):
                    cached_model()

    def test_stratification(self):
        rows=[dict(question=str(i),answer=('<<1+1=2>>'*([1,3,5][i%3]))+'\n#### 2') for i in range(300)]
        excluded=list(range(150));selected=choose_questions(rows,excluded,20261014)
        self.assertEqual(selected,choose_questions(rows,excluded,20261014))
        self.assertEqual(len({q['question_id'] for q in selected}),60)
        self.assertFalse(set(excluded)&{q['question_id'] for q in selected})
        for s in ('0-2','3-4','5+'):self.assertEqual(sum(q['stratum']==s for q in selected),20)
        with self.assertRaises(ValueError):choose_questions(rows[:160],excluded,42)

    def test_write_ahead_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl';k=(10,'random',4);allowed={k}
            header=dict(question_id=10,strategy='random',parallelism=4)
            with path.open('a') as f:append_event(f,dict(event='started',**header))
            records,attempts=recover_journal(path,allowed)
            self.assertFalse(records);self.assertEqual(attempts[k],1)
            with path.open('a') as f:
                append_event(f,dict(event='started',**header))
                append_event(f,dict(event='completed',result=header,**header))
                f.write('{"partial"')
            records,attempts=recover_journal(path,allowed)
            self.assertEqual(records[k],header);self.assertEqual(attempts[k],2)
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            self.assertEqual(recover_journal(path,allowed)[0],records)
            with path.open('a') as f:append_event(f,dict(event='completed',result=header,**header))
            with self.assertRaises(ValueError):recover_journal(path,allowed)

    def test_actual_selection(self):
        selection=json.loads(Path('outputs/phase2/selection.json').read_text())
        self.assertEqual(selection['selection_seed'],20261014)
        selected=selection['questions'];excluded=selection['excluded_phase1_ids']
        self.assertEqual(len(selected),60);self.assertEqual(len(set(excluded)),150)
        self.assertFalse(set(excluded)&{q['question_id'] for q in selected})
        for s in ('0-2','3-4','5+'):self.assertEqual(sum(q['stratum']==s for q in selected),20)


if __name__=='__main__':unittest.main()
