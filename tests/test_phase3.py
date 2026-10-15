"""CPU fixtures only; no model weights or inference from a pretrained model."""
import unittest
import json
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from dllm_unmasking.phase3_schedule import schedule
from dllm_unmasking.phase3_sampler import generate
from experiments.phase3.run import validate
from dllm_unmasking.phase2_data import recover_journal,append_event


class Toy:
    def __init__(self):self.inputs=[]
    def __call__(self,x,use_cache=False):
        self.inputs.append(x.clone())
        count=int((x[:,2:]!=9).sum())
        logits=torch.zeros((*x.shape,10));logits[...,1+count%8]=3;logits[...,9]=100
        return SimpleNamespace(logits=logits)


class Tests(unittest.TestCase):
    def test_coverage_spacing_determinism(self):
        batches=schedule();self.assertEqual(batches,schedule());self.assertEqual(len(batches),32)
        self.assertEqual(batches[0],[0,2,4,6,8,10,12,14])
        self.assertEqual(batches[-1],[241,243,245,247,249,251,253,255])
        self.assertEqual(sorted(p for b in batches for p in b),list(range(256)))
        for b in batches:
            self.assertEqual(len(set(b)),8);self.assertTrue(all(y-x==2 for x,y in zip(b,b[1:])))

    def test_single_baseline_and_simultaneous_commit(self):
        toy=Toy();prompt=torch.tensor([[3,7]])
        ids,gpu,wall,trace=generate(toy,prompt,mask_id=9)
        self.assertEqual(len(toy.inputs),32);self.assertEqual(ids,[1]*256)
        self.assertEqual([t['positions'] for t in trace],schedule())
        self.assertEqual(sum(t['forward_passes'] for t in trace),32);self.assertIsNone(gpu)
        for i,x in enumerate(toy.inputs):
            self.assertTrue(torch.equal(x[:,:2],prompt));self.assertEqual(int((x[:,2:]!=9).sum()),8*i)
            known=[p for b in schedule()[:i] for p in b]
            self.assertTrue(all(x[0,2+p]==1 for p in known))
        self.assertGreater(wall,0)

    def test_exact_phase2_selection(self):
        import yaml
        cfg=yaml.safe_load(Path('configs/phase3.yaml').read_text())
        source=Path(os.environ.get('PHASE2_RESULTS',cfg['phase2_results'])).expanduser()
        selection,_,_=validate(cfg,source)
        self.assertEqual(len(selection['questions']),60)
        for s in ('0-2','3-4','5+'):
            self.assertEqual(sum(q['stratum']==s for q in selection['questions']),20)
        cfg=dict(cfg,steps=64)
        with self.assertRaises(ValueError):validate(cfg,source)

    def test_phase3_journal_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl';header=dict(question_id=1,strategy='ltr_spaced',parallelism=8)
            allowed={(1,'ltr_spaced',8)}
            with path.open('a') as f:
                append_event(f,dict(event='started',**header));f.write('{"partial"')
            records,attempts=recover_journal(path,allowed)
            self.assertFalse(records);self.assertEqual(attempts[(1,'ltr_spaced',8)],1)
            with path.open('a') as f:
                append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=header,**header))
            records,_=recover_journal(path,allowed);self.assertEqual(len(records),1)
            with path.open('a') as f:append_event(f,dict(event='completed',result=header,**header))
            with self.assertRaises(ValueError):recover_journal(path,allowed)


if __name__=='__main__':unittest.main()
