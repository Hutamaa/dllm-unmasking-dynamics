"""CPU fixtures for the AR interface; never access a pretrained model or network."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from dllm_unmasking.ar_baseline import (MODEL, POLICY, inputs, digest, prompt_inputs, generate_one,
    response_record, validate_result, read_journal, cache_check, snapshot, complete_snapshot, check_access)
from dllm_unmasking.ar_analysis import summarize, FRONTIER
from dllm_unmasking.phase2_data import append_event


class Tokenizer:
    pad_token_id = None
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages; self.template_args = kwargs
        return 'NATIVE TEMPLATE: ' + messages[0]['content']
    def __call__(self, text, **kwargs):
        self.encode_args = kwargs
        return {'input_ids':torch.tensor([[7,8]]), 'attention_mask':torch.ones((1,2),dtype=torch.long)}
    def decode(self, ids, skip_special_tokens):
        return '#### 2' if ids and ids[0] == 2 else ''


class Toy(torch.nn.Module):
    def __init__(self, finish=True):
        super().__init__(); self.finish=finish; self.calls=0
    def forward(self, input_ids, **kwargs):
        self.calls+=1
        return torch.zeros(1)
    def generate(self, input_ids, **kwargs):
        self.kwargs=kwargs
        ids=[2,9] if self.finish else [2]*kwargs['max_new_tokens']
        for i in range(len(ids)):
            self(input_ids=input_ids if i == 0 else torch.tensor([[ids[i-1]]]))
        return torch.cat([input_ids,torch.tensor([ids])],dim=1)


def fixture():
    q=dict(question_id=1, question='Fixture only', gold_answer='#### 2', stratum='0-2', calculation_count=1)
    token=Tokenizer(); _,encoded=prompt_inputs(token,q['question'],' suffix')
    result=generate_one(Toy(),token,encoded,[9])
    result.update(question_id=1, policy=POLICY, run_hash='fixture', stratum='0-2', calculation_count=1,
                  inference_seed=42, gold_numeric='2',predicted_numeric='2',extraction_method='marker',
                  extraction_failed=False,correct=True,eos_token_ids=[9],prompt_tokens=2)
    return q,result


class Phase8(unittest.TestCase):
    def test_native_template_and_prompt_exclusion(self):
        tokenizer=Tokenizer();text,encoded=prompt_inputs(tokenizer,'Question',' suffix')
        self.assertEqual(tokenizer.messages,[dict(role='user',content='Question suffix')])
        self.assertEqual(tokenizer.template_args,dict(tokenize=False,add_generation_prompt=True))
        self.assertFalse(tokenizer.encode_args['add_special_tokens'])
        model=Toy();result=generate_one(model,tokenizer,encoded,[9])
        self.assertEqual(result['generated_token_ids'],[2,9])
        self.assertEqual(result['response'],'#### 2');self.assertEqual(result['response_tokens'],1)
        self.assertEqual(result['termination'],'eos');self.assertEqual(result['model_forward_passes'],2)
        self.assertFalse(model.kwargs['do_sample']);self.assertTrue(model.kwargs['use_cache'])
        self.assertEqual(model.kwargs['max_new_tokens'],256);self.assertEqual(model.kwargs['num_beams'],1)

    def test_length_stop_and_corrupt_termination(self):
        result=generate_one(Toy(False),Tokenizer(),{'input_ids':torch.tensor([[7,8]])},[9])
        self.assertEqual(result['generated_tokens'],256);self.assertEqual(result['termination'],'length_limit')
        for ids in ([2],[2,9,2],[]):
            with self.assertRaises(ValueError):response_record(ids,[9],Tokenizer(),256)
        self.assertEqual(response_record([2]*255+[9],[9],Tokenizer(),256)['termination'],'eos')

    def test_evaluator_and_timing_validation(self):
        q,r=fixture();validate_result(r,q,'fixture',False)
        for key,value in [('correct',False),('run_hash','bad'),('generated_tokens',3),('generation_wall_seconds',float('nan')),('inference_seed',0)]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(ValueError):validate_result(bad,q,'fixture',False)
        with self.assertRaises(ValueError):validate_result(r,q,'fixture',True)
        gpu=dict(r,gpu_prefill_decode_seconds=.2,gpu_prefill_forward_seconds=.1,gpu_decode_forward_seconds=.05,tokens_per_second=10.,peak_allocated_bytes=100)
        validate_result(gpu,q,'fixture',True)  # Synthetic schema fixture, not research measurements.

    def test_resume_partial_config_and_duplicate_rejection(self):
        q,r=fixture();header=dict(question_id=1,policy=POLICY,run_hash='fixture')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl'
            with path.open('a') as file:
                append_event(file,dict(event='started',**header));file.write('{partial')
            with self.assertRaises(ValueError):read_journal(path,[q],'fixture',require_gpu=False)
            self.assertFalse(read_journal(path,[q],'fixture',repair=True,require_gpu=False))
            with path.open('a') as file:append_event(file,dict(event='completed',result=r,**header))
            before=path.read_bytes()
            self.assertEqual(len(read_journal(path,[q],'fixture',require_gpu=False)),1)
            self.assertEqual(before,path.read_bytes())
            with self.assertRaises(ValueError):read_journal(path,[q],'different',require_gpu=False)
            with path.open('a') as file:append_event(file,dict(event='completed',result=r,**header))
            with self.assertRaises(ValueError):read_journal(path,[q],'fixture',require_gpu=False)

    def test_exact_60_selection_and_mismatch_rejection(self):
        cfg=yaml.safe_load(Path('configs/phase8_llama.yaml').read_text())
        public=json.loads(Path(cfg['public_selection']).read_text())
        # Text-only synthetic fixture follows the exact public 60 IDs/strata.
        questions=[dict(q,question='Fixture only',gold_answer='#### 2') for q in public['questions']]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selection=root/'selection.json';index=root/'index.json';provenance=root/'provenance.json'
            selection.write_text(json.dumps(dict(dataset_revision='740312add88f781978c0658806c59bc2815b9866',questions=questions)))
            public['selection_sha256']=digest(selection);index.write_text(json.dumps(public))
            prov=json.loads(Path(cfg['phase7_provenance']).read_text());prov['input_sha256']['benchmark_selection.json']=digest(index);provenance.write_text(json.dumps(prov))
            cfg.update(selection_file=str(selection),public_selection=str(index),phase7_provenance=str(provenance))
            self.assertEqual(len(inputs(cfg)),60)
            public['questions'][0]['question_id']=-1;index.write_text(json.dumps(public))
            with self.assertRaises(ValueError):inputs(cfg)

    def test_read_only_cache_preflight_and_snapshot_completeness(self):
        cfg=dict(cache_environment='PHASE8_HUB_CACHE')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'PHASE8_HUB_CACHE':tmp,'HF_HUB_CACHE':str(Path(tmp).parent/'other-cache')}):
            self.assertEqual(cache_check(cfg)['status'],'unavailable');self.assertFalse(list(Path(tmp).iterdir()))
            revision='a'*40;path=snapshot(tmp,revision);path.mkdir(parents=True)
            for name in ('config.json','generation_config.json','tokenizer.json'): (path/name).write_text('{}')
            (path/'config.json').write_text(json.dumps(dict(model_type='llama')))
            (path/'tokenizer_config.json').write_text(json.dumps(dict(chat_template='native')))
            (path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight':'model-01.safetensors'})))
            self.assertFalse(complete_snapshot(path))
            header=json.dumps({'w':dict(dtype='BF16',shape=[1],data_offsets=[0,2])}).encode()
            blob=len(header).to_bytes(8,'little')+header+b'00'
            (path/'model-01.safetensors').write_bytes(blob)
            pin=dict(model=MODEL,revision=revision,file_sizes={'model-01.safetensors':len(blob)})
            self.assertEqual(cache_check(cfg,pin)['status'],'cached_complete')
            pin['file_sizes']['model-01.safetensors']=len(blob)+1;self.assertFalse(complete_snapshot(path,pin['file_sizes']))

    def test_authorized_and_denied_metadata_checks_do_not_download(self):
        class Gated(Exception): pass
        class HTTP(Exception): pass
        info=SimpleNamespace(sha='b'*40,siblings=[SimpleNamespace(rfilename='model-01.safetensors',size=100)])
        api=SimpleNamespace(model_info=lambda *a,**k:info)
        fake=SimpleNamespace(HfApi=lambda:api,get_hf_file_metadata=lambda *a,**k:None,hf_hub_url=lambda *a,**k:'metadata-only')
        with patch.dict('sys.modules',{'huggingface_hub':fake,'huggingface_hub.utils':SimpleNamespace(GatedRepoError=Gated,HfHubHTTPError=HTTP)}),patch('dllm_unmasking.ar_baseline.cache_check',return_value=dict(status='unavailable')):
            state=check_access({});self.assertEqual(state['status'],'authorized_download_required');self.assertEqual(state['revision'],'b'*40)
            fake.get_hf_file_metadata=lambda *a,**k:(_ for _ in ()).throw(Gated('DO NOT PRINT'))
            state=check_access({});self.assertEqual(state['status'],'denied');self.assertNotIn('DO NOT PRINT',str(state))

    def test_summary_paired_matching_and_latency_scopes(self):
        import csv
        q,r=fixture()
        cfg=yaml.safe_load(Path('configs/phase8_llama.yaml').read_text())
        public=json.loads(Path(cfg['public_selection']).read_text())
        questions=[dict(v,question='Fixture',gold_answer='#### 2') for v in public['questions']]
        selected=questions[0];r.update(question_id=selected['question_id'],stratum=selected['stratum'],calculation_count=selected['calculation_count'])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);summarize(out,{selected['question_id']:r},questions,cfg)
            with (out/'paired_diffusion.csv').open() as file:
                pairs=list(csv.DictReader(file))
            self.assertEqual([v['diffusion'] for v in pairs],list(FRONTIER))
            self.assertTrue(all(v['n_matched']=='1' for v in pairs))
            with (out/'latency_comparison.csv').open() as file:
                latencies=list(csv.DictReader(file))
            self.assertEqual(len(latencies),5)
            self.assertTrue(all(v['directly_comparable']=='False' for v in latencies))
            before=(out/'paired_diffusion.csv').read_bytes()
            summarize(out,{selected['question_id']:r},questions,cfg)
            self.assertEqual(before,(out/'paired_diffusion.csv').read_bytes())

    def test_historical_sources_and_public_artifacts_unchanged(self):
        paths=[p for directory in ('reports','research','src','configs','experiments')
               for p in Path(directory).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.csv','.json','.png')]
        before={p:digest(p) for p in paths}
        fixture()
        self.assertTrue(all(digest(p)==h for p,h in before.items()))
