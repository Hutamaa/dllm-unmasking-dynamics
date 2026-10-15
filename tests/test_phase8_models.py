"""Multi-model Phase 8 contracts using synthetic cache/generate/results fixtures."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm_unmasking.ar_baseline import (MODEL, POLICY, load_config, model_spec, snapshot, cache_check, complete_snapshot,
    prompt_inputs, generate_one, validate_result, read_journal, digest, object_hash)
from dllm_unmasking.ar_analysis import summarize
from dllm_unmasking.ar_compare import compare, load_run, mark_dominance
from dllm_unmasking.phase2_data import append_event
from tests.test_phase8 import Toy, Tokenizer, fixture

QWEN='Qwen/Qwen2.5-7B-Instruct'


class QwenTemplateFixture(Tokenizer):
    """Untooled ChatML contract fixture, not a downloaded Qwen tokenizer."""
    pad_token_id=11  # Synthetic fixture ID; production reads the native tokenizer.
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages;self.template_args=kwargs
        return ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
                + ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)
                + ('<|im_start|>assistant\n' if kwargs.get('add_generation_prompt') else ''))


class MultiModelTests(unittest.TestCase):
    def test_selection_and_legacy_llama_defaults(self):
        q=load_config();l=load_config(model='llama')
        self.assertEqual(q,load_config('configs/phase8.yaml'))
        self.assertEqual(q['model'],QWEN);self.assertEqual(l['model'],MODEL)
        self.assertEqual(model_spec(MODEL)['policy'],POLICY)
        self.assertEqual(model_spec(QWEN)['policy'],'qwen_greedy')
        self.assertFalse(model_spec(QWEN)['gated'])
        self.assertNotEqual(q['output_dir'],l['output_dir']);self.assertNotEqual(q['model_pin'],l['model_pin'])
        for field in ('selection_file','prompt_suffix','max_new_tokens','do_sample','use_cache','batch_size','dtype','inference_seed'):
            self.assertEqual(q[field],l[field])
        with self.assertRaises(ValueError):load_config('configs/phase8_llama.yaml','qwen')
        with self.assertRaises(ValueError):model_spec('unsupported/model')

    def test_qwen_native_contract_special_tokens_and_prompt_exclusion(self):
        tokenizer=QwenTemplateFixture();cfg=load_config()
        text,encoded=prompt_inputs(tokenizer,'Question',cfg['prompt_suffix'])
        self.assertTrue(text.endswith('<|im_start|>assistant\n'))
        self.assertIn('<|im_start|>user\nQuestion'+cfg['prompt_suffix']+'<|im_end|>',text)
        self.assertEqual(tokenizer.template_args,dict(tokenize=False,add_generation_prompt=True))
        self.assertNotIn('enable_thinking',tokenizer.template_args)
        self.assertFalse(tokenizer.encode_args['add_special_tokens'])
        toy=Toy();r=generate_one(toy,tokenizer,encoded,[9])
        self.assertEqual(toy.kwargs['pad_token_id'],11);self.assertEqual(toy.kwargs['eos_token_id'],[9])
        self.assertEqual(r['generated_token_ids'],[2,9]);self.assertEqual(r['termination'],'eos')
        self.assertEqual(r['response'],'#### 2')
        r=generate_one(Toy(False),tokenizer,encoded,[9])
        self.assertEqual(r['generated_tokens'],256);self.assertEqual(r['termination'],'length_limit')

    def test_model_revision_and_journal_isolation(self):
        q,r=fixture();cfg=load_config()
        r.update(model=QWEN,model_revision=cfg['model_revision'],policy='qwen_greedy')
        validate_result(r,q,'fixture',False,QWEN,cfg['model_revision'])
        with self.assertRaises(ValueError):validate_result(r,q,'fixture',False,MODEL)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl'
            header={k:r[k] for k in ('question_id','policy','run_hash','model','model_revision')}
            with p.open('w') as f:
                append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=r,**header))
            self.assertEqual(len(read_journal(p,[q],'fixture',require_gpu=False,model=QWEN,revision=cfg['model_revision'])),1)
            with self.assertRaises(ValueError):read_journal(p,[q],'fixture',require_gpu=False)
            with self.assertRaises(ValueError):read_journal(p,[q],'fixture',require_gpu=False,model=QWEN,revision='0'*40)
            with self.assertRaises(ValueError):summarize(Path(tmp),{q['question_id']:r},[q],load_config(model='llama'))

    def test_revision_specific_cache_selection(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'PHASE8_HUB_CACHE':tmp,'HF_HUB_CACHE':str(Path(tmp).parent/'legacy-cache')}):
            q=load_config();l=load_config(model='llama')
            qp=snapshot(tmp,q['model_revision'],QWEN);lp=snapshot(tmp,l['model_revision'])
            self.assertNotEqual(qp,lp)
            self.assertIn('models--Qwen--Qwen2.5-7B-Instruct',str(qp))
            self.assertEqual(cache_check(q)['revision'],q['model_revision'])
            with self.assertRaises(ValueError):cache_check(q,dict(model=MODEL,revision=l['model_revision']))
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_cache_model_type_cannot_be_mislabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for name in ('config.json','generation_config.json','tokenizer.json'):(path/name).write_text('{}')
            (path/'tokenizer_config.json').write_text(json.dumps(dict(chat_template='native fixture')))
            header=json.dumps({'w':dict(dtype='BF16',shape=[1],data_offsets=[0,2])}).encode()
            (path/'model.safetensors').write_bytes(len(header).to_bytes(8,'little')+header+b'00')
            (path/'config.json').write_text(json.dumps(dict(model_type='qwen2')))
            self.assertTrue(complete_snapshot(path,model=QWEN))
            self.assertFalse(complete_snapshot(path,model=MODEL))

    def test_qwen_metadata_is_explicitly_anonymous(self):
        from types import SimpleNamespace
        class Gated(Exception):pass
        class HTTP(Exception):pass
        from dllm_unmasking.ar_baseline import check_access
        cfg=load_config();seen=[]
        def info(model,**kwargs):
            seen.append((model,kwargs));return SimpleNamespace(sha=cfg['model_revision'],siblings=[SimpleNamespace(rfilename='model.safetensors',size=100)])
        fake=SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=info),
                             get_hf_file_metadata=lambda *a,**k:seen.append(('HEAD',k)),hf_hub_url=lambda *a,**k:'fixture')
        with patch.dict('sys.modules',{'huggingface_hub':fake,'huggingface_hub.utils':SimpleNamespace(GatedRepoError=Gated,HfHubHTTPError=HTTP)}),patch('dllm_unmasking.ar_baseline.cache_check',return_value=dict(status='unavailable')):
            state=check_access(cfg,cfg['model_revision'])
        self.assertEqual(state['status'],'authorized_download_required')
        self.assertEqual(seen[0][0],QWEN);self.assertTrue(all(x[1]['token'] is False for x in seen))

    def test_combined_analysis_keeps_complete_models_separate(self):
        # Synthetic GPU metrics exist only within this temporary fixture directory.
        public=json.loads(Path('research/benchmark_selection.json').read_text())
        qs=[dict(q,question='Fixture only',gold_answer='#### 2') for q in public['questions']]
        with tempfile.TemporaryDirectory(dir='results') as tmp:
            root=Path(tmp);selection=root/'selection.json';selection.write_text(json.dumps(qs))
            configs=[];paths=[]
            for alias,latency in [('qwen',.2),('llama',.3)]:
                cfg=load_config(model=alias);cfg['selection_file']=str(selection)
                path=root/alias;path.mkdir();paths.append(path);configs.append(cfg)
                manifest=dict(config=cfg,model=cfg['model'],model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],
                              selection_sha256=digest(selection),diffusion_sha256=digest(cfg['diffusion_results']))
                h=object_hash(manifest);manifest['run_hash']=h;(path/'manifest.json').write_text(json.dumps(manifest))
                with (path/'journal.jsonl').open('w') as f:
                    for q in qs:
                        _,r=fixture();r.update(question_id=q['question_id'],stratum=q['stratum'],calculation_count=q['calculation_count'],run_hash=h,
                                               gpu_prefill_decode_seconds=latency,gpu_prefill_forward_seconds=.1,gpu_decode_forward_seconds=.05,
                                               tokens_per_second=2/latency,peak_allocated_bytes=100)
                        header=dict(question_id=q['question_id'],policy=POLICY,run_hash=h)
                        if alias=='qwen':
                            r.update(model=QWEN,model_revision=cfg['model_revision'],policy='qwen_greedy')
                            header.update(model=QWEN,model_revision=cfg['model_revision'],policy='qwen_greedy')
                        # Llama intentionally retains its original schema without model fields.
                        append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=r,**header))
            before={str(p/n):digest(p/n) for p in paths for n in ('manifest.json','journal.jsonl')}
            with patch('dllm_unmasking.ar_compare.inputs',return_value=qs):
                points=compare(root/'combined',paths,configs[0])
                self.assertEqual(len(points),6)
                ar=[p for p in points if p['family']=='ar'];self.assertEqual([p['n'] for p in ar],[60,60])
                self.assertFalse(ar[0]['descriptive_cross_scope_dominated']);self.assertTrue(ar[1]['descriptive_cross_scope_dominated'])
                with self.assertRaises(ValueError):compare(root/'duplicate',[paths[0],paths[0]],configs[0])
            self.assertTrue((root/'combined/accuracy_latency.png').exists())
            self.assertTrue(all(digest(p)==h for p,h in before.items()))

    def test_partial_observations_do_not_define_frontier(self):
        points=[dict(n=60,mean_gpu_seconds=1.,accuracy=.5),dict(n=12,mean_gpu_seconds=.01,accuracy=1.)]
        mark_dominance(points)
        self.assertFalse(points[0]['descriptive_cross_scope_dominated'])
        self.assertIsNone(points[1]['descriptive_cross_scope_dominated'])
