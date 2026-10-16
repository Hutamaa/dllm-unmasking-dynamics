"""Phase 9 prospective controls, journal safety and blinded analysis: CPU fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dllm_unmasking import phase9_protocol as protocol
from dllm_unmasking.ar_baseline import load_config,object_hash,digest,read_journal,prompt_inputs
from dllm_unmasking.analyze_phase9 import analyze,first_explicit,load_completed
from dllm_unmasking.phase9_review import make_review
from dllm_unmasking.phase2_data import append_event
from dllm_unmasking.reasoning_budget import ByteLevelDecoder
from tests.test_phase8 import fixture,Tokenizer


def questions():
    public=json.loads(Path('research/benchmark_selection.json').read_text())
    return [dict(q,question=f'Synthetic math question {i}',gold_answer='#### 2') for i,q in enumerate(public['questions'])]


def sample(q,cfg,correct=True):
    _,r=fixture();answer=2 if correct else 3
    r.update(question_id=q['question_id'],stratum=q['stratum'],calculation_count=q['calculation_count'],
        model=cfg['model'],model_revision=cfg['model_revision'],policy='qwen_greedy',max_new_tokens=256,
        response=f'#### {answer}',raw_response=f'#### {answer}<eos>',generated_token_ids=[answer,9],predicted_numeric=str(answer),correct=correct,
        gpu_prefill_decode_seconds=.2,gpu_prefill_forward_seconds=.1,gpu_decode_forward_seconds=.05,tokens_per_second=10.,peak_allocated_bytes=100,
        rendered_prompt='USER: '+q['question']+'\n'+protocol.ORIGINAL+'\nASSISTANT:')
    return r


def write_journal(path,records):
    with path.open('w') as f:
        for r in records.values():
            header={k:r[k] for k in ('question_id','policy','run_hash','model','model_revision','max_new_tokens')}
            append_event(f,dict(event='started',**header));append_event(f,dict(event='completed',result=r,**header))


def synthetic_run(root):
    qs=questions();cfg=protocol.load_config();dirs={}
    tokenizer=root/'tokenizer';tokenizer.mkdir()
    (tokenizer/'tokenizer.json').write_text(json.dumps(dict(model=dict(type='BPE',vocab={'####Ġ2':2,'####Ġ3':3}),decoder=dict(type='ByteLevel'),added_tokens=[dict(id=9,content='<eos>',special=True)])))
    (tokenizer/'tokenizer_config.json').write_text(json.dumps(dict(chat_template='synthetic template')))
    native=dict(model=cfg['model'],model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],model_config=dict(model_type='qwen2'),native_generation_config=dict(eos_token_id=[9]),
        tokenizer_sha256=digest(tokenizer/'tokenizer.json'),tokenizer_config_sha256=digest(tokenizer/'tokenizer_config.json'),chat_template='synthetic template',selection_sha256='synthetic')
    refs={}
    for cap,total in ((256,28),(512,50)):
        d=root/f'baseline{cap}';d.mkdir();dirs[cap]=d;cfg[f'baseline{cap}_dir']=str(d)
        rs={q['question_id']:sample(q,cfg,i<total) for i,q in enumerate(qs)}
        (d/'manifest.json').write_text(json.dumps(native));(d/'environment.json').write_text('{}');write_journal(d/'journal.jsonl',rs);refs[cap]=(rs,native)
    m=dict(native,config=cfg,configuration_hash=object_hash(cfg),intervention=dict(condition=protocol.CONDITION,instruction=protocol.CONCISE,phase8_provenance_sha256=digest(cfg['phase8_provenance']),
        reference_manifest_sha256={str(c):digest(dirs[c]/'manifest.json') for c in dirs},reference_journal_sha256={str(c):digest(dirs[c]/'journal.jsonl') for c in dirs}))
    h=object_hash(m);m['run_hash']=h
    d=root/'run';d.mkdir();(d/'manifest.json').write_text(json.dumps(m));(d/'environment.json').write_text('{}')
    rs={}
    for i,q in enumerate(qs):
        r=sample(q,cfg,i<35);r.update(protocol.header(cfg));r['run_hash']=h;r['rendered_prompt']=r['rendered_prompt'].replace(protocol.ORIGINAL,protocol.CONCISE);rs[q['question_id']]=r
    write_journal(d/'journal.jsonl',rs)
    return cfg,qs,refs,rs,d,tokenizer


class Phase9Tests(unittest.TestCase):
    def test_exact_single_intervention_and_budget(self):
        cfg=protocol.load_config();base=load_config(model='qwen')
        self.assertEqual({k for k in base if cfg[k]!=base[k]},{'prompt_suffix','output_dir'})
        self.assertEqual(cfg['prompt_suffix'],'\n'+protocol.CONCISE)
        self.assertEqual(cfg['max_new_tokens'],256);self.assertFalse(cfg['do_sample']);self.assertTrue(cfg['use_cache'])
        for key,value in [('prompt_suffix','Be short.'),('max_new_tokens',512),('model_revision','0'*40),('output_dir',base['output_dir']),('review_seed',0)]:
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.validate_config(dict(cfg,**{key:value}))
        with self.assertRaises(ValueError):protocol.load_config(model='llama')

    def test_native_prompt_role_excludes_gold(self):
        t=Tokenizer();prompt_inputs(t,'Question',protocol.load_config()['prompt_suffix'])
        self.assertEqual(t.messages,[dict(role='user',content='Question\n'+protocol.CONCISE)])
        self.assertNotIn('gold',t.messages[0]['content'])

    def test_shared_generation_path_and_default_api(self):
        import inspect
        from experiments.phase8 import run as shared
        from experiments.phase9 import run as prospective
        from dllm_unmasking.ar_baseline import generate_one
        self.assertIs(shared.generate_one,generate_one);self.assertIs(prospective.main,shared.main)
        self.assertIsNone(inspect.signature(shared.main).parameters['protocol'].default)

    def test_condition_identity_and_prompt_placement(self):
        cfg=protocol.load_config();q=questions()[0];original=sample(q,cfg);r=copy.deepcopy(original)
        r.update(protocol.header(cfg));r['rendered_prompt']=r['rendered_prompt'].replace(protocol.ORIGINAL,protocol.CONCISE)
        with patch.object(protocol,'references',return_value=([q],{256:({q['question_id']:original},{})})):
            protocol.validate_records(cfg,{q['question_id']:r})
            for field,value in [('condition','original'),('configuration_hash','different'),('rendered_prompt','SYSTEM: '+protocol.CONCISE)]:
                bad=copy.deepcopy(r);bad[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):protocol.validate_records(cfg,{q['question_id']:bad})

    def test_resume_duplicate_and_wrong_config_rejection(self):
        cfg=protocol.load_config();q=questions()[0];r=sample(q,cfg);r.update(protocol.header(cfg))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'journal.jsonl';write_journal(p,{q['question_id']:r});before=digest(p)
            self.assertEqual(len(read_journal(p,[q],'fixture',model=cfg['model'],revision=cfg['model_revision'])),1)
            with self.assertRaises(ValueError):read_journal(p,[q],'different',repair=True,model=cfg['model'],revision=cfg['model_revision'])
            self.assertEqual(digest(p),before)
            with p.open('a') as f:append_event(f,dict(event='completed',result=r,**{k:r[k] for k in ('question_id','policy','run_hash','model','model_revision')}))
            with self.assertRaises(ValueError):read_journal(p,[q],'fixture',model=cfg['model'],revision=cfg['model_revision'])
            write_journal(p,{q['question_id']:r})
            with p.open('ab') as f:f.write(b'{"partial":')
            with self.assertRaises(ValueError):read_journal(p,[q],'fixture',model=cfg['model'],revision=cfg['model_revision'])
            self.assertEqual(len(read_journal(p,[q],'fixture',repair=True,model=cfg['model'],revision=cfg['model_revision'])),1)

    def test_environment_matches_original_gpu_and_software(self):
        cfg=protocol.load_config()
        with tempfile.TemporaryDirectory() as temp:
            cfg['baseline256_dir']=temp;(Path(temp)/'environment.json').write_text(json.dumps(dict(gpu='fixture GPU',dtype='bfloat16')))
            protocol.validate_environment(cfg,dict(gpu='fixture GPU',dtype='bfloat16'))
            with self.assertRaises(ValueError):protocol.validate_environment(cfg,dict(gpu='different GPU',dtype='bfloat16'))

    def test_blinded_packet_deterministic_key_and_no_overwrite(self):
        cfg=protocol.load_config();qs=questions();orig={q['question_id']:sample(q,cfg) for q in qs};concise=copy.deepcopy(orig)
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/'a';b=Path(temp)/'b';make_review(a,qs,orig,concise,20261015);make_review(b,qs,orig,concise,20261015)
            for p in a.rglob('*'):
                if p.is_file():self.assertEqual(p.read_bytes(),(b/p.relative_to(a)).read_bytes())
            html=(a/'review_packet/comparison.html').read_text();csv=(a/'review_packet/ratings.csv').read_text()
            self.assertNotIn('concise',html.lower());self.assertNotIn('original',html.lower());self.assertNotIn('question_id',csv)
            key=json.loads((a/'unblinding/key.json').read_text())['key'];self.assertEqual(len(key),120)
            self.assertTrue(all({r['condition'] for r in key if r['case_id']==c}=={'original','concise'} for c in {r['case_id'] for r in key}))
            with self.assertRaises(ValueError):make_review(a,qs,orig,concise,20261015)

    def test_future_analysis_full_synthetic_fixture_is_private(self):
        Path('results/phase9').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir='results/phase9') as temp:
            root=Path(temp);cfg,qs,refs,rs,d,tokenizer=synthetic_run(root)
            original_hash=digest(d/'journal.jsonl')
            with patch.object(protocol,'inputs',return_value=qs),patch.object(protocol,'references',return_value=(qs,refs)):
                contrast,review=analyze(d,root/'analysis',tokenizer)
                self.assertEqual(contrast['b_recovers'],7);self.assertEqual(contrast['b_regresses'],0);self.assertEqual(review['n_questions'],60)
                self.assertFalse(review['reasoning_quality_assessed'])
                with self.assertRaises(ValueError):analyze(d,root/'analysis',tokenizer)
                # A foreign raw journal cannot be accepted as the concise run.
                saved=(d/'journal.jsonl').read_bytes();(d/'journal.jsonl').write_bytes((root/'baseline256/journal.jsonl').read_bytes())
                with self.assertRaises(ValueError):load_completed(d)
                (d/'journal.jsonl').write_bytes(saved)
            self.assertEqual(digest(d/'journal.jsonl'),original_hash)
            self.assertIn('reasoning quality has not been assessed',(root/'analysis/PHASE9_REPORT.md').read_text())
            self.assertTrue((root/'analysis/accuracy_latency.png').exists())

    def test_analysis_rejects_public_output_and_incomplete_run(self):
        with self.assertRaises(ValueError):analyze('not-accessed','reports/phase9','not-accessed')
        Path('results/phase9').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir='results/phase9') as temp:
            root=Path(temp);cfg,qs,refs,rs,d,tokenizer=synthetic_run(root)
            write_journal(d/'journal.jsonl',{q:r for i,(q,r) in enumerate(rs.items()) if i<59})
            with patch.object(protocol,'inputs',return_value=qs),patch.object(protocol,'references',return_value=(qs,refs)),self.assertRaises(ValueError):load_completed(d)

    def test_explicit_answer_position_is_not_last_number_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,qs,refs,rs,d,tokenizer=synthetic_run(Path(temp));decoder=ByteLevelDecoder(tokenizer/'tokenizer.json')
            r=next(iter(rs.values()));positions=first_explicit(r,decoder.view(r))
            self.assertEqual(positions['first_explicit_prefix_tokens'],1)
            self.assertEqual(positions['first_explicit_native_end'],1)


    def test_phase9_disables_downloads_and_cache_miss_never_generates(self):
        from experiments.phase8 import run as shared
        with patch('sys.argv',['phase9','--download-model']),self.assertRaises(SystemExit):shared.main(protocol=protocol)
        qs=questions()
        with patch('sys.argv',['phase9','--check-cache']),patch.object(protocol,'inputs',return_value=qs),patch.object(shared,'cache_check',return_value=dict(status='unavailable')),patch.object(shared,'generate_one') as generate,self.assertRaises(SystemExit):
            shared.main(protocol=protocol)
        generate.assert_not_called()
