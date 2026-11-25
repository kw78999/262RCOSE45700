"""Synthetic analysis checks only. These fixtures are NEVER NPU measurements."""
from pathlib import Path
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import write_json, timed_call
from analyze import summarize, compare_outputs, valid, warm_state, diagnostic_rows


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic_analysis_only_',dir=Path(__file__).parent)
        self.run=Path(self.temp.name)
        (self.run/'outputs').mkdir(); (self.run/'logs').mkdir()
        self.config=dict(run_id='SYNTHETIC_ANALYSIS_TEST_ONLY',repeats=3,model_id='fixture',weights_id='fixture',
                         dtype='torch.bfloat16',cache_state='fresh_process_persistent_cache_possible',atol=None,rtol=None,
                         shape_sequence=[1,1,4,4,1,4])
        self.assets=dict(weights_sha256='test_weight_hash',batch_hashes={'1':'test_input_hash'},input_ids=['fixture_0'],
                         unique_samples=1,synthetic_input=True)
        self.case=dict(case_id='fixture_eager',mode='eager',kind='fixed',batch=1,trial=1,status='ok')
        self.manifest=dict(workers=[self.case],timing_scope='test_scope')

    def tearDown(self): self.temp.cleanup()

    def store(self,rows,status='ok'):
        write_json(self.run/'cases/fixture_eager/result.json',dict(status=status,timings=rows,
                   output_structure_valid=True,dtype_valid=True,runner_setup_ms=.1))

    def test_sample_sd_throughput_and_negative_extra(self):
        rows=[dict(phase='first',elapsed_ms=1.,timing_valid=True,status='ok')]
        rows += [dict(phase='repeat',elapsed_ms=x,timing_valid=True,status='ok') for x in [2.,4.,6.]]
        self.store(rows)
        s,_=summarize(self.run,self.config,self.manifest,self.assets)
        self.assertEqual(s[0]['mean_ms'],4.)
        self.assertEqual(s[0]['stdev_ms'],2.)  # sample, not population SD
        self.assertEqual(s[0]['throughput_images_s'],250.)
        self.assertEqual(s[0]['first_call_extra_ms'],-3.)

    def test_invalid_and_incomplete_never_get_throughput(self):
        rows=[dict(phase='repeat',elapsed_ms=x,timing_valid=True,status='ok') for x in [0.,-1.,2.]]
        self.store(rows,status='timeout')
        s,_=summarize(self.run,self.config,self.manifest,self.assets)
        self.assertEqual(s[0]['valid_repeats'],1)
        self.assertEqual(s[0]['mean_ms'],2.)
        self.assertIsNone(s[0]['stdev_ms'])
        self.assertIsNone(s[0]['throughput_images_s'])
        self.assertFalse(valid(dict(elapsed_ms=float('nan'),timing_valid=True,status='ok')))

    def output_pair(self,u,v):
        import torch
        e=dict(self.case,input_hash='h',weights_sha256='w',dtype='torch.bfloat16',timing_scope='s',cache_state='c')
        k=dict(e,case_id='fixture_compile',mode='compile')
        torch.save(u,self.run/'outputs/fixture_eager.pt'); torch.save(v,self.run/'outputs/fixture_compile.pt')
        return compare_outputs(self.run,e,k,self.assets,self.config)

    def test_unset_tolerance_and_mismatch_details(self):
        import torch
        u=torch.zeros(1,1000); v=u.clone()
        u[0,2]=2; v[0,3]=2
        out=self.output_pair(u,v)
        self.assertEqual(out['status'],'structure_finite_ok_tolerance_unset')
        self.assertEqual(out['top1_matches'],0)
        self.assertEqual(out['compared_images'],1)
        self.assertEqual(out['max_abs_error'],2)
        self.assertAlmostEqual(out['mean_abs_error'],.004,places=8)
        self.assertEqual(out['mismatches'][0]['eager_class'],2)
        self.assertEqual(out['mismatches'][0]['compile_class'],3)
        self.assertEqual(out['cosine_mean'],0.)

    def test_zero_cosine_and_nonfinite(self):
        import torch
        u=torch.zeros(1,1000)
        out=self.output_pair(u,u)
        self.assertIsNone(out['cosine_mean'])
        self.assertTrue(out['cosine_undefined_reasons'])
        v=u.clone(); v[0,0]=float('inf')
        self.assertEqual(self.output_pair(u,v)['status'],'failed')

    def test_explicit_tolerance(self):
        import torch
        self.config.update(atol=0.,rtol=0.)
        u=torch.ones(1,1000); v=u.clone(); v[0,0]=1.5
        self.assertEqual(self.output_pair(u,v)['status'],'failed')
        self.assertEqual(self.output_pair(u,u)['status'],'passed_explicit_tolerance')

    def test_timer_contract(self):
        calls=[]
        def sync(): calls.append('sync')
        def runner(x): calls.append('runner'); return x
        with patch('common.time.perf_counter_ns',side_effect=[1000000,3000000]):
            y,ms,ok=timed_call(runner,'x',sync)
        self.assertEqual(calls,['sync','runner','sync'])
        self.assertEqual((y,ms,ok),('x',2.,True))

    def test_diagnostic_counts_are_distinct(self):
        case=dict(case_id='switch_compile',kind='switch',mode='compile',status='ok')
        path=self.run/'cases/switch_compile'; path.mkdir(parents=True)
        events=[dict(event='backend_entry',call_index=1),dict(event='backend_entry',call_index=3)]
        import json
        (path/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
        (self.run/'logs/switch_compile.stderr.log').write_text(
            'VIT_CALL_BEGIN {"call_index": 1, "batch": 1}\ncompile without given compiler config\nVIT_CALL_END 1 ok\n'
            'VIT_CALL_BEGIN {"call_index": 3, "batch": 4}\n[__recompiles] Recompiling function forward\n'
            '[__recompiles] guard failure size mismatch\nVIT_CALL_END 3 ok\n')
        rows=diagnostic_rows(self.run,dict(workers=[case]),self.config)
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0]['backend_calls'],1)
        self.assertEqual(rows[0]['pytorch_recompile_events'],0)
        self.assertEqual(rows[2]['pytorch_recompile_events'],1)
        self.assertEqual(rows[2]['backend_event_kind'],'subsequent_backend_entry')
        self.assertIsNone(rows[0]['furiosa_actual_compile_count'])
        self.assertIsNone(rows[0]['disk_cache_hit'])

    def test_warm_state_is_only_a_heuristic(self):
        self.assertEqual(warm_state([1.]*9)[0],'unknown')
        self.assertEqual(warm_state([1.]*20)[0],'preliminary_stable')
        self.assertEqual(warm_state([1.]*10+[5.]*10)[0],'not_stable')


if __name__=='__main__': unittest.main()
