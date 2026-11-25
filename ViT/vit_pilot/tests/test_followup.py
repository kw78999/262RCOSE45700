"""Synthetic fixtures for follow-up gates only; no model forward or NPU timing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import write_json
from followup_analysis import compare_real,default_path_evidence,trace_summary


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='SYNTHETIC_FOLLOWUP_TEST_ONLY_',dir=Path(__file__).parent)
        self.root=Path(self.temp.name);(self.root/'outputs').mkdir();(self.root/'logs').mkdir()
        write_json(self.root/'config.json',dict(run_id='SYNTHETIC_TEST_ONLY',atol=None,rtol=None,trials=3))
        assets=dict(input_ids=[str(i) for i in range(4)],unique_samples=4,synthetic_input=False)
        write_json(self.root/'assets.json',assets)
        write_json(self.root/'image_manifest.json',dict(images=[dict(id=str(i),actual_absolute_path=f'/fixture/{i}',
             file_sha256='fixture_file_'+str(i),preprocessed_bf16_tensor_sha256='fixture_tensor_'+str(i)) for i in range(4)]))
        self.workers=[]
        import torch
        for b in [1,4]:
            tensor=torch.zeros(b,1000)
            for i in range(b):tensor[i,i]=1
            for mode in ['eager','compile']:
                cid=f'validation_{mode}_b{b}';s=dict(case_id=cid,stage='validation',mode=mode,batch=b,trial=1,status='ok')
                self.workers.append(s)
                write_json(self.root/'cases'/cid/'result.json',dict(output_valid=True,sync_checked=True,status='ok',
                    weights_sha256='fixture_weight',input_hash='fixture_input_'+str(b),dtype='torch.bfloat16',timing_scope='fixture_scope'))
                torch.save(tensor,self.root/'outputs'/f'{cid}.pt')
        write_json(self.root/'manifest.json',dict(workers=self.workers))

    def tearDown(self):self.temp.cleanup()

    def path(self,root,mode,b):return dict(actual_cpu_fallback_observed=mode=='eager',npu_use_confirmed=True)

    def test_mixed_eager_holds_only_that_mode(self):
        with patch('followup_analysis.default_path_evidence',side_effect=self.path):
            c=compare_real(self.root)
        self.assertEqual(c['tolerance_status'],'pending')
        for b in c['batches']:
            self.assertTrue(b['eligible_for_timing'])  # Output pair is structurally valid and predictions agree.
            self.assertFalse(b['mode_eligibility']['eager']['eligible'])
            self.assertTrue(b['mode_eligibility']['compile']['eligible'])
            self.assertFalse(b['accuracy_computed']);self.assertFalse(b['loss_computed'])

    def test_top1_mismatch_holds_both_modes(self):
        import torch
        t=torch.zeros(1,1000);t[0,9]=1;torch.save(t,self.root/'outputs/validation_compile_b1.pt')
        with patch('followup_analysis.default_path_evidence',side_effect=self.path):c=compare_real(self.root)
        b=c['batches'][0]
        self.assertFalse(b['eligible_for_timing'])
        self.assertIn('Top-1 mismatch',b['hold_reason'])

    def test_native_fallback_not_missed_by_python_logger_absence(self):
        cid='diagnostic_eager_b1'
        specs=[dict(case_id=cid,stage='diagnostic',mode='eager',batch=1,status='ok',worker_id=999),
               dict(case_id='strict_eager_b1',stage='strict',mode='eager',batch=1,status='error')]
        write_json(self.root/'manifest.json',dict(workers=specs))
        write_json(self.root/'cases'/cid/'result.json',dict(status='ok',output_valid=True,
                   device_use_snapshot=dict(stdout='| 999 | npu1:0 | fixture |')))
        write_json(self.root/'cases/strict_eager_b1/result.json',dict(error='RuntimeError: CPU fallback is disabled for aten::_native_multi_head_attention'))
        write_json(self.root/'cases'/cid/'torch_cpu_summary.json',[dict(name='aten::_native_multi_head_attention',count=12,total_cpu_ms=1000.)])
        evidence=default_path_evidence(self.root,'eager',1)
        self.assertTrue(evidence['actual_cpu_fallback_observed'])
        self.assertTrue(evidence['npu_use_confirmed'])
        self.assertEqual(evidence['full_operator_placement'],'unverified')

    def test_trace_counts_not_codegen(self):
        p=self.root/'trace.json'
        write_json(p,dict(traceEvents=[dict(name='npu1pe0::Task::compile',ph='X',dur=10),
                                      dict(name='run_by_cpu::executed_by_cpu',ph='X',dur=4)]))
        d=trace_summary(p)
        self.assertEqual(d['npu_task_count'],1)
        self.assertEqual(d['cpu_fallback_execution_spans'],1)
        self.assertEqual(d['npu_task_durations_ms'],[.01])

    def test_prior_compile_prediction_change_holds_later_trials(self):
        import torch
        for mode,status in [('eager','held'),('compile','ok')]:
            self.workers.append(dict(case_id=f'performance_{mode}_b1_t1',stage='performance',mode=mode,batch=1,trial=1,status=status))
        write_json(self.root/'manifest.json',dict(workers=self.workers))
        t=torch.zeros(1,1000);t[0,20]=2;torch.save(t,self.root/'outputs/performance_compile_b1_t1.pt')
        write_json(self.root/'cases/performance_compile_b1_t1/result.json',dict(timings=[dict(phase='repeat',status='ok',timing_valid=True,elapsed_ms=1.) for _ in range(50)]))
        with patch('followup_analysis.default_path_evidence',side_effect=self.path):c=compare_real(self.root)
        self.assertFalse(c['batches'][0]['eligible_for_timing'])
        self.assertIn('differs from validation predictions',c['batches'][0]['hold_reason'])


if __name__=='__main__':unittest.main()
