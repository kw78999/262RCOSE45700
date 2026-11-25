"""CPU-only fixtures. Never included in NPU run measurements."""
import importlib.util
import sys
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
worker=module('scaling_worker','worker.py');analysis=module('scaling_analysis','analyze.py')

class ContractTests(unittest.TestCase):
    def setUp(self):self.cfg=dict(min_warmup=100,max_warmup=500,window=20,mean_change=.02,cv=.05)
    def test_minimum_not_a_steady_state_proof(self):
        self.assertFalse(worker.stabilization([3]*99,self.cfg)['stable'])
        self.assertTrue(worker.stabilization([3]*100,self.cfg)['stable'])
        self.assertTrue(worker.stabilization([3]*100,self.cfg)['heuristic_only'])
    def test_recent_window_drift_and_noise(self):
        self.assertFalse(worker.stabilization([1]*40+[1.03]*20+[1]*40,self.cfg)['stable'])
        self.assertFalse(worker.stabilization([1]*40+[.9,1.1]*30,self.cfg)['stable'])
        self.assertTrue(worker.stabilization([100]*100+[1]*100,self.cfg)['stable'])
    def test_descriptive_stats_keep_slow_calls(self):
        s=analysis.stats([1,1,1,1,101])
        self.assertEqual(s['n'],5);self.assertEqual(s['mean_ms'],21)
        self.assertEqual(s['median_ms'],1);self.assertEqual(s['sample_sd_ms'],statistics.stdev([1,1,1,1,101]))
        self.assertAlmostEqual(s['p95_ms'],81);self.assertAlmostEqual(s['p99_ms'],97)
        self.assertEqual(analysis.stats([]),{})
    def test_output_metrics_have_pending_tolerance(self):
        import torch
        x=torch.tensor([[0.,1.,2.],[3.,2.,1.]])
        r=analysis.compare(x,x+.1)
        self.assertEqual(r['status'],'finite_tolerance_pending');self.assertIsNone(r['numeric_tolerance'])
        self.assertAlmostEqual(r['mean_abs_error'],.1,places=6);self.assertEqual(r['top1_matches'],2)
        self.assertEqual(analysis.compare(x,x[:1])['status'],'structural_error')
        bad=x.clone();bad[0,0]=float('nan');self.assertEqual(analysis.compare(x,bad)['status'],'nonfinite')
    def test_missing_host_header_is_not_model_unsupported(self):
        wrapped={'error':'UnsupportedOpError: failed to compile\nfatal error: assert.h: No such file or directory'}
        self.assertEqual(analysis.failure_category({'status':'error'},wrapped),'host_c_header_missing')
        self.assertEqual(analysis.failure_category({'status':'timeout'},{'error':''}),'timeout')
    def test_reanalysis_pins_finished_experiment_clock(self):
        self.assertEqual(analysis.observation_end({'status':'finished_with_coverage_recorded',
            'updated_utc':'2026-11-18T19:22:05+00:00'}).isoformat(),'2026-11-18T19:22:05+00:00')
        from datetime import datetime,timezone
        current=analysis.observation_end({'status':'running','updated_utc':'2020-01-01T00:00:00+00:00'})
        self.assertLess(abs((datetime.now(timezone.utc)-current).total_seconds()),2)

if __name__=='__main__':unittest.main()

class ArtifactFixture(unittest.TestCase):
    def test_end_to_end_grouping_pairing_scaling_without_pooling_exploration(self):
        import csv,json,tempfile
        from datetime import datetime,timezone
        experiment=module('scaling_experiment','experiment.py')
        with tempfile.TemporaryDirectory(prefix='vit_scaling_cpu_fixture_') as td:
            run=Path(td);(run/'code').mkdir();(run/'cases').mkdir()
            cfg=dict(experiment.DEFAULT,created_utc=datetime.now(timezone.utc).isoformat(),run_id='CPU_FIXTURE')
            def put(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x))
            put(run/'experiment_config.json',cfg)
            put(run/'assets.json',dict(weights_sha256='weights',state_dict_bf16_cpu_sha256='state',input_ids=['image0'],batches={'1':dict(tensor_sha256='input')}))
            workers=[]
            for stage,p,mode,latency in [('formal',1,'eager',40),('formal',1,'compile',20),('formal',2,'eager',20),('formal',2,'compile',10),('exploration',1,'eager',1)]:
                case=f'{stage}_{p}_{mode}';w=dict(case_id=case,stage=stage,pe=p,batch=1,mode=mode,trial=1,attempt=1,worker_protocol=cfg['worker_protocol'],status='ok',cache_path=case,pid=100,started_utc=cfg['created_utc'])
                n=max(500,10000//latency)
                times=[dict(phase='first',iteration=1,elapsed_ms=100,offset_s=0)]
                times += [dict(phase='warmup',iteration=i+1,elapsed_ms=latency,offset_s=i/100) for i in range(100)]
                times += [dict(phase='repeat',iteration=i+1,elapsed_ms=latency,offset_s=i/100) for i in range(n)]
                r=dict(w,timings=times,controls=[dict(mha_fastpath=False,cpu_fallback=False)],input_hash='input',weights_sha256='weights',state_dict_bf16_cpu_sha256='state',fusion=dict(num_fusion=p),repeat_minimum_met=True,runner_setup_ms=1,stabilization_status='heuristic_satisfied')
                put(run/'cases'/case/'result.json',r);workers.append(w)
            legacy=dict(workers[0],case_id='legacy_logging_exception',trial=2,worker_protocol=1)
            legacy_result=json.loads((run/'cases'/workers[0]['case_id']/'result.json').read_text())
            legacy_result.update(legacy)
            put(run/'cases'/legacy['case_id']/'result.json',legacy_result);workers.append(legacy)
            put(run/'manifest.json',dict(status='finished_fixture',workers=workers,events=[]))
            audit=analysis.analyze(run,plots=False)
            self.assertTrue(audit['positive_finite_timings'])
            with (run/'paired_speedups.csv').open() as f:paired=list(csv.DictReader(f))
            self.assertEqual(len(paired),2)
            self.assertTrue(all(float(r['eager_over_compile'])==2 for r in paired))
            with (run/'scaling_summary.csv').open() as f:scaling=list(csv.DictReader(f))
            multi=[r for r in scaling if r['pe']=='2']
            self.assertEqual(len(multi),2)
            self.assertTrue(all(float(r['pe_scaling_speedup'])==2 for r in multi))
            with (run/'condition_summary.csv').open() as f:summary=list(csv.DictReader(f))
            self.assertEqual(len(summary),5)
            formal=[r for r in summary if r['stage']=='formal']
            self.assertTrue(all(r['trials_completed']=='1' for r in formal))
            self.assertEqual(max(float(r['median_trial_images_s']) for r in formal),100)
            self.assertIn('provisional',(run/'report.md').read_text())
