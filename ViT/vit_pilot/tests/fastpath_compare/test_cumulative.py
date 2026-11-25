"""Synthetic arithmetic fixtures ONLY; never written into an NPU run."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastpath_compare_analysis import analyze_cumulative


class CumulativeIntegrationTest(unittest.TestCase):
    def fixture(self,root,changed_start=False):
        workers=[]
        for stage in ['performance','cumulative']:
            for batch in [1,4]:
                for mode in ['eager','compile']:
                    cid=f'{stage}_{mode}_b{batch}_t1'
                    workers.append(dict(case_id=cid,stage=stage,mode=mode,batch=batch,trial=1,
                                        status='ok' if batch==1 else 'held',target_calls=30))
                    if batch==4:continue
                    first=4 if mode=='eager' else 9 if changed_start and stage=='cumulative' else 104
                    work=[10 if mode=='eager' else 5]* (50 if stage=='performance' else 30)
                    if changed_start and stage=='cumulative' and mode=='compile':work[2]=30
                    timings=[]
                    for phase,values in [('first',[first]),('warmup',[.8]*5),('repeat' if stage=='performance' else 'work',work)]:
                        timings += [dict(phase=phase,iteration=i,elapsed_ms=v,status='ok',timing_valid=True) for i,v in enumerate(values,1)]
                    folder=root/'cases'/cid;folder.mkdir(parents=True)
                    (folder/'result.json').write_text(json.dumps(dict(status='ok',output_valid=True,runner_setup_ms=2,timings=timings)))
        return dict(workers=workers)

    def run_fixture(self,changed_start=False):
        with tempfile.TemporaryDirectory(prefix='vit_analysis_fixture_') as temp:
            root=Path(temp);m=self.fixture(root,changed_start)
            result=analyze_cumulative(root,dict(run_id='SYNTHETIC_TEST_ONLY',warmup=5,repeats=50),m,
                [dict(batch=1,trial=1,N_win=21)],
                [dict(batch=1,representative_trial=1,N_win=dict(median=21,minimum=21,maximum=21)),dict(batch=4,representative_trial=None,N_win=dict(median=None))],
                [dict(stage='cumulative',batch=1,eligible=True),dict(stage='cumulative',batch=4,eligible=False)])
            with (root/'cumulative_validation.csv').open() as f:rows=list(csv.DictReader(f))
            return result,rows

    def test_exact_observed_crossing(self):
        result,rows=self.run_fixture()
        b=result['batches'][0]
        self.assertEqual(b['first_observed_compile_advantage_N'],21)
        self.assertEqual(b['final_winning_suffix_start_N'],21)
        self.assertEqual(b['first_observed_K'],27)
        tie=next(r for r in rows if int(r['N'])==20)
        self.assertEqual(float(tie['observed_difference_ms']),0)

    def test_transitions_and_no_refit(self):
        result,rows=self.run_fixture(changed_start=True)
        b=result['batches'][0]
        self.assertEqual(b['first_observed_compile_advantage_N'],2)
        self.assertEqual(b['final_winning_suffix_start_N'],7)
        r=next(r for r in rows if int(r['N'])==21)
        self.assertEqual(float(r['predicted_compile_ms']),215)
        self.assertEqual(float(r['predicted_eager_ms']),220)
        self.assertFalse(result['primary_predictions_refitted'])


if __name__=='__main__':unittest.main()
