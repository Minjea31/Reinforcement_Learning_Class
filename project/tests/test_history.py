import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from training.history import TrainingHistory,atomic_json
from web.training_history import CSVTail,HistoryFeed,ReportScheduler
from evaluation.training_report import collect,generate,generate_all


class HistoryTests(unittest.TestCase):
    def test_live_report_scheduler_throttles_and_finishes_after_live_job(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);atomic_json(run/'config.json',dict(report=True))
            scheduler=ReportScheduler(run);data=dict(steps=10,phase='training')
            with patch('web.training_history.report_process',return_value=None),patch('web.training_history.launch_report') as launch:
                scheduler.tick(data,True,now=0);launch.assert_called_once_with(run,live=True)
                scheduler.tick(dict(data,steps=20),True,now=59);self.assertEqual(launch.call_count,1)
                scheduler.tick(dict(data,steps=20),True,now=60);self.assertEqual(launch.call_count,2)
                scheduler.tick(dict(data,steps=20),True,now=120);self.assertEqual(launch.call_count,2)
                terminal=dict(data,phase='finished')
                with patch('web.training_history.report_process',return_value=123):scheduler.tick(terminal,False,now=121)
                self.assertEqual(launch.call_count,2)
                scheduler.tick(terminal,False,now=122);launch.assert_called_with(run)
                atomic_json(run/'reports/index.json',dict(live=False,status='finished',steps=10))
                scheduler.tick(terminal,False,now=123);self.assertEqual(launch.call_count,3)

    def test_batches_keep_stages_separate_and_skip_unchanged_completed_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);atomic_json(run/'config.json',dict(total_steps=40))
            atomic_json(run/'run_status.json',dict(phase='training'))
            metrics=run/'metrics.csv';metrics.write_text('steps,stage,mean_return\n10,1,2\n20,2,3\n')
            def fake_generate(run,stage=None,**kwargs):
                return dict(stage=stage,steps=[row['steps'] for row in kwargs['dataset']['metrics']])
            with patch('evaluation.training_report.generate',side_effect=fake_generate) as render:
                first=generate_all(run,live=True)
                self.assertEqual(first['1']['steps'],[10]);self.assertEqual(first['2']['steps'],[20]);self.assertNotIn('3',first)
                self.assertEqual(render.call_count,3)
                generate_all(run,live=True);self.assertEqual(render.call_count,3)
                with metrics.open('a') as stream:stream.write('30,2,4\n')
                generate_all(run,live=True);self.assertEqual([call.args[1] for call in render.call_args_list[3:]],[None,2])
                with metrics.open('a') as stream:stream.write('40,3,5\n')
                latest=generate_all(run,live=True)
                self.assertEqual(latest['3']['steps'],[40]);self.assertNotIn('4',latest)
                self.assertEqual(json.loads((run/'reports/index.json').read_text())['stages'],[1,2,3])

    def test_exact_episode_return_is_durable_and_resumes_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);config=dict(total_steps=16,report=False,curriculum=True)
            atomic_json(run/'config.json',config)
            info=dict(episode_return=12.345,is_success=True,used_arms=np.array([True,False]),episode_lifted=True,
                      episode_effort=.12,goal_distance=.01,episode_step=42)
            for step in (8,16):
                history=TrainingHistory(run,config,step-8);history.episode(info,step,0,2,0,True);history.finish('finished',step)
            with (run/'episodes.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual([row['episode'] for row in rows],['1','2'])
            self.assertEqual([float(row['reward']) for row in rows],[12.345,12.345])
            feed=HistoryFeed(run).snapshot(False)
            self.assertEqual(feed['episode_mode'],'exact');self.assertEqual(len(feed['episodes']),2)
            self.assertEqual(feed['episodes'][0]['episode_steps'],42)
            manifest=json.loads((run/'artifacts/source_manifest.json').read_text())
            self.assertTrue(any(row['path']=='models/dual_reacher.xml' for row in manifest))
            self.assertFalse(any('runs/' in row['path'] or 'token' in row['path'] for row in manifest))

    def test_csv_tail_waits_for_complete_lines_and_preserves_empty_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.csv';path.write_text('steps,mean_return\n1,\n2,3')
            tail=CSVTail();self.assertEqual(tail.read(path),[dict(steps='1',mean_return='')])
            with path.open('a') as stream:stream.write('.5\n')
            self.assertEqual(tail.read(path)[-1],dict(steps='2',mean_return='3.5'))
            path.write_text('steps,mean_return\n');self.assertEqual(tail.read(path),[])

    def test_report_limits_parent_branch_and_stage_and_exports_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'parent';current=root/'current'
            parent.mkdir();current.mkdir()
            for run in (parent,current):atomic_json(run/'config.json',dict(total_steps=40,task_version=2))
            (parent/'metrics.csv').write_text('steps,stage,mean_return,training_success_rate,fps\n10,1,2,0,100\n20,1,5,1,100\n30,2,-4,0,100\n')
            (current/'metrics.csv').write_text('steps,stage,mean_return,training_success_rate,fps\n25,2,7,0,100\n40,2,9,1,100\n')
            atomic_json(current/'artifacts/run_metadata.json',dict(parent_run=str(parent),parent_cutoff_steps=20))
            atomic_json(parent/'validation_20.json',dict(env_config=dict(stage=1),success_rate=1.))
            atomic_json(parent/'validation_30.json',dict(env_config=dict(stage=2),success_rate=0.))
            atomic_json(parent/'stage1_final_evaluation.json',dict(episodes=2,success_rate=1.,mean_return=5.,records=[dict(seed=1,success=True)]))
            self.assertEqual([row['steps'] for row in collect(current)['metrics']],[10,20,25,40])
            self.assertEqual([row['steps'] for row in collect(parent,1)['metrics']],[10,20])
            summary=generate(parent,1)
            self.assertEqual((summary['last_steps'],summary['total_steps']),(20,20))
            self.assertEqual(summary['status'],'stage_completed')
            for filename in summary['figures']:
                path=parent/'reports/stage1/figures'/filename
                with Image.open(path) as image:self.assertGreater(image.width,2000)
                self.assertTrue(path.with_suffix('.svg').exists())
            self.assertEqual(summary['completed_episode_records'],0)
            self.assertTrue((parent/'reports/stage1/ppt_notes.md').exists())


if __name__=='__main__':unittest.main()
