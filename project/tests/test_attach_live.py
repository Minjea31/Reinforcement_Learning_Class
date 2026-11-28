from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from web.attach_live import verify_trainer,Telemetry
from web.window_capture import find_viewer, is_descendant, WindowCapture,scene_only
from PIL import Image


class AttachLiveTests(unittest.TestCase):
    def test_scene_crop_removes_ui_panels_and_top_text(self):
        image=Image.new('RGB',(1000,600),(20,35,50))
        image.paste((70,70,70),(0,0,140,600));image.paste((70,70,70),(860,0,1000,600))
        image.paste((255,255,255),(140,0,860,90))
        scene=scene_only(image)
        self.assertEqual(scene.size,(720,504))
        self.assertEqual(scene.getpixel((0,0)),(20,35,50))
        self.assertEqual(scene.getpixel((719,503)),(20,35,50))

    def test_validation_phase_updates_even_when_checkpoint_has_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'latest.pt').write_bytes(b'unchanged checkpoint')
            (run/'metrics.csv').write_text('steps,fps\n100,200\n')
            telemetry=Telemetry.__new__(Telemetry)
            telemetry.run=run;telemetry.stamp=(run/'latest.pt').stat().st_mtime_ns
            telemetry.next_eval=100;telemetry.metrics=dict(phase='training',steps=96,total_steps=200)
            self.assertEqual(telemetry.refresh()['phase'],'evaluating')

    def test_live_process_identity_and_run_directory_are_required(self):
        run=Path('/tmp/current-experiment')
        command=b'python\0-m\0training.train\0--run-dir\0/tmp/current-experiment\0'
        with patch('web.attach_live.Path.read_bytes',return_value=command):
            verify_trainer(123,run)
            with self.assertRaisesRegex(ValueError,'another experiment'):
                verify_trainer(123,Path('/tmp/old-experiment'))
        with patch('web.attach_live.Path.read_bytes',return_value=b'python\0browser.py\0'):
            with self.assertRaisesRegex(ValueError,'not a training'):
                verify_trainer(123,run)

    def test_viewer_selection_excludes_browser_and_unrelated_simulations(self):
        tree='''0x123 "MuJoCo : Dual spherical-joint Reacher"\n0x456 "MuJoCo : Dual spherical-joint Reacher"\n0x789 "Dual Reacher - Google Chrome"'''
        responses=[Mock(stdout=tree),Mock(stdout='_NET_WM_PID(CARDINAL) = 444'),Mock(stdout='_NET_WM_PID(CARDINAL) = 555')]
        with patch('web.window_capture.subprocess.run',side_effect=responses),patch('web.window_capture.is_descendant',side_effect=[False,True]):
            self.assertEqual(find_viewer(123),(0x456,555))

    def test_window_capture_rejects_processes_outside_training_tree(self):
        with patch('web.window_capture.is_descendant',return_value=False):
            with self.assertRaisesRegex(ValueError,'outside'):
                WindowCapture(0x123,999,123)

    def test_process_tree_check_is_bounded_and_handles_exited_processes(self):
        self.assertTrue(is_descendant(os.getpid(),os.getpid()))
        with patch('web.window_capture.Path.read_text',side_effect=FileNotFoundError):
            self.assertFalse(is_descendant(99999999,123))


if __name__=='__main__':unittest.main()
