from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

from training.recordings import RecordingJobs
from web.live_access import ensure_mission_watcher,wait_for_stream_port


class EntrypointTests(unittest.TestCase):
    def test_restart_waits_for_old_video_source_to_release_its_port(self):
        connection=MagicMock();connection.__enter__.return_value=connection
        connection.connect_ex.side_effect=[0,0,111]
        with patch('web.live_access.socket.socket',return_value=connection),patch('web.live_access.time.sleep') as sleep:
            wait_for_stream_port(8767)
        self.assertEqual(sleep.call_count,2)

    def test_recording_launches_evaluation_module_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'recording'
            checkpoint=Path(directory)/'checkpoint.pt'
            process=Mock(pid=12345)
            process.poll.return_value=None
            with patch('training.recordings.subprocess.Popen',return_value=process) as spawn:
                jobs=RecordingJobs()
                jobs.start(checkpoint,output,1,50,5_000_000)
            args=spawn.call_args.args[0]
            self.assertEqual(args[:4],[sys.executable,'-B','-m','evaluation.record_policy'])
            self.assertIn(str(checkpoint),args)
            self.assertIn('5000000',args)
            self.assertEqual(spawn.call_args.kwargs['cwd'],Path(__file__).resolve().parents[1])
            self.assertEqual((output/'recording.pid').read_text(),'12345\n')

    def test_existing_watcher_is_reused_for_both_module_and_legacy_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory).resolve()
            (run/'mission_replays.pid').write_text('12345\n')
            project=Path(__file__).resolve().parents[1]
            commands=[['python','-m','web.mission_replays','--run-dir',str(run)],
                      ['python',str(project/'mission_replays.py'),'--run-dir',str(run)]]
            for command in commands:
                raw=b'\0'.join(part.encode() for part in command)+b'\0'
                with patch('web.live_access.Path.read_bytes',return_value=raw),patch('web.live_access.subprocess.Popen') as spawn:
                    self.assertEqual(ensure_mission_watcher(run),12345)
                    spawn.assert_not_called()


if __name__=='__main__':unittest.main()
