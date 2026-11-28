import json
import os
from pathlib import Path
import signal
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch

from web.live_access import read_credentials
from web.reconnect import SITE, SetupError, load_profile, reconnect, start_broker, start_tunnel, tunnel_pid


PROFILE = dict(origin=SITE, site_token='test-only-site-token', stream_token='s' * 64, stream_port=8767)


class ReconnectTests(unittest.TestCase):
    def test_after_reboot_profile_is_read_from_desktop_keyring(self):
        backend = Mock()
        backend.get_password.return_value = json.dumps(PROFILE)
        with patch('web.reconnect.desktop_keyring', return_value=backend), \
                patch('web.reconnect.read_credentials', side_effect=ConnectionRefusedError):
            profile, running, _ = load_profile('/tmp/unused')
        self.assertEqual(profile, PROFILE)
        self.assertFalse(running)

    def test_existing_connection_registers_tunnel_without_restarting_or_training(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder/'runs/random_run/latest.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.touch()
            origin = 'https://existing-connection.trycloudflare.com'
            with patch('web.reconnect.load_profile', return_value=(PROFILE, True, backend)), \
                    patch('web.reconnect.start_tunnel', return_value=(origin, 123)), \
                    patch('web.reconnect.site_request', side_effect=[{'run_name': 'random_run'}, {'ok': True}, {'configured': True, 'origin': origin}]) as remote, \
                    patch('web.reconnect.start_broker') as broker, \
                    patch('web.reconnect.subprocess.Popen') as spawn, \
                    patch('web.reconnect.ROOT', folder), patch('builtins.print') as output:
                self.assertEqual(reconnect(folder), 123)
                self.assertIn('runs/random_run/latest.pt', output.call_args.args[0])
            broker.assert_not_called()
            spawn.assert_not_called()
            self.assertEqual(remote.call_args_list[1].args, (PROFILE, '/api/live-connection', dict(origin=origin, token=PROFILE['stream_token'])))
            self.assertEqual(json.loads(backend.set_password.call_args.args[2]), PROFILE)
            self.assertNotIn(PROFILE['site_token'], ''.join(path.read_text() for path in folder.iterdir() if path.is_file()))

    def test_failed_authentication_does_not_change_running_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('web.reconnect.load_profile', return_value=(PROFILE, True, Mock())), \
                    patch('web.reconnect.site_request', side_effect=SetupError('expired')), \
                    patch('web.reconnect.start_tunnel') as tunnel:
                with self.assertRaisesRegex(SetupError, 'expired'):
                    reconnect(directory)
                tunnel.assert_not_called()

    def test_stale_pid_is_not_treated_as_a_tunnel_and_existing_tunnel_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'tunnel.pid').write_text(str(os.getpid()))
            self.assertIsNone(tunnel_pid(folder))
            (folder / 'tunnel.log').write_text('https://existing.trycloudflare.com\n')
            with patch('web.reconnect.tunnel_pid', return_value=123), patch('web.reconnect.subprocess.Popen') as spawn:
                self.assertEqual(start_tunnel(folder), ('https://existing.trycloudflare.com', 123))
                spawn.assert_not_called()

    def test_cold_start_broker_passes_secrets_in_stdin_and_serves_private_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pid = start_broker(folder, PROFILE)
            try:
                self.assertEqual(read_credentials(folder), PROFILE)
                self.assertEqual(stat.S_IMODE((folder / 'live_access.sock').stat().st_mode), 0o600)
                args = Path(f'/proc/{pid}/cmdline').read_bytes()
                self.assertNotIn(PROFILE['site_token'].encode(), args)
                self.assertNotIn(PROFILE['stream_token'].encode(), args)
                self.assertNotIn(PROFILE['site_token'], (folder / 'connection.log').read_text())
            finally:
                os.kill(pid, signal.SIGTERM)
                os.waitpid(pid, 0)


if __name__ == '__main__':
    unittest.main()
