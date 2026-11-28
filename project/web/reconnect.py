"""Restore private Sites access and its video tunnel without starting training."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from urllib import error, request

from web.live_access import read_credentials

ROOT = Path(__file__).resolve().parents[1]
SITE = 'https://dual-reacher-best-replay.lunarknoll21.chatgpt.site'
KEYRING_SERVICE = 'dual-reacher-web'
TUNNEL_URL = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')


class SetupError(RuntimeError):
    pass


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def desktop_keyring():
    # Select the encrypted desktop keyring explicitly; never a plaintext fallback.
    from keyring.backends.SecretService import Keyring
    backend = Keyring()
    if backend.priority <= 0:
        raise SetupError('우분투 로그인 키링을 사용할 수 없습니다. 데스크톱에 로그인한 터미널에서 실행하세요.')
    return backend


def load_profile(connection_dir, configure=False):
    backend = desktop_keyring()
    try:
        live = read_credentials(connection_dir)
    except (OSError, ValueError):
        live = None
    if configure:
        token = getpass.getpass('Sites 서비스 인증 값 (화면에 표시되지 않음): ').strip()
        if not token:
            raise SetupError('인증 값이 비어 있습니다.')
        profile = dict(origin=SITE, site_token=token,
                       stream_token=live['stream_token'] if live else secrets.token_urlsafe(48), stream_port=8767)
        if live and profile != live:
            raise SetupError('실행 중인 연결의 인증 변경은 학습과 웹 연결을 종료한 뒤 실행하세요.')
    elif live:
        profile = live
    else:
        saved = backend.get_password(KEYRING_SERVICE, SITE)
        if not saved:
            raise SetupError('등록된 웹 인증이 없습니다. ./reconnect_web.sh --configure 로 최초 등록하세요.')
        profile = json.loads(saved)
    if (profile.get('origin') != SITE or not profile.get('site_token') or
            not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', profile.get('stream_token', '')) or profile.get('stream_port') != 8767):
        raise SetupError('저장된 웹 연결 설정이 올바르지 않습니다.')
    return profile, live is not None, backend


def site_request(profile, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = request.Request(profile['origin'] + path, data=body,
                          method='GET' if data is None else 'PUT',
                          headers={'OAI-Sites-Authorization': 'Bearer ' + profile['site_token'],
                                   'Content-Type': 'application/json'})
    try:
        with request.build_opener(NoRedirect()).open(req, timeout=20) as response:
            return json.load(response)
    except error.HTTPError as ex:
        if ex.code in (401, 403):
            raise SetupError('웹 인증이 만료됐거나 권한이 없습니다. 새 인증 값으로 --configure 를 실행하세요.') from None
        raise SetupError(f'웹 연결 요청이 실패했습니다 (HTTP {ex.code}).') from None
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        raise SetupError('웹에 연결하지 못했습니다. 인터넷 연결을 확인하고 다시 실행하세요.') from None


def tunnel_pid(folder,port=8767):
    try:
        pid = int((folder / 'tunnel.pid').read_text())
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if (Path(os.fsdecode(args[0])).name == 'cloudflared' and b'tunnel' in args and
                f'http://127.0.0.1:{port}'.encode() in args):
            return pid
    except (OSError, ValueError, IndexError):
        pass
    return None


def start_tunnel(folder, timeout=40, port=8767):
    pid = tunnel_pid(folder,port)
    log_path = folder / 'tunnel.log'
    process = None
    if pid is None:
        binary = shutil.which('cloudflared')
        if not binary:
            candidate = Path.home() / '.local/bin/cloudflared'
            if candidate.is_file() and os.access(candidate, os.X_OK):
                binary = str(candidate)
        if not binary:
            raise SetupError('cloudflared가 없습니다. 설치 후 다시 실행하세요.')
        with log_path.open('wb') as log:
            process = subprocess.Popen([binary, '--no-autoupdate', 'tunnel', '--url',
                f'http://127.0.0.1:{port}', '--protocol', 'quic', '--edge-ip-version', '4'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        pid = process.pid
        (folder / 'tunnel.pid').write_text(str(pid) + '\n')
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise SetupError('터널 프로세스가 종료됐습니다. runtime/web_connection/tunnel.log를 확인하세요.')
        matches = TUNNEL_URL.findall(log_path.read_text(errors='replace')) if log_path.exists() else []
        if matches:
            return matches[-1], pid
        time.sleep(.2)
    if process is not None:
        process.terminate()
        process.wait(timeout=5)
    raise SetupError('터널 주소를 받지 못했습니다. 인터넷 연결을 확인하고 다시 실행하세요.')


def start_broker(folder, profile, timeout=8):
    with (folder / 'connection.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-B', '-m', 'web.live_access', 'serve', '--run-dir', str(folder)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
    try:
        process.stdin.write((json.dumps(profile) + '\n').encode())
        process.stdin.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                if read_credentials(folder) == profile:
                    return process.pid
            except (OSError, ValueError):
                pass
            time.sleep(.1)
    except (OSError, ValueError):
        pass
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
    raise SetupError('로컬 웹 인증 연결을 준비하지 못했습니다. connection.log를 확인하세요.')


def reconnect(folder, configure=False):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'reconnect.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError('다른 웹 재연결 작업이 진행 중입니다.') from None
        profile, broker_running, backend = load_profile(folder, configure)
        # Authenticate before changing the tunnel or saving a credential.
        snapshot = site_request(profile, '/api/live')
        origin, pid = start_tunnel(folder)
        site_request(profile, '/api/live-connection', dict(origin=origin, token=profile['stream_token']))
        backend.set_password(KEYRING_SERVICE, SITE, json.dumps(profile))
        if not broker_running:
            start_broker(folder, profile)
        status = site_request(profile, '/api/live-connection')
        if not status.get('configured') or status.get('origin') != origin:
            raise SetupError('웹 터널 등록을 확인하지 못했습니다. 다시 실행하세요.')
    print('웹 연결 준비 완료: ' + SITE)
    print('기존 학습은 그대로 유지합니다. 새 학습을 자동으로 시작하지 않습니다.')
    run = snapshot.get('run_name', 'stage1_1kg')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', run) or not (ROOT/'runs'/run/'latest.pt').exists():
        run = 'stage1_1kg'
    print(f'학습 재개: ./run_web_training.sh --run-dir runs/{run} '
          f'--connection-dir runtime/web_connection --resume runs/{run}/latest.pt --device cuda --viewer')
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-dir', default='runtime/web_connection')
    parser.add_argument('--configure', action='store_true', help='Register or renew a Sites service credential in the login keyring')
    args = parser.parse_args()
    try:
        reconnect(args.connection_dir, args.configure)
    except SetupError as ex:
        print(str(ex), file=sys.stderr)
        return 1
    except Exception:
        print('웹 연결 준비에 실패했습니다. 우분투 로그인 키링과 인터넷 연결을 확인하세요.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
