"""Keep private web credentials in memory and pass them to a training launch."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys


def address(run_dir):
    return str(Path(run_dir).resolve()/'live_access.sock')


def read_credentials(run_dir):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(3)
        connection.connect(address(run_dir))
        connection.sendall(b'credentials\n')
        with connection.makefile('rb') as response:
            return json.loads(response.readline(32768))


def serve(run_dir):
    if sys.stdin.isatty():
        import termios
        settings=termios.tcgetattr(sys.stdin.fileno())
        settings[3]&=~termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,settings)
    print('Ready for private web credentials on stdin (hidden).',flush=True)
    credentials=json.loads(sys.stdin.readline())
    if not credentials['origin'].startswith('https://'):
        raise ValueError('The web origin must use HTTPS')
    for name in ('site_token','stream_token'):
        if not credentials.get(name):raise ValueError('Missing web credentials')
    path=Path(address(run_dir));path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as probe:
                probe.connect(str(path))
        except (ConnectionRefusedError,FileNotFoundError):path.unlink(missing_ok=True)
        else:raise RuntimeError('The private web connection is already running')
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    payload=(json.dumps(credentials)+'\n').encode()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        previous=os.umask(0o177)
        try:server.bind(str(path))
        finally:os.umask(previous)
        server.listen(4);server.settimeout(.5)
        (path.parent/'live_access.pid').write_text(str(os.getpid())+'\n')
        print('Private web connection ready; credentials stay in memory.',flush=True)
        try:
            while not stopped:
                try:connection,_=server.accept()
                except socket.timeout:continue
                with connection:
                    connection.settimeout(2)
                    _,uid,_=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    if uid!=os.getuid():continue
                    try:
                        with connection.makefile('rb') as request:
                            if request.readline(64)==b'credentials\n':connection.sendall(payload)
                    except (OSError,TimeoutError):pass
        finally:path.unlink(missing_ok=True)


def launch(run_dir,arguments):
    if arguments[:1]==['--']:arguments=arguments[1:]
    if any(arg.split('=',1)[0] in ('--live-url','--live-stream-port') for arg in arguments):
        raise ValueError('The web launcher supplies its own verified live connection')
    credentials=read_credentials(run_dir)
    env=os.environ.copy()
    env['REACHER_LIVE_TOKEN']=credentials['site_token']
    env['REACHER_STREAM_TOKEN']=credentials['stream_token']
    command=[sys.executable,'-B',str(Path(__file__).with_name('train.py')),'--curriculum',
        '--run-dir',str(Path(run_dir).resolve()),'--live-url',credentials['origin'],
        '--live-stream-port',str(credentials.get('stream_port',8767)),*arguments]
    if '--help' not in arguments and '-h' not in arguments:
        (Path(run_dir)/'train.pid').write_text(str(os.getpid())+'\n')
    os.execvpe(sys.executable,command,env)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('serve','launch'))
    parser.add_argument('--run-dir',default='runs/stage1_100m')
    args,remaining=parser.parse_known_args()
    try:
        if args.action=='serve':serve(args.run_dir)
        else:launch(args.run_dir,remaining)
    except (OSError,ValueError,KeyError,RuntimeError):
        print('웹 연결 준비가 필요합니다. 비공개 웹 연결 프로세스와 터널 상태를 확인하세요.',file=sys.stderr)
        sys.exit(1)
