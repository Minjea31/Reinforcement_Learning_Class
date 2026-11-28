"""Keep private web credentials in memory and pass them to a training launch."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time


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


def launch(run_dir,arguments,connection_dir=None):
    if arguments[:1]==['--']:arguments=arguments[1:]
    if any(arg.split('=',1)[0] in ('--live-url','--live-stream-port') for arg in arguments):
        raise ValueError('The web launcher supplies its own verified live connection')
    credentials=read_credentials(connection_dir or run_dir)
    env=os.environ.copy()
    env['REACHER_LIVE_TOKEN']=credentials['site_token']
    env['REACHER_STREAM_TOKEN']=credentials['stream_token']
    command=[sys.executable,'-B','-m','training.train','--curriculum',
        '--run-dir',str(Path(run_dir).resolve()),'--live-url',credentials['origin'],
        '--live-stream-port',str(credentials.get('stream_port',8767)),*arguments]
    if '--help' not in arguments and '-h' not in arguments:
        wait_for_stream_port(credentials.get('stream_port',8767))
        ensure_mission_watcher(run_dir,connection_dir)
        ensure_history_watcher(run_dir,connection_dir,os.getpid())
        (Path(run_dir)/'train.pid').write_text(str(os.getpid())+'\n')
    os.execvpe(sys.executable,command,env)


def wait_for_stream_port(port,timeout=12.):
    """Let the previous viewer bridge close before starting the native feed."""
    deadline=time.monotonic()+timeout
    while True:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            probe.settimeout(.2)
            if probe.connect_ex(('127.0.0.1',port))!=0:return
        if time.monotonic()>=deadline:
            raise RuntimeError('The previous live stream is still running; stop the old training before restarting')
        time.sleep(.2)


def ensure_mission_watcher(run_dir,connection_dir=None):
    run=Path(run_dir).resolve();pid_file=run/'mission_replays.pid'
    run.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    legacy_script=project/'mission_replays.py'
    if pid_file.exists():
        try:
            pid=int(pid_file.read_text())
            command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if (b'web.mission_replays' in command or str(legacy_script).encode() in command) and str(run).encode() in command:return pid
        except (OSError,ValueError):pass
    env=os.environ.copy();env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (run/'mission_replays.log').open('ab',buffering=0) as log:
        command=[sys.executable,'-B','-m','web.mission_replays','--run-dir',str(run)]
        if connection_dir:command+=['--connection-dir',str(Path(connection_dir).resolve())]
        process=subprocess.Popen(command,
            cwd=project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    pid_file.write_text(str(process.pid)+'\n')
    return process.pid


def ensure_history_watcher(run_dir,connection_dir,training_pid):
    run=Path(run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    pid_file=run/'history_feed.pid'
    if pid_file.exists():
        try:
            pid=int(pid_file.read_text());command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'web.training_history' in command and str(run).encode() in command and str(training_pid).encode() in command:return pid
        except (OSError,ValueError):pass
    command=[sys.executable,'-B','-m','web.training_history','--run-dir',str(run),'--training-pid',str(training_pid)]
    if connection_dir:command+=['--connection-dir',str(Path(connection_dir).resolve())]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (run/'history_feed.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_file.write_text(str(process.pid)+'\n');return process.pid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('serve','launch'))
    parser.add_argument('--run-dir',default='runs/stage1_100m')
    parser.add_argument('--connection-dir',default='runtime/web_connection')
    args,remaining=parser.parse_known_args()
    try:
        if args.action=='serve':serve(args.run_dir)
        else:launch(args.run_dir,remaining,args.connection_dir)
    except (OSError,ValueError,KeyError,RuntimeError):
        print('웹 연결 준비가 필요합니다. 비공개 웹 연결 프로세스와 터널 상태를 확인하세요.',file=sys.stderr)
        sys.exit(1)


if __name__=='__main__':main()
