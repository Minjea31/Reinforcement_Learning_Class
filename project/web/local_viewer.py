"""Show each running algorithm's existing live stream in a local window."""
import argparse
import fcntl
import io
import json
import queue
import signal
import threading
import time
from pathlib import Path
from urllib import request

from PIL import Image, ImageDraw, ImageFont
from web.live_access import read_credentials
from web.training_history import CSVTail
from web.study_connection import study_identity


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def jpeg_frames(stream):
    """Read the bounded multipart frames emitted by web.live_stream."""
    while True:
        boundary=stream.readline(4096)
        if not boundary:return
        if not boundary.startswith(b'--frame'):continue
        size=None
        for _ in range(16):
            line=stream.readline(4096)
            if not line:raise EOFError('Video connection closed')
            if line in (b'\r\n',b'\n'):break
            key,_,value=line.partition(b':')
            if key.lower()==b'content-length':size=int(value.strip())
        else:raise ValueError('Invalid video headers')
        if size is None or not 1<=size<=2*1024*1024:raise ValueError('Invalid video frame size')
        data=bytearray()
        while len(data)<size:
            part=stream.read(size-len(data))
            if not part:raise EOFError('Incomplete video frame')
            data.extend(part)
        yield bytes(data)


def trainer_alive(run):
    pid=read_json(run/'run_status.json').get('pid')
    if not pid:return False
    try:command=Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except OSError:return False
    return b'training.algorithm_study' in command and str(run).encode() in command


def label_image(text,width,font,color='#b8c8d8'):
    # This Conda Tk exposes only bitmap fonts. Render Korean with FreeType
    # instead of relying on Tk's missing CJK font fallback.
    bounds=font.getbbox(text);height=bounds[3]-bounds[1]+8
    image=Image.new('RGB',(width,height),'#0c1016');draw=ImageDraw.Draw(image)
    x=(width-(bounds[2]-bounds[0]))//2-bounds[0]
    draw.text((x,4-bounds[1]),text,font=font,fill=color)
    return image


def receive(run,connection_dir,frames,stopped,stats):
    while not stopped.is_set() and trainer_alive(run):
        try:
            config=read_json(run/'config.json');port=int(config['stream_port'])
            if not 1<=port<=65535:raise ValueError('Invalid local video port')
            token=read_credentials(connection_dir)['stream_token']
            req=request.Request(f'http://127.0.0.1:{port}/stream',headers={'Authorization':'Bearer '+token})
            with request.urlopen(req,timeout=5) as response:
                for data in jpeg_frames(response):
                    if stopped.is_set():return
                    with Image.open(io.BytesIO(data)) as source:image=source.convert('RGB')
                    if frames.full():
                        try:frames.get_nowait()
                        except queue.Empty:pass
                    frames.put_nowait(image)
                    stats.update(frames=stats['frames']+1,last_frame_at=time.time(),connected=True)
        except (OSError,ValueError,KeyError,EOFError,queue.Full):
            stats['connected']=False
        stopped.wait(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='runs/studies/three_models_1kg_v3')
    parser.add_argument('--connection-dir',default='runtime/web_connection')
    parser.add_argument('--wait',type=float,default=0.,help='Seconds to wait for the three trainers to start')
    args=parser.parse_args();run=Path(args.run_dir).resolve()
    project=Path(__file__).resolve().parents[1]
    runtime=project/'runtime'
    algorithms=('ppo','a2c','sac')
    original=run==project/'runs/studies/three_models_1kg_v3'
    if not original:
        runtime=runtime/'viewers'/study_identity(run)
    runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/'local_viewer.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Local training windows are already open.');return
    read_credentials(args.connection_dir)
    deadline=time.monotonic()+args.wait
    while time.monotonic()<deadline and not all(trainer_alive(run/name) for name in algorithms):
        time.sleep(.25)
    import tkinter as tk
    from PIL import ImageTk
    root=tk.Tk();root.withdraw();panels={};quitting=False
    fonts=Path('/usr/share/fonts/truetype/nanum')
    regular=ImageFont.truetype(str(fonts/'NanumGothic.ttf'),15)
    bold=ImageFont.truetype(str(fonts/'NanumGothicBold.ttf'),21)
    width=min(600,max(320,(root.winfo_screenwidth()-120)//3));height=round(width*500/800)+106
    def set_label(widget,text,font=regular,color='#b8c8d8'):
        if getattr(widget,'rendered_text',None)==text:return
        photo=ImageTk.PhotoImage(label_image(text,width-12,font,color))
        widget.configure(image=photo);widget.image=photo;widget.rendered_text=text
    def close(name):
        panel=panels.pop(name,None)
        if panel:panel['stop'].set();panel['window'].destroy()
        if not panels:root.quit()
    for index,name in enumerate(algorithms):
        folder=run/name
        if not trainer_alive(folder):continue
        window=tk.Toplevel(root)
        window.title(f'{name.upper()} · {run.name}')
        top=80 if original else 120
        window.geometry(f'{width}x{height}+{70+index*(width+20)}+{top}');window.configure(bg='#0c1016')
        heading=tk.Label(window,bg='#0c1016');heading.pack(pady=(6,0))
        set_label(heading,name.upper()+' · 실제 학습 영상',bold,'#7fdbbd')
        caption=tk.Label(window,bg='#0c1016');caption.pack();set_label(caption,'영상 연결 중')
        episode_label=tk.Label(window,bg='#0c1016');episode_label.pack()
        picture=tk.Label(window,bg='#0c1016');picture.pack(expand=True,fill='both',padx=6,pady=6)
        frames=queue.Queue(maxsize=1);stop=threading.Event();stats=dict(frames=0,connected=False)
        panels[name]=dict(window=window,caption=caption,episode_label=episode_label,episodes=CSVTail(1),
                          picture=picture,frames=frames,stop=stop,stats=stats,run=folder)
        window.protocol('WM_DELETE_WINDOW',lambda name=name:close(name))
        threading.Thread(target=receive,args=(folder,args.connection_dir,frames,stop,stats),daemon=True).start()
    if not panels:root.destroy();raise RuntimeError('No running algorithm training was found')
    def stop(*_):
        nonlocal quitting
        quitting=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    phases={'training':'학습 중','evaluating':'검증 중 · 학습 화면 잠시 정지','stopped':'학습 종료','finished':'학습 완료','error':'학습 오류'}
    last_status=0.
    def refresh():
        nonlocal last_status
        if quitting:root.quit();return
        for name,panel in list(panels.items()):
            if not trainer_alive(panel['run']):close(name);continue
            try:
                frame=panel['frames'].get_nowait();frame.thumbnail((width-12,height-106))
                photo=ImageTk.PhotoImage(frame);panel['picture'].configure(image=photo);panel['picture'].image=photo
            except queue.Empty:pass
            state=read_json(panel['run']/'run_status.json')
            text=phases.get(state.get('phase'),'연결 확인 중')
            if not panel['stats']['connected']:text='영상 연결 중'
            set_label(panel['caption'],f'{text} · {state.get("steps",0):,} 스텝')
            completed=panel['episodes'].read(panel['run']/'episodes.csv')
            episode=int(completed[-1]['episode'])+1 if completed else 1
            set_label(panel['episode_label'],f'현재 학습 에피소드: {episode:,}회',color='#e7ecf3')
            panel['stats']['episode']=episode
        if time.monotonic()-last_status>=1:
            path=runtime/'local_viewer_status.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(updated_at=time.time(),panels={name:panel['stats'].copy() for name,panel in panels.items()})))
            tmp.replace(path);last_status=time.monotonic()
        if panels:root.after(40,refresh)
    root.after(0,refresh)
    try:root.mainloop()
    finally:
        for panel in panels.values():panel['stop'].set()
        root.destroy();lock.close()


if __name__=='__main__':main()
