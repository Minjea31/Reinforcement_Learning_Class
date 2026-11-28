"""Publish actual worker 0 as continuous video plus nonblocking Site telemetry."""
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time


def _publisher(messages,origin,token,run_dir,stream_port=0,stream_token=None):
    os.environ.setdefault('LP_NUM_THREADS','2')
    import base64,io,json,signal,threading
    from urllib import request,error
    import mujoco
    from PIL import Image
    from simulation.dual_reacher import DualReacherEnv
    from web.live_stream import FrameHub,start_stream
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    class NoRedirect(request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    client=request.build_opener(NoRedirect())
    env=DualReacherEnv();renderer=mujoco.Renderer(env.model,height=500,width=800)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0,0,.78];camera.distance,camera.azimuth,camera.elevation=2.45,90,-40
    metrics={};guard=threading.Lock();stop=threading.Event();hub=FrameHub();state=None
    server=start_stream(stream_port,stream_token,hub) if stream_port else None
    run_path=Path(run_dir)
    def local_json(name,data):
        temporary=run_path/(name+'.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False)+'\n');os.replace(temporary,run_path/name)
    def upload():
        sequence=0;last_ok=None;last_warning=0.
        while True:
            with hub.condition:jpeg=hub.jpeg
            if jpeg is None:
                if stop.wait(.1):break
                continue
            with guard:current=metrics.copy()
            sequence+=1;payload=dict(current,sequence=sequence,sent_at=time.time(),stream_enabled=bool(server),image=base64.b64encode(jpeg).decode('ascii'))
            temporary=run_path/'live_frame.tmp';temporary.write_bytes(jpeg);os.replace(temporary,run_path/'live_frame.jpg')
            local_json('live_status.json',{k:v for k,v in payload.items() if k!='image'})
            delay=1.
            try:
                req=request.Request(origin+'/api/live',data=json.dumps(payload,separators=(',',':')).encode(),method='POST',headers={'Content-Type':'application/json','OAI-Sites-Authorization':'Bearer '+token})
                with client.open(req,timeout=4) as response:
                    if response.status!=200:raise RuntimeError(f'HTTP {response.status}')
                last_ok=time.time();local_json('live_health.json',dict(last_upload_at=last_ok,status='connected',sequence=sequence))
            except Exception as exc:
                code=exc.code if isinstance(exc,error.HTTPError) else type(exc).__name__
                local_json('live_health.json',dict(last_upload_at=last_ok,status='retrying',error=str(code),sequence=sequence))
                if time.monotonic()-last_warning>15:print(f'Live upload pending ({code}); training continues.',flush=True);last_warning=time.monotonic()
                delay=5.
            if stop.is_set():break
            stop.wait(delay)
    uploader=threading.Thread(target=upload,daemon=True);uploader.start()
    next_render=0.;quitting=False;captured_at=None;frame_steps=None
    try:
        while not quitting:
            try:
                items=[messages.get(timeout=.02)]
                while True:
                    try:items.append(messages.get_nowait())
                    except queue.Empty:break
                for item in items:
                    if item is None:quitting=True;continue
                    with guard:metrics.update(item['metrics'])
                    if item.get('state') is not None:
                        state=item['state'];captured_at=item['captured_at'];frame_steps=item['metrics']['steps']
            except queue.Empty:pass
            now=time.monotonic()
            if state is not None and now>=next_render:
                env.set_state(state);renderer.update_scene(env.data,camera=camera)
                output=io.BytesIO();Image.fromarray(renderer.render()).save(output,format='JPEG',quality=72)
                with guard:metrics.update(frame_at=captured_at,frame_steps=frame_steps)
                hub.publish(output.getvalue());state=None
                with hub.condition:readers=hub.readers
                next_render=time.monotonic()+(1/20 if readers else .5)
    finally:
        stop.set();uploader.join(timeout=5);hub.close()
        if server:server.shutdown();server.server_close()
        renderer.close();env.close()


class LiveTrainingFeed:
    def __init__(self,origin,token,run_dir,stream_port=0,stream_token=None):
        self.process=None;self.messages=None;self.last=0.;self.interval=1/30 if stream_port else .5
        if origin:
            from urllib.parse import urlparse
            parsed=urlparse(origin)
            if parsed.scheme!='https' and parsed.hostname not in ('localhost','127.0.0.1'):raise ValueError('Live Site must use HTTPS')
            if stream_port and not stream_token:raise ValueError('Stream credentials are required')
            context=mp.get_context('spawn');self.messages=context.Queue(maxsize=4)
            self.process=context.Process(target=_publisher,args=(self.messages,origin.rstrip('/'),token,str(run_dir),stream_port,stream_token),daemon=True)
            previous=os.environ.get('MUJOCO_GL');os.environ['MUJOCO_GL']='egl'
            try:self.process.start()
            finally:
                if previous is None:os.environ.pop('MUJOCO_GL',None)
                else:os.environ['MUJOCO_GL']=previous
    def ready(self):return self.process is not None and self.process.is_alive() and time.monotonic()-self.last>=self.interval
    def _put(self,item):
        if self.messages is None or not self.process.is_alive():return
        try:self.messages.put_nowait(item)
        except queue.Full:
            try:self.messages.get_nowait();self.messages.put_nowait(item)
            except (queue.Empty,queue.Full):pass
    def send(self,state,metrics):
        if self.ready():self.last=time.monotonic();self._put(dict(state=state,metrics=metrics,captured_at=time.time()))
    def status(self,metrics):self._put(dict(metrics=metrics))
    def close(self):
        if self.process is not None:
            try:self.messages.put(None,timeout=.5)
            except queue.Full:pass
            self.process.join(timeout=7)
            if self.process.is_alive():self.process.terminate();self.process.join()
            self.messages.cancel_join_thread();self.messages.close()
