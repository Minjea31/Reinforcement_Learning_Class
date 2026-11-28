"""Publish learning history and periodically refresh overall/per-stage figures."""
import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from urllib import request

from training.history import atomic_json,launch_report,report_process
from web.live_access import read_credentials


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class CSVTail:
    def __init__(self,limit=200):
        self.limit=limit;self.rows=deque(maxlen=limit);self.position=0;self.header=None;self.path=None

    def read(self,path):
        path=Path(path)
        if not path.exists():return list(self.rows)
        if self.path!=path or path.stat().st_size<self.position:
            self.rows.clear();self.position=0;self.header=None;self.path=path
        with path.open() as stream:
            if self.position==0:self.header=next(csv.reader([stream.readline()]));self.position=stream.tell()
            stream.seek(self.position)
            while True:
                start=stream.tell();line=stream.readline()
                if not line:break
                if not line.endswith('\n'):stream.seek(start);break
                values=next(csv.reader([line]))
                if len(values)==len(self.header):self.rows.append(dict(zip(self.header,values)))
                self.position=stream.tell()
        return list(self.rows)


def number(value):
    try:
        result=float(value) if value not in ('',None) else None
        return result if result is not None and math.isfinite(result) else None
    except (TypeError,ValueError):return None


class HistoryFeed:
    def __init__(self,run):
        self.run=Path(run).resolve();self.metrics=CSVTail(300);self.episodes=CSVTail(200)

    def snapshot(self,alive=True):
        config=json.loads((self.run/'config.json').read_text())
        metrics_path=self.run/'metrics_v2.csv'
        if not metrics_path.exists():metrics_path=self.run/'metrics.csv'
        updates=self.metrics.read(metrics_path);episodes=self.episodes.read(self.run/'episodes.csv')
        last=updates[-1] if updates else {};steps=int(number(last.get('steps')) or 0)
        status_path=self.run/'run_status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if not status and (self.run/'live_status.json').exists():status=json.loads((self.run/'live_status.json').read_text())
        steps=max(steps,int(number(status.get('steps')) or 0),max((int(number(row.get('steps')) or 0) for row in episodes),default=0))
        phase=status.get('phase','training') if alive else status.get('phase') if status.get('phase') in ('finished','stopped','error') else 'finished' if steps>=config['total_steps'] else 'stopped'
        rows=[]
        for raw in episodes:
            rows.append({key:number(raw.get(key)) for key in ('episode','env_id','steps','stage','reward','success','lifted','effort','episode_steps')}|{'termination':raw.get('termination')})
        return dict(run_name=config.get('web_run_name',self.run.name),algorithm=config.get('algorithm'),study_id=config.get('study_id'),updated_at=time.time(),phase=phase,steps=steps,total_steps=config['total_steps'],
            last_update_at=metrics_path.stat().st_mtime if metrics_path.exists() else None,
            stage=int(number(status.get('stage')) or number(last.get('stage')) or config.get('stage') or 1),graph_refresh_seconds=60,
            payload_mass=config.get('payload_mass',1.),reward_version=config.get('reward_version'),task_version=config.get('task_version') or 1,layout_mode=config.get('layout_mode') or 'fixed',
            episode_mode='exact' if (self.run/'episodes.csv').exists() else 'update_average',
            episodes=rows,updates=[{key:number(row.get(key)) for key in ('steps','updates','fps','episodes','mean_return','training_success_rate','stage','level','policy_loss','value_loss','actor_loss','critic_loss','temperature','entropy','action_saturation','action_std')} for row in updates])


class ReportScheduler:
    def __init__(self,run,interval=60):
        self.run=Path(run);self.interval=interval;self.next_at=0.;self.last_steps=None

    def tick(self,data,alive,now=None):
        now=time.monotonic() if now is None else now
        config=json.loads((self.run/'config.json').read_text())
        if not config.get('report',True):return
        if report_process(self.run):return
        if alive:
            if now<self.next_at or data['steps']==self.last_steps:return
            launch_report(self.run,live=True)
            self.next_at=now+self.interval;self.last_steps=data['steps']
        elif not self.final_ready(data):launch_report(self.run)

    def final_ready(self,data):
        path=self.run/'reports/index.json'
        if not path.exists():return False
        index=json.loads(path.read_text())
        return not index.get('live',True) and index.get('status')==data['phase'] and index.get('steps',0)>=data['steps']


def publish(run,connection_dir,data,previous_report=None):
    credentials=read_credentials(connection_dir or run);client=request.build_opener(NoRedirect())
    headers={'OAI-Sites-Authorization':'Bearer '+credentials['site_token']}
    req=request.Request(credentials['origin']+'/api/training-history',data=json.dumps(data,separators=(',',':')).encode(),
                        method='POST',headers=dict(headers,**{'Content-Type':'application/json'}))
    with client.open(req,timeout=8) as response:
        if response.status!=200:raise RuntimeError('History upload rejected')
    previous_report=previous_report or {}
    slots=[('report',Path(run)/'reports',data['run_name'],None)]
    slots.extend(('stage_reports',Path(run)/f'reports/stage{stage}',data['run_name'],str(stage)) for stage in range(1,5))
    for field,folder,report_run,stage in slots:
        summary=folder/'summary.json'
        cache_key=field+(stage or '')
        if not summary.exists() or summary.stat().st_mtime_ns==previous_report.get(cache_key):continue
        stamp=summary.stat().st_mtime_ns
        report=json.loads(summary.read_text())
        report['asset_base']=f'/api/training-assets/{report_run}/{report["generation"]}/'
        for name in report['figures']:
            path=folder/'figures'/name
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=report['figure_hashes'][name]:raise RuntimeError('Report generation is still in progress')
            req=request.Request(credentials['origin']+report['asset_base']+name,data=raw,
                method='PUT',headers=dict(headers,**{'Content-Type':'image/png'}))
            with client.open(req,timeout=20) as response:
                if response.status!=200:raise RuntimeError('Figure upload rejected')
        value={stage:report} if stage else report
        req=request.Request(credentials['origin']+'/api/training-history',data=json.dumps(dict(data,**{field:value}),separators=(',',':')).encode(),
            method='POST',headers=dict(headers,**{'Content-Type':'application/json'}))
        with client.open(req,timeout=8) as response:
            if response.status!=200:raise RuntimeError('Report publication rejected')
        previous_report[cache_key]=stamp
    return previous_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--connection-dir')
    parser.add_argument('--training-pid',type=int,required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    try:os.nice(10)
    except OSError:pass
    (run/'history_feed.pid').write_text(str(os.getpid())+'\n')
    feed=HistoryFeed(run);scheduler=ReportScheduler(run);stopped=False;generated=False;last_report=None
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopped:
        alive=Path(f'/proc/{args.training_pid}').exists()
        try:
            data=feed.snapshot(alive)
            if not alive and not generated:
                config=json.loads((run/'config.json').read_text())
                path=run/'run_status.json'
                status=json.loads(path.read_text()) if path.exists() else {}
                status.update(phase=data['phase'],steps=data['steps'],total_steps=data['total_steps'])
                atomic_json(path,status)
                generated=True
            scheduler.tick(data,alive)
            last_report=publish(run,args.connection_dir,data,last_report)
            atomic_json(run/'history_health.json',dict(status='connected',updated_at=time.time(),steps=data['steps'],episode_mode=data['episode_mode']))
            if not alive and (not config.get('report',True) or scheduler.final_ready(data)):break
        except Exception as ex:
            atomic_json(run/'history_health.json',dict(status='retrying',error=type(ex).__name__,updated_at=time.time()))
        if args.once:break
        for _ in range(25):
            if stopped:break
            time.sleep(.2)


if __name__=='__main__':main()
