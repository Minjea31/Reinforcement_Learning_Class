"""Publish each algorithm's own frozen best-model top ten, then finish cleanly."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

os.environ.setdefault('LP_NUM_THREADS','2')

from evaluation.export_algorithm_replay import bundle_digest,export
from web.mission_replays import publish
from training.history import atomic_json


def prepare(run, source=None):
    run=Path(run);source=source or run/'best';digest=bundle_digest(source)
    metadata=json.loads((source/'metadata.json').read_text());stage=metadata['config']['stage']
    folder=run/'mission_replays'/f'stage-{stage}'/digest;frozen=folder/'checkpoint'
    if not frozen.exists():
        temporary=folder/'checkpoint.partial';temporary.mkdir(parents=True,exist_ok=True)
        for name in ('model.zip','normalization.pkl','metadata.json'):shutil.copyfile(source/name,temporary/name)
        if bundle_digest(temporary)!=digest:shutil.rmtree(temporary);raise RuntimeError('Best policy changed during snapshot')
        os.replace(temporary,frozen)
    assets=folder/'assets'
    if (assets/'replays.json').exists():return assets,json.loads((assets/'replays.json').read_text())
    return export(frozen,assets,top_k=10,episodes=50)


def mission_sources(run):
    sources = {}
    current = run/'best'
    if (current/'metadata.json').exists():
        stage = json.loads((current/'metadata.json').read_text())['config']['stage']
        sources[str(stage)] = current
    for archive in sorted((run/'stage_checkpoints').glob('stage-*')):
        if (archive/'metadata.json').exists() and not archive.name.endswith('.partial'):
            sources.setdefault(archive.name.rsplit('-',1)[-1], archive)
    return sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--connection-dir',required=True)
    p.add_argument('--training-pid',required=True,type=int);p.add_argument('--once',action='store_true');args=p.parse_args()
    try:os.nice(10)
    except OSError:pass
    run=Path(args.run_dir);state_path=run/'mission_replays/published.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    stages = state.get('stages', {})
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopped:
        alive=Path(f'/proc/{args.training_pid}').exists()
        try:
            sources = mission_sources(run)
            for stage, source in sources.items():
                digest=bundle_digest(source)
                old=stages.get(stage, state if state.get('hash')==digest else {})
                if digest==old.get('hash') or (alive and time.time()-old.get('published_at',0)<=300):
                    continue
                output,manifest=prepare(run, source)
                if stopped:break
                stage=str(manifest['stage'])
                if source.name=='best' and stage in stages and stages[stage]['hash']==manifest['checkpoint_sha256']:
                    continue
                publish(run,output,manifest,args.connection_dir)
                state=dict(hash=manifest['checkpoint_sha256'],published_at=time.time(),steps=manifest['checkpoint_steps'])
                stages[stage]=state
                atomic_json(state_path,dict(state,stages=stages))
            if not alive and all(bundle_digest(source)==stages.get(stage,state).get('hash') for stage,source in sources.items()):
                break
        except Exception as exc:
            atomic_json(run/'replay_health.json',dict(status='retrying',error=type(exc).__name__,updated_at=time.time()))
            if args.once:raise
        if args.once:break
        for _ in range(100):
            if stopped:break
            time.sleep(.2)


if __name__=='__main__':main()
