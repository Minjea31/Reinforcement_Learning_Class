"""Register a comparison experiment and separate authenticated video tunnels."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
from urllib import request

from web.live_access import read_credentials
from web.reconnect import NoRedirect, start_tunnel, tunnel_pid


def study_identity(root):
    root=Path(root).resolve()
    label=re.sub(r'[^A-Za-z0-9_-]','_',root.name).strip('_-')[:55] or 'comparison'
    return label+'-'+hashlib.sha256(str(root).encode()).hexdigest()[:8]


def request_json(credentials,path,data,method='POST'):
    req=request.Request(credentials['origin']+path,data=None if data is None else json.dumps(data).encode(),method=method,
        headers={'OAI-Sites-Authorization':'Bearer '+credentials['site_token'],'Content-Type':'application/json'})
    with request.build_opener(NoRedirect()).open(req,timeout=30) as response:
        return json.load(response)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def register_study(root,study_id,algorithms,credentials,stage,total_steps):
    entries=[dict(name=name,run_name=f'{study_id}_{name}',
                  stage=stage[name] if isinstance(stage,dict) else stage,
                  total_steps=total_steps) for name in algorithms]
    name = Path(root).name
    labels = {'three_models_1kg_v3': 'three_models_1kg',
              'three_models_1kg_masked': '한 팔 마스크 · 단기 시험',
              'three_models_1kg_masked_full': '한 팔 마스크 · 장시간 학습'}
    response = request_json(credentials,'/api/studies',dict(id=study_id,label=labels.get(name, name),algorithms=entries))
    if not response.get('ok') or response.get('id') != study_id:
        raise RuntimeError('The Site did not register this study')
    return entries


def connect_study(root,study_id,algorithms,connection_dir,stage,total_steps,active=None):
    credentials=read_credentials(connection_dir)
    entries=register_study(root,study_id,algorithms,credentials,stage,total_steps)
    connections={}
    try:
        for entry in entries:
            if active is not None and entry['name'] not in active:continue
            folder=Path(connection_dir).resolve()/'studies'/study_id/entry['name'];folder.mkdir(parents=True,exist_ok=True)
            port_file=folder/'stream_port'
            port=int(port_file.read_text()) if port_file.exists() else free_port()
            if tunnel_pid(folder,port) is None:
                port=free_port();port_file.write_text(str(port)+'\n')
            origin,pid=start_tunnel(folder,port=port)
            connections[entry['name']]=dict(port=port,pid=pid,folder=folder)
            request_json(credentials,'/api/live-connection?run='+entry['run_name'],
                         dict(origin=origin,token=credentials['stream_token']),'PUT')
    except BaseException:
        close_connections(connections);raise
    return connections


def close_connections(connections):
    for item in connections.values():
        if tunnel_pid(item['folder'],item['port'])==item['pid']:
            try:os.kill(item['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
