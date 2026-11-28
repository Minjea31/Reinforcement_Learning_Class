"""Save per-algorithm scientific figures and source CSVs without invented data."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
import numpy as np

from training.history import atomic_json


FIGURES=('training_overview','reward_curve','success_rate','effort_curve','ppo_diagnostics','learning_speed')


def rows(path):
    if not path.exists():return []
    with path.open() as stream:return list(csv.DictReader(stream))


def series(records,key):
    points=[]
    for row in records:
        try:points.append((float(row['steps'])/1e6,float(row[key])))
        except (KeyError,ValueError,TypeError):pass
    return np.array(points).reshape(-1,2)


def curve(axis,records,key,label):
    data=series(records,key)
    if len(data):axis.plot(data[:,0],data[:,1],label=label,linewidth=1.)
    axis.grid(alpha=.25);axis.set_xlabel('Environment steps (millions)');axis.set_title(label)
    if len(data):axis.legend(fontsize=7)
    else:axis.text(.5,.5,'No recorded data yet',ha='center',va='center',transform=axis.transAxes)


def generate(run,live=False):
    run=Path(run).resolve();config=json.loads((run/'config.json').read_text())
    status=json.loads((run/'run_status.json').read_text()) if (run/'run_status.json').exists() else dict(phase='training',steps=0)
    metrics=rows(run/'metrics.csv');episodes=rows(run/'episodes.csv')
    validation=[]
    for path in sorted(run.glob('validation_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
        value=json.loads(path.read_text());validation.append(dict(value,steps=value.get('steps',int(path.stem.split('_')[-1]))))
    latest=int(status.get('steps',0));algorithm=config['algorithm'].upper();generation_time=datetime.now(timezone.utc).isoformat()
    # Publish summaries only after the full batch exists. The history uploader
    # verifies file hashes, so it never uploads a mixed generation.
    for folder,stage in [(run/'reports',None),(run/f'reports/stage{config["stage"]}',config['stage'])]:
        figures=folder/'figures';figures.mkdir(parents=True,exist_ok=True)
        for name in FIGURES:
            if name=='training_overview':
                fig,axes=plt.subplots(2,2,figsize=(11,6));plots=[(episodes,'reward','Episode return'),(validation,'success_rate','Validation success'),(metrics,'mean_effort','Physical effort'),(metrics,'fps','Training FPS')]
                for axis,(data,key,label) in zip(axes.flat,plots):curve(axis,data,key,label)
            elif name=='ppo_diagnostics':
                fig,axes=plt.subplots(1,2,figsize=(11,3))
                keys=('actor_loss','critic_loss') if config['algorithm']=='sac' else ('policy_loss','value_loss')
                for axis,key in zip(axes,keys):curve(axis,metrics,key,key.replace('_',' ').title())
            else:
                fig,axis=plt.subplots(figsize=(10,3.5))
                data,key,label={'reward_curve':(episodes,'reward','Episode total return'),'success_rate':(validation,'success_rate','Held-out validation success'),
                    'effort_curve':(episodes,'effort','Motor and gripper effort'),'learning_speed':(metrics,'fps','Training throughput')}[name]
                curve(axis,data,key,label)
            fig.suptitle(algorithm);fig.tight_layout()
            for ext in ('png','svg'):
                temporary=figures/(name+'.partial.'+ext);fig.savefig(temporary,dpi=300);os.replace(temporary,figures/(name+'.'+ext))
            plt.close(fig)
        for source,target in [('metrics.csv','training_metrics.csv'),('episodes.csv','episodes.csv')]:
            if (run/source).exists():shutil.copyfile(run/source,folder/target)
        with (folder/'validation.csv').open('w',newline='') as stream:
            fields=['steps','success_rate','grasp_rate','lift_rate','mean_return'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for value in validation:writer.writerow({key:value.get(key) for key in fields})
        hashes={name+'.png':hashlib.sha256((figures/(name+'.png')).read_bytes()).hexdigest() for name in FIGURES}
        summary=dict(run_name=config['web_run_name'],algorithm=config['algorithm'],stage=stage,last_steps=latest,status=status['phase'],live=live,
            generated_at=generation_time,figures=list(hashes),figure_hashes=hashes,generation=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            payload_mass=config['payload_mass'],task_version=2,layout_mode=config['layout_mode'])
        atomic_json(folder/'summary.json',summary)
        reward={'reviewed_v4':'수정 보상 v4','reviewed_v3':'새 보상 v3'}.get(config.get('reward_version'),'기존 보상 v2')
        (folder/'presentation_notes.md').write_text(f'# {algorithm} 학습 자료\n\n- 환경 1개, 물체 {config["payload_mass"]}kg, 관측 105개·연속 행동 14개\n- 단계 {config["stage"]}, 난도 {config["level"]}/4, {reward}\n- 누적 {latest:,}스텝, 상태 {status["phase"]}\n- figures: 300dpi PNG·SVG, 원본 CSV 동봉\n- ../mission_replays: 평가 50회 중 상위 10개 영상·시작·최대 높이·종료 장면·보상 시계열\n- ../recordings: 설정한 주기별 평가 영상\n- 다른 모델의 승급과 독립적이다. 서로 다른 난도의 총보상이나 성공률은 같은 조건의 직접 비교가 아니다.\n- 보상만으로 성공을 판단하지 말고 검증 성공률·실제 잡기·들기·안착을 함께 확인한다.\n')
    atomic_json(run/'reports/index.json',dict(steps=latest,status=status['phase'],live=live,algorithm=config['algorithm'],generated_at=generation_time))
