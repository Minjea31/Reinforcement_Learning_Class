"""Export full learning history as PNG/SVG figures, CSV data and PPT notes."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from training.history import atomic_json


def read_rows(run,name):
    result=[]
    for path in sorted(Path(run).glob(name)):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    if not row.get('steps'):continue
                    row={k:(v if k in ('termination','run_name') else float(v) if v not in ('',None) else None) for k,v in row.items() if k!='time_utc'}
                    if not all(not isinstance(value,(int,float)) or math.isfinite(value) for value in row.values()):continue
                except (ValueError,TypeError):continue
                result.append(row)
    return result


def lineage(run,cutoff=float('inf'),seen=None):
    run=Path(run).resolve();seen=set() if seen is None else seen
    if run in seen:return []
    seen.add(run);config_path=run/'config.json';config=json.loads(config_path.read_text()) if config_path.exists() else {}
    meta_path=run/'artifacts/run_metadata.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    parent=meta.get('parent_run');bound=meta.get('parent_cutoff_steps')
    if not parent and config.get('resume'):
        checkpoint=Path(config['resume'])
        if not checkpoint.is_absolute():checkpoint=Path(__file__).resolve().parents[1]/checkpoint
        if checkpoint.exists() and checkpoint.parent.resolve()!=run:
            import torch
            transfer=run/'transfer_initial.pt'
            source=torch.load(transfer if transfer.exists() else checkpoint,map_location='cpu',weights_only=False)
            parent=str(checkpoint.parent.resolve());bound=source['steps']
    ancestors=lineage(parent,min(cutoff,bound),seen) if parent and bound is not None else []
    return ancestors+[(run,cutoff,config)]


def collect(run,stage=None):
    metrics=[];episodes=[];validations=[];sources=[]
    for directory,cutoff,config in lineage(run):
        sources.append(dict(run_name=directory.name,cutoff_steps=None if math.isinf(cutoff) else cutoff))
        unique={int(row['steps']):row for row in read_rows(directory,'metrics*.csv') if row['steps']<=cutoff and (stage is None or row.get('stage')==stage)}
        for row in unique.values():metrics.append(dict(row,run_name=directory.name,task_version=config.get('task_version') or 1,layout_mode=config.get('layout_mode') or 'fixed',payload_mass=config.get('payload_mass',1.)))
        for row in read_rows(directory,'episodes.csv'):
            if row['steps']<=cutoff and (stage is None or row.get('stage')==stage):episodes.append(dict(row,run_name=directory.name))
        for path in directory.glob('validation_*.json'):
            steps=int(path.stem.split('_')[1])
            if steps>cutoff:continue
            try:data=json.loads(path.read_text())
            except json.JSONDecodeError:continue  # A validation may still be writing.
            if stage is not None and data.get('env_config',{}).get('stage')!=stage:continue
            validations.append(dict(steps=steps,stage=data.get('env_config',{}).get('stage'),run_name=directory.name,**{k:data.get(k) for k in ('success_rate','grasp_rate','lift_rate','mean_return','mean_goal_progress')}))
    for rows in (metrics,episodes,validations):rows.sort(key=lambda row:row['steps'])
    return dict(metrics=metrics,episodes=episodes,validations=validations,sources=sources)


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row)) or ['steps']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def collect_materials(run,report,stage):
    """Index existing evaluation assets without copying models or credentials."""
    from PIL import Image
    materials=report/'materials';materials.mkdir(exist_ok=True)
    entries=[]
    for path in sorted((run/'recordings').glob('*/metadata.json')):
        data=json.loads(path.read_text())
        if stage is not None and (data.get('env_config') or {}).get('stage')!=stage:continue
        folder=path.parent
        entries.append(dict(kind='checkpoint_evaluation',checkpoint_steps=data.get('checkpoint_steps'),
                            video=str((folder/'video.mp4').relative_to(run)),metadata=str(path.relative_to(run))))
        for episode in data.get('episodes',[]):
            for name in episode.get('images',[]):
                source=folder/name
                if source.parent!=folder or source.suffix!='.png' or not source.exists():continue
                import shutil
                target=materials/f'{folder.name}_{source.name}';shutil.copy2(source,target)
                entries.append(dict(kind='evaluation_scene',image=str(target.relative_to(report)),source=str(source.relative_to(run)),seed=episode.get('seed')))
        video=folder/'video.mp4'
        if video.exists():
            # Old recordings predate screenshot export; extract real video frames.
            import subprocess
            for label,fraction in (('start',0.),('middle',.5),('end',.98)):
                target=materials/f'{folder.name}_{label}.png'
                seek=max(0.,float(data.get('video_seconds',0))*fraction)
                if not target.exists():
                    result=subprocess.run(['ffmpeg','-v','error','-ss',str(seek),'-i',str(video),'-frames:v','1','-threads','1','-y',str(target)],capture_output=True)
                    if result.returncode:continue
                if target.exists():
                    with Image.open(target) as image:image.verify()
                    entries.append(dict(kind='saved_video_frame',image=str(target.relative_to(report)),video=str(video.relative_to(run)),offset_seconds=seek))
    stages=[stage] if stage else range(1,5)
    for number in stages:
        for path in sorted((run/'mission_replays'/f'stage-{number}').glob('*/assets/**/poster.jpg')):
            target=materials/f'stage{number}_{path.parents[2].name[:10]}_{path.parent.name}.jpg'
            import shutil
            shutil.copy2(path,target)
            entries.append(dict(kind='evaluation_poster',image=str(target.relative_to(report)),source=str(path.relative_to(run))))
    atomic_json(report/'materials.json',entries)
    return entries


def generate(run,stage=None,*,live=False,dataset=None,report_phase=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run).resolve();report=run/'reports'
    if stage is not None:report=report/f'stage{stage}'
    figures=report/'figures';figures.mkdir(parents=True,exist_ok=True)
    lock=(report/'.generation.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();return None
    try:
        data=collect(run,stage) if dataset is None else dataset
        metrics=data['metrics'];episodes=data['episodes'];validations=data['validations']
        if stage is not None and not (metrics or episodes or validations):return None
        write_csv(report/'training_metrics.csv',metrics);write_csv(report/'episodes.csv',episodes);write_csv(report/'validation.csv',validations)
        plt.rcParams.update({'figure.dpi':120,'savefig.dpi':300,'svg.fonttype':'none','font.size':10,'axes.grid':True,'grid.alpha':.2})
        def series(rows,key):return [(row['steps']/1e6,float(row[key])) for row in rows if row.get(key) is not None]
        def curve(ax,rows,key,label,color=None,window=1):
            # Do not smooth across different tasks or reward/layout revisions.
            groups={}
            for row in rows:
                group=(row.get('run_name'),row.get('stage'),row.get('task_version'),row.get('layout_mode'),row.get('payload_mass'))
                groups.setdefault(group,[]).append(row)
            shown=False
            for group,values in groups.items():
                points=series(values,key)
                if not points:continue
                x,y=np.asarray(points).T
                if window>1:y=np.array([np.mean(y[max(0,i-window+1):i+1]) for i in range(len(y))])
                stride=max(1,len(x)//5000)
                line=ax.plot(x[::stride],y[::stride],label=label if not shown else '_nolegend_',color=color,linewidth=1.3)
                color=line[0].get_color();shown=True
        def finish(fig,name):
            for ax in fig.axes:
                ax.set_xlabel('Environment steps (millions)')
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(figures/f'{name}.png');fig.savefig(figures/f'{name}.svg');plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        curve(axes[0,0],metrics,'mean_return','Completed-episode return: update mean',window=25)
        curve(axes[0,0],episodes,'reward','Episode return: rolling 25',window=25)
        axes[0,0].set_title('Episode return');axes[0,0].set_ylabel('Return')
        curve(axes[0,1],metrics,'training_success_rate','Training: update mean',window=25)
        curve(axes[0,1],validations,'success_rate','Validation')
        axes[0,1].set_title('Task success');axes[0,1].set_ylim(-.05,1.05)
        curve(axes[1,0],metrics,'grasp_rate','Grasp',window=25);curve(axes[1,0],metrics,'lift_rate','Lift',window=25)
        axes[1,0].set_title('Grasp and lift');axes[1,0].set_ylim(-.05,1.05)
        curve(axes[1,1],metrics,'mean_effort','Episode effort: update mean',window=25)
        axes[1,1].set_title('Physical effort')
        finish(fig,'training_overview')
        fig,ax=plt.subplots(figsize=(11,4));curve(ax,metrics,'mean_return','Update mean',window=25);curve(ax,episodes,'reward','Episode return (rolling 25)',window=25)
        ax.set_title('Episode total return');ax.set_ylabel('Return');finish(fig,'reward_curve')
        fig,ax=plt.subplots(figsize=(11,4));curve(ax,metrics,'training_success_rate','Training',window=25);curve(ax,validations,'success_rate','Validation')
        ax.set_title('Training and validation success');ax.set_ylim(-.05,1.05);finish(fig,'success_rate')
        fig,ax=plt.subplots(figsize=(11,4));curve(ax,metrics,'mean_effort','Update mean',window=25);curve(ax,episodes,'effort','Episode effort (rolling 25)',window=25)
        ax.set_title('Motor and gripper effort');finish(fig,'effort_curve')
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for ax,key,title in zip(axes.flat,('policy_loss','value_loss','entropy','kl'),('Policy loss','Value loss','Policy entropy','Approximate KL')):
            curve(ax,metrics,key,title,window=25);ax.set_title(title)
        finish(fig,'ppo_diagnostics')
        fig,ax=plt.subplots(figsize=(11,4));curve(ax,metrics,'fps','Reported aggregate FPS');ax.set_title('Learning throughput');ax.set_ylabel('Environment steps / second');finish(fig,'learning_speed')
        config=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
        status=json.loads((run/'run_status.json').read_text()) if (run/'run_status.json').exists() else {}
        final_root=run
        # Inherited stage records retain the original stage's evaluation/provenance.
        if stage and metrics and metrics[-1].get('run_name')!=run.name:
            for source,cutoff,_ in lineage(run):
                checkpoint=source/f'stage{stage}_final.pt'
                if source.name==metrics[-1]['run_name'] and checkpoint.exists():
                    import torch
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if saved['steps']<=cutoff and saved['steps']==metrics[-1]['steps']:final_root=source
        final_path=final_root/f'stage{stage}_final_evaluation.json' if stage else None
        final=json.loads(final_path.read_text()) if final_path and final_path.exists() else None
        if live:
            material_path=report/'materials.json'
            materials=json.loads(material_path.read_text()) if material_path.exists() else []
        else:materials=collect_materials(run,report,stage)
        profile=metrics[-1] if stage and metrics else config
        summary=dict(run_name=run.name,stage=stage,status='stage_completed' if final else report_phase or status.get('phase','snapshot'),generated_at=datetime.now(timezone.utc).isoformat(),live=live,
            last_steps=max((int(row['steps']) for row in metrics),default=status.get('steps',0)),
            total_steps=config.get('total_steps'),completed_episode_records=len(episodes),update_records=len(metrics),validation_records=len(validations),
            sources=data['sources'],task_version=profile.get('task_version') or 1,layout_mode=profile.get('layout_mode') or 'fixed',
            figures=[f'{name}.png' for name in ('training_overview','reward_curve','success_rate','effort_curve','ppo_diagnostics','learning_speed')],
            latest_validation=validations[-1] if validations else None,final_evaluation={k:v for k,v in final.items() if k!='records'} if final else None,
            material_count=len(materials),payload_mass=profile.get('payload_mass',1.))
        if final:
            summary['total_steps']=summary['last_steps']
            write_csv(report/'final_evaluation_episodes.csv',final.get('records',[]))
            atomic_json(report/'final_evaluation.json',final)
        checkpoint=final_root/f'stage{stage}_final.pt' if stage else None
        if checkpoint and checkpoint.exists() and not live:
            import torch
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            atomic_json(report/'model_metadata.json',dict(checkpoint=checkpoint.name,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                steps=saved['steps'],updates=saved['updates'],obs_dim=saved['obs_dim'],env_config=saved.get('env_config'),config=saved['config'],
                provenance='Metadata from the original saved model; historical source code was not archived at training time.'))
        summary['figure_hashes']={name:hashlib.sha256((figures/name).read_bytes()).hexdigest() for name in summary['figures']}
        summary['generation']=hashlib.sha256(json.dumps(summary['figure_hashes'],sort_keys=True).encode()).hexdigest()
        atomic_json(report/'summary.json',summary)
        notes=f'''# 학습 발표 자료\n\n- 실행: `{run.name}`\n- 기록 상태: `{summary['status']}`\n- 마지막 누적 스텝: {summary['last_steps']:,}\n- PPO / PyTorch / 관측 105개·행동 14개 / {summary["payload_mass"]:g}kg 물체 / 두 팔 로봇\n- 보상 버전: {summary['task_version']}, 배치: `{summary['layout_mode']}`\n\n## 자료 위치\n\n- `figures/`: 전체 학습 흐름과 보상·성공률·힘 사용·PPO 진단·속도 그래프. PNG 300dpi, SVG 벡터를 함께 저장.\n- `training_metrics.csv`, `episodes.csv`, `validation.csv`: 그래프의 원본 데이터.\n- `../artifacts/`: 설정·라이브러리 버전·시작 시점의 소스 코드와 해시.\n- `../recordings/`: 500만 스텝마다 저장한 평가 영상과 시작·종료·최대 상승 장면.\n- `../mission_replays/`: 검증 상위 시나리오 영상·미리보기·보상 시계열.\n\n## 발표 시 해석\n\n초기 1단계부터 이어온 부모 실행의 로그는 이어받은 체크포인트 스텝까지만 포함한다. 중단한 다른 학습 분기는 포함하지 않는다. 과거 실행에 에피소드 원본 로그가 없으면 업데이트별 에피소드 평균 보상을 사용하며 개별 에피소드 값으로 표현하지 않는다. 서로 다른 단계·보상·배치의 점수는 직접 비교하지 말고 성공률과 실제 움직임을 함께 확인한다. 짧은 테스트나 중단 실행은 전체 학습 완료를 의미하지 않는다.\n\n## 슬라이드 구성 예시\n\n1. 문제와 환경: 두 팔 로봇·시작점·목표점 이미지\n2. 방법: PPO 모델과 단계별 보상·배치 구성\n3. 학습 과정: training_overview와 reward_curve\n4. 결과: success_rate, effort_curve와 평가 영상의 대표 장면\n5. 한계와 추가 연구: 실패 사례, 여러 시뮬레이션 및 신경망 구조 비교\n'''
        notes+='\n## 장면 자료\n\n`materials/`에 기존 평가 영상에서 뽑은 시작·중간·마지막 장면과 상위 정책의 미리보기 이미지를 저장한다. `materials.json`에는 원본 영상·모델 스텝·추출 위치를 기록한다. 영상 추출 장면은 최대 상승 장면을 의미하지 않는다. 새 녹화부터는 `recordings/`에 시작·종료·최대 상승 PNG를 직접 저장한다.\n'
        if stage:
            notes+=f'\n## {stage}단계 자료의 범위\n\n학습·검증 그래프는 stage={stage} 기록만 포함한다. 이후 단계의 로그는 포함하지 않는다.\n'
        if final:
            notes+=f'\n종료 평가: {final["episodes"]}회, 성공률 {final["success_rate"]:.1%}, 평균 총보상 {final["mean_return"]:.3f}. 개별 평가 결과는 `final_evaluation_episodes.csv`에 보존했다. 이는 학습 에피소드의 원본 로그와 구분한다. 1단계는 고정된 좌우 두 배치를 평가했으므로 서로 다른 랜덤 환경 {final["episodes"]}개에 대한 일반화 성능을 뜻하지 않는다.\n'
        if stage is not None:
            for name in ('artifacts','recordings','mission_replays'):notes=notes.replace(f'`../{name}/`',f'`../../{name}/`')
        if not (run/'artifacts/run_metadata.json').exists():
            notes+='\n## 과거 기록의 한계\n\n이 실행을 시작할 때 소스 스냅샷을 저장하지 않았다. 현재 소스를 당시 실행 코드라고 기록하지 않는다. 개별 학습 에피소드 CSV가 없는 구간은 당시의 업데이트 구간 평균 로그로 복원했다. 종료 평가의 개별 결과는 학습 에피소드 원본과 구분한다.\n'
        (report/'ppt_notes.md').write_text(notes)
        index=['# 발표용 장면','', '저장된 평가 영상의 추출 장면과 정책 평가 이미지입니다. 출처는 materials.json에 기록했습니다.','']
        for entry in materials:
            if 'image' in entry:index.extend([f'## {Path(entry["image"]).stem}','',f'![{entry["kind"]}]({entry["image"]})',''])
        (report/'scene_index.md').write_text('\n'.join(index))
        print(f'Saved training figures and PPT materials: {report}',flush=True)
        return summary
    finally:lock.close()


def generate_all(run,live=False):
    """Refresh visited stages only; reuse unchanged completed-stage figures."""
    run=Path(run).resolve();folder=run/'reports';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.batch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
        data=collect(run)
        stages=sorted({int(row['stage']) for name in ('metrics','episodes','validations') for row in data[name] if row.get('stage') in (1,2,3,4)})
        status_path=run/'run_status.json'
        phase=json.loads(status_path.read_text()).get('phase','snapshot') if status_path.exists() else 'snapshot'
        reports={}
        for stage in [None,*stages]:
            subset=data if stage is None else {name:[row for row in data[name] if row.get('stage')==stage] for name in ('metrics','episodes','validations')}|{'sources':data['sources']}
            report_phase='stage_completed' if stage is not None and stage<max(stages) else phase
            fingerprint=hashlib.sha256(json.dumps(dict(phase=report_phase,live=live,sources=data['sources'],
                tails={name:dict(count=len(subset[name]),last=subset[name][-1] if subset[name] else None) for name in ('metrics','episodes','validations')}),sort_keys=True).encode()).hexdigest()
            report_dir=folder if stage is None else folder/f'stage{stage}'
            path=report_dir/'summary.json'
            old=json.loads(path.read_text()) if path.exists() else {}
            if old.get('input_fingerprint')==fingerprint:result=old
            else:
                result=generate(run,stage,live=live,dataset=subset,report_phase=report_phase)
                if result is not None:
                    result['input_fingerprint']=fingerprint;atomic_json(path,result)
            if result is not None:reports['overall' if stage is None else str(stage)]=result
        index=dict(run_name=run.name,status=phase,live=live,steps=max((int(row['steps']) for row in data['metrics']),default=0),
                   generated_at=datetime.now(timezone.utc).isoformat(),stages=stages)
        atomic_json(folder/'index.json',index)
        return reports


def main():
    import sys
    # Keep the native report pipeline intact; SB3 stores different model and
    # diagnostic formats and has its own report generator.
    if '--run-dir' in sys.argv:
        candidate=Path(sys.argv[sys.argv.index('--run-dir')+1])/'config.json'
        if candidate.exists() and json.loads(candidate.read_text()).get('framework')=='sb3':
            from evaluation.study_report import generate
            generate(candidate.parent,live='--live' in sys.argv)
            return
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);parser.add_argument('--stage',type=int,choices=(1,2,3,4))
    parser.add_argument('--live',action='store_true',help='Refresh plots during training; defer video frame extraction until exit')
    args=parser.parse_args()
    try:os.nice(10)
    except OSError:pass
    if args.stage:generate(args.run_dir,args.stage,live=args.live)
    else:generate_all(args.run_dir,live=args.live)


if __name__=='__main__':main()
