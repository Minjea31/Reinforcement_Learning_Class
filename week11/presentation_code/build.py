from pathlib import Path
import sys, json, copy
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.enum.shapes import MSO_CONNECTOR
import pymupdf

R = Path("/home/a/ppt_workspace")
D = R / "ppt_info/RL_week11"
O = R / "output/RL_week11"
C = Path("/home/a/Reinforcement_Learning_class/week11/assignment")
sys.path.insert(0, str(C))
from evaluate import SavedPolicy

FIG = D / "figures"
FIG.mkdir(exist_ok=True)
p = Presentation(R / "templet/base.pptx")
original = list(p.slides)
for sid in list(p.slides._sldIdLst):
    p.part.drop_rel(sid.rId)
    p.slides._sldIdLst.remove(sid)


def text(sh, value):
    tf = sh.text_frame
    old = tf.paragraphs[0]
    size = next((r.font.size for r in old.runs if r.font.size), Pt(18))
    color = next(
        (r.font.color.rgb for r in old.runs if r.font.color.type == 1),
        RGBColor(8, 8, 8),
    )
    align = old.alignment
    tf.clear()
    lines = value.split("\n")
    for i, line in enumerate(lines):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = 1.3
        par.space_after = Pt(
            (6 if i + 1 < len(lines) and not lines[i + 1].startswith("• ") else 14)
            if sh.name in ("Body", "PanelBody")
            else 6
        )
        r = par.add_run()
        if sh.name in ("Body", "PanelBody"):
            bullet = line.startswith("• ")
            line = line[2:] if bullet else line
            pp = par._p.get_or_add_pPr()
            pp.set("marL", str(int(Pt(18))))
            pp.set("indent", str(int(Pt(-18 if bullet else 0))))
            for node in list(pp):
                if node.tag.rsplit("}", 1)[-1] in (
                    "buChar",
                    "buAutoNum",
                    "buNone",
                    "buFont",
                ):
                    pp.remove(node)
            bu = OxmlElement("a:buChar" if bullet else "a:buNone")
            if bullet:
                bu.set("char", "•")
            pp.append(bu)
        r.text = line
        r.font.name = "Pretendard"
        r.font.bold = True
        r.font.size = size
        r.font.color.rgb = color
    return sh


def setv(s, name, v):
    return text(next(sh for sh in s.shapes if sh.name == name), v)


def slide(
    idx,
    title=None,
    source=None,
):
    s = p.slides.add_slide(p.slide_layouts[6])
    for sh in original[idx].shapes:
        s.shapes._spTree.insert_element_before(copy.deepcopy(sh.element), "p:extLst")
    for sh in s.shapes:
        if sh.name == "Title" and title:
            text(sh, title)
        if sh.name == "Page":
            text(sh, f"{len(p.slides):02}")
    if idx != 0 and source:
        sh = s.shapes.add_textbox(Pt(130), Pt(514), Pt(710), Pt(12))
        sh.text = source
        for par in sh.text_frame.paragraphs:
            par.line_spacing = 1.3
            for r in par.runs:
                r.font.name = "Pretendard"
                r.font.size = Pt(8)
    return s


def body(title, lines):
    s = slide(3, title)
    setv(s, "Body", "\n".join("• " + x for x in lines))
    return s


def image(s, name, file):
    from PIL import Image

    sh = next(x for x in s.shapes if x.name == name)
    x, y, w, h = sh.left, sh.top, sh.width, sh.height
    iw, ih = Image.open(file).size
    k = min(w / iw, h / ih)
    sh.element.getparent().remove(sh.element)
    s.shapes.add_picture(
        str(file),
        int(x + (w - iw * k) / 2),
        int(y + (h - ih * k) / 2),
        width=int(iw * k),
        height=int(ih * k),
    )


def panel(title, file, heading, lines):
    s = slide(4, title)
    image(s, "ImageSlot", file)
    setv(s, "PanelTitle", heading)
    setv(s, "PanelBody", "\n".join("• " + x for x in lines))
    return s


def code(title, source, lines):
    s = slide(5, title)
    setv(s, "CodeBody", source)
    setv(s, "PanelTitle", "코드의 동작과 설계")
    setv(s, "PanelBody", "\n".join("• " + x for x in lines))


def savefig(fig, name):
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    for ax in fig.axes:
        if ax.get_ylabel() in ['Original total reward (higher is better)', 'Validation total reward', 'Velocity', 'Terrain height (illustration)']:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if ax.get_xlabel() in ['Position', 'Velocity', 'Held-out total reward']:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.xaxis.set_major_locator(MaxNLocator(4))
    path = FIG / (name + ".png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
import csv
from matplotlib.font_manager import FontProperties,fontManager
fontManager.addfont(str(R/'templet/fonts/Pretendard-Bold.ttf'));font=FontProperties(fname=str(R/'templet/fonts/Pretendard-Bold.ttf'))
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.weight':'bold','font.size':12})
S=json.loads((C/'results/summary.json').read_text());E=json.loads((C/'results/independent_evaluation.json').read_text())
BASE=S['base'];GROUPS=S['sweeps'];RECORDS=[]
import hashlib
PARAMETER_ANALYSIS=json.loads((D/'parameter_analysis.json').read_text())
assert PARAMETER_ANALYSIS['summary_sha256']==hashlib.sha256((C/'results/summary.json').read_bytes()).hexdigest()
NAMES=dict(gamma='할인율 γ',actor_lr='Actor 학습률',critic_lr='Critic 학습률',width='은닉 노드 수',depth='은닉층 수',beta1='Adam 1차 모멘트 β₁',velocity_scale='속도 입력 스케일',episodes='학습 에피소드 수')
MEANINGS=dict(gamma='미래의 보상을 현재 가치에 반영하는 비율',actor_lr='행동 확률을 갱신하는 크기',critic_lr='1-step TD 목표에 맞춰 가치를 갱신하는 크기',width='각 은닉층의 노드 수 · 표현 용량',depth='ReLU 은닉층을 쌓는 개수',beta1='Adam이 이전 기울기의 방향을 반영하는 비율',velocity_scale='입력 속도에 곱하는 값 · 위치와 수치 크기 조절',episodes='환경을 reset하여 학습한 전체 에피소드 수')
def fmt(x):return f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)
def val(name,x):return f'{x*1000:.3f} ×10⁻³' if name.endswith('_lr') else fmt(x)
def result(c):return next(r for r in S['results'] if r['config']==c)
def metric(q):return f"평균 {q['mean']:.3f} · 도달률 {q['success']*100:.3f}%"
def winnertext(c):
 changes=[NAMES[k]+' '+val(k,v) for k,v in c.items() if k in NAMES and v!=BASE[k]]
 return ' / '.join(changes) if changes else '기준 설정'
with (C/'results/comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['study','parameter','value','mean','std','success'])
 for name,vs in GROUPS.items():
  for v in vs:
   q=result(dict(BASE,**{name:v}))['validation'];w.writerow(['basic',name,val(name,v),f"{q['mean']:.3f}",f"{q['std']:.3f}",f"{q['success']:.3f}"])
 for name,vs in S['refinement_sweeps'].items():
  for v in vs:
   q=result(dict(S['refinement_base'],**{name:v}))['validation'];w.writerow(['additional',name,val(name,v),f"{q['mean']:.3f}",f"{q['std']:.3f}",f"{q['success']:.3f}"])
envfig=D/'native_gym_preview.png'
s=slide(0);setv(s,'CoverTitle','강화학습 week11');setv(s,'Author','Quiz1 · OpenAI Gym MountainCar · 1-step Actor-Critic')
body('과제와 실험 조건',['OpenAI Gym MountainCar에 강의의 1-step Actor-Critic 적용','학습 시드 0으로 고정 · 모든 후보에 동일하게 적용','기본·추가 탐색에서 각각 한 조건만 변경 · 우수값 조합 재학습'])
panel('MountainCar · 목표와 보상',envfig,'환경 설정',['상태: 위치 x · 속도 v\n행동: 왼쪽 / 유지 / 오른쪽','매 스텝 −1.000 · 최대 200스텝\n빠르게 도달할수록 총보상이 큼','위치 0.500 이상 · 속도 0.000 이상\n목표 도달 또는 시간 제한에서 종료'])
code('코드 · OpenAI Gym 환경 사용',"# experiment.py / train\nimport gym\nenv = gym.make('MountainCar-v0')\nstate, _ = env.reset()\n\naction, prob = agent.get_action(state)\nns, r, term, trunc, _ = env.step(action)\ndone = term or trunc\nagent.update(state, prob, r, ns, done)\nstate = ns",['OpenAI Gym의 실제 환경을 직접 실행','env.step 직후 매번 1회 갱신\n환경 전이 수와 갱신 횟수가 같음','위치·속도와 보상은 Gym에서 받음'])
code('코드 · 네트워크와 행동 선택',"# experiment.py / Agent.get_action\nprobs = self.pi(state[None, :])[0]\np = probs.data.astype(np.float64)\np /= p.sum()\naction = np.random.choice(3, p=p)\nreturn action, probs[action]",['별도 Actor·Critic · ReLU 신경망','학습·검증·독립 평가·영상 모두\n확률에 따라 행동을 표본 추출','속도 입력 스케일은 별도 비교\n관측값의 수치 크기만 조절'])
code('코드 · Actor와 Critic 갱신',"# experiment.py / Agent.update\ntarget = r + gamma * self.v(ns) * (1-done)\ntarget.unchain()\nvalue = self.v(s)\nloss_v = F.mean_squared_error(value, target)\ndelta = target - value\ndelta.unchain()\nloss_pi = -F.log(action_prob) * delta\n\nself.v.cleargrads(); self.pi.cleargrads()\nloss_v.backward(); loss_pi.backward()\nself.opt_v.update()\nself.opt_pi.update()",['강의와 같은 1-step TD 목표 계산','target·delta는 unchain으로 분리\n두 손실의 역전파 후 각각 Adam 갱신','이 과정을 모든 환경 스텝에서 수행'])
body('코드 · 종료와 시간 제한 처리',['Gym의 terminated 또는 truncated를 done으로 사용','강의 코드의 (1 − done) 마스크를 그대로 적용','목표 도달·200스텝 제한에서는 다음 상태 가치 0.000','학습에는 Gym 원래 보상 −1.000을 그대로 사용'])
body('평가와 정책 선택 규칙',[f"모든 설정에 같은 검증 초기 상태 {S['metadata']['validation_episodes']}개 · 확률 정책 평가",'검증 평균 총보상으로 선택 · 동률이면 보상 표준편차','학습과 평가에 같은 행동 선택 방식 적용','선택 이후 별도 초기 상태 100개로 독립 평가\n오차막대는 검증 에피소드 간 표준편차'])
for name,vs in GROUPS.items():
 qs=[result(dict(BASE,**{name:v}))['validation'] for v in vs];bestidx=max(range(len(vs)),key=lambda i:(qs[i]['mean'],-qs[i]['std']));best=vs[bestidx]
 fig,ax=plt.subplots(figsize=(6,4.6));xx=np.arange(len(vs));ax.bar(xx,[q['mean'] for q in qs],color=['#BE1551' if v==BASE[name] else '#1C50B3' for v in vs]);ax.errorbar(xx,[q['mean'] for q in qs],yerr=[q['std'] for q in qs],fmt='none',color='black',capsize=5);ax.set_xticks(xx,[val(name,v) for v in vs],fontsize=10);ax.set(ylim=(-225,0),ylabel='Original total reward (higher is better)',xlabel=NAMES[name]);ax.grid(axis='y',alpha=.2)
 for i,q in enumerate(qs):ax.text(i,-218,f"{q['mean']:.3f}",ha='center',fontsize=10)
 path=savefig(fig,name)
 panel('하이퍼파라미터 · '+NAMES[name],path,'한 조건만 변경',['후보: '+' / '.join(val(name,v) for v in vs),MEANINGS[name],'그룹 우수값: '+val(name,best)+'\n'+metric(qs[bestidx]),'분홍: 기준값 · 오차막대: 표준편차'])
 lines=PARAMETER_ANALYSIS['parameters'][name]['slides']
 # Keep the user's deleted caveats and selection boilerplate absent.
 body(NAMES[name]+' · 결과 해석',lines)
 RECORDS.append(dict(name=name,title=NAMES[name],values=vs,results=qs,figure=str(path),meaning=MEANINGS[name],interpretation=lines,report_analysis=PARAMETER_ANALYSIS['parameters'][name]['report']))
combo=S['combination_result'];win=result(S['winner']);base=result(BASE);test=S['test']
basic_candidates=[result(dict(BASE,**{name:value})) for name,values in GROUPS.items() for value in values]
extra_candidates=[result(dict(S['refinement_base'],**{name:value})) for name,values in S['refinement_sweeps'].items() for value in values]
basic_best=max(basic_candidates,key=lambda r:(r['validation']['mean'],-r['validation']['std']))
extra_best=max(extra_candidates,key=lambda r:(r['validation']['mean'],-r['validation']['std']))
# Full additional-study table keeps all tested candidate values reviewable.
extra_rows=[('추가 탐색 기준','—',result(S['refinement_base'])['validation'])]
for name,values in S['refinement_sweeps'].items():
 for value in values:
  if value!=S['refinement_base'][name]:extra_rows.append((NAMES[name],val(name,value),result(dict(S['refinement_base'],**{name:value}))['validation']))
s=slide(3,'추가 탐색 · 후보별 결과')
sh=next(x for x in s.shapes if x.name=='Body');sh.element.getparent().remove(sh.element)
table=s.shapes.add_table(len(extra_rows)+1,5,Pt(44),Pt(110),Pt(872),Pt(312)).table
for col,width in zip(table.columns,[260,180,150,150,132]):col.width=Pt(width)
rows=[['변경 항목','후보값','평균 보상','표준편차','도달률']]+[[label,value,f"{q['mean']:.3f}",f"{q['std']:.3f}",f"{q['success']*100:.3f}%"] for label,value,q in extra_rows]
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
for i,row in enumerate(rows):
 table.rows[i].height=Pt(26)
 for j,value in enumerate(row):
  cell=table.cell(i,j);cell.text=value;cell.margin_left=Pt(8);cell.margin_right=Pt(8);cell.margin_top=Pt(2);cell.margin_bottom=Pt(2);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
  cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string('1C50B3' if i==0 else ('EEF3FB' if i%2 else 'FFFFFF'))
  par=cell.text_frame.paragraphs[0];par.line_spacing=1.3;par.space_before=Pt(0);par.space_after=Pt(0);par.alignment=PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER
  for run in par.runs:run.font.name='Pretendard';run.font.bold=True;run.font.size=Pt(14);run.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '080808')
caption=s.shapes.add_textbox(Pt(44),Pt(439),Pt(872),Pt(31));caption.text='기준: 노드 256 · 속도 입력 스케일 10000.000 · 나머지 조건은 기본 탐색 기준값'
for par in caption.text_frame.paragraphs:
 par.line_spacing=1.3
 for run in par.runs:run.font.name='Pretendard';run.font.bold=True;run.font.size=Pt(14)
qbase=result(S['refinement_base'])['validation']
q8=result(dict(S['refinement_base'],beta1=.8))['validation'];q95=result(dict(S['refinement_base'],beta1=.95))['validation']
body('추가 탐색 · 결과 해석',['기본 탐색 최고: '+metric(basic_best['validation']),'추가 탐색 최고: '+metric(extra_best['validation']),f"추가 탐색 기준 대비 평균 보상 {extra_best['validation']['mean']-qbase['mean']:.3f} 개선",f"Adam β₁ 0.800 / 0.950의 도달률: {q8['success']*100:.3f}% / {q95['success']*100:.3f}%"])

comboitems=[NAMES[k]+' '+val(k,v) for k,v in S['combination'].items() if k in NAMES and v!=BASE[k]]
body('우수값 조합 · 실제 재학습 결과',[winnertext(S['combination']),'조합 검증: '+metric(combo['validation']),'추가 탐색 조합: '+metric(S['refinement_combination_result']['validation']),f"총 {len(S['configs'])}개 고유 설정 비교 · 두 조합도 별도 재학습"])
body('최종 정책 · 선택 결과',[winnertext(S['winner']),'검증: '+metric(win['validation']),f"독립 평가 100회: 평균 {test['mean']:.3f} · 표준편차 {test['std']:.3f}",f"도달률 {test['success']*100:.3f}% · 평균 소요 {-test['mean']:.3f}스텝"])
fig,axes=plt.subplots(1,2,figsize=(11,4))
for rr,label in [(base,'Baseline'),(win,'Selected')]:axes[0].plot([h['episode'] for h in rr['history']],[h['train_mean'] for h in rr['history']],label=label)
axes[0].set(xlabel='Training episodes',ylabel='Original total reward (higher is better)');axes[0].legend();axes[1].hist(test['returns'],bins=12,color='#1C50B3');axes[1].set(xlabel='Held-out total reward',ylabel='Episodes');plt.tight_layout();curve=savefig(fig,'learning_and_test')
s=slide(7,'학습 경과와 독립 평가 분포');image(s,'LeftImageSlot',curve);sh=next(x for x in s.shapes if x.name=='RightImageSlot');sh.element.getparent().remove(sh.element);pic=s.shapes[-1];pic.left=Pt(27);pic.top=Pt(132);pic.width=Pt(904);pic.height=Pt(329);setv(s,'LeftTitle','학습 보상 경과 / 독립 평가 보상 분포');setv(s,'RightTitle','');setv(s,'Caption','• 왼쪽: 최근 100개 학습 에피소드 평균 · 오른쪽: 확률 정책의 독립 평가')
pol=SavedPolicy(C/f"results/{S['winner_key']}/actor.npz");fig,axes=plt.subplots(1,2,figsize=(11,4));X,V=np.meshgrid(np.linspace(-1.2,.6,180),np.linspace(-.07,.07,140));probs=pol.probabilities(np.stack([X,V],-1));aa=probs.argmax(-1);im=axes[0].pcolormesh(X,V,aa,cmap=matplotlib.colors.ListedColormap(['#1C50B3','#ddd','#BE1551']),shading='auto',vmin=0,vmax=2);fig.colorbar(im,ax=axes[0],ticks=[0,1,2],label='Most probable action');axes[0].set(xlabel='Position',ylabel='Velocity');tr=np.array(E['episodes'][0]['trajectory']);axes[1].plot(tr[:,0],tr[:,1]);axes[1].scatter(tr[[0,-1],0],tr[[0,-1],1]);axes[1].set(xlabel='Position',ylabel='Velocity');plt.tight_layout();policyfig=savefig(fig,'policy_trajectory')
s=slide(7,'정책 맵과 첫 평가 에피소드');image(s,'LeftImageSlot',policyfig);sh=next(x for x in s.shapes if x.name=='RightImageSlot');sh.element.getparent().remove(sh.element);pic=s.shapes[-1];pic.left=Pt(27);pic.top=Pt(132);pic.width=Pt(904);pic.height=Pt(329);setv(s,'LeftTitle','확률이 가장 큰 행동의 분포 / 실제 평가 궤적');setv(s,'RightTitle','');setv(s,'Caption','• 정책 맵은 확률 분포의 시각화 · 실제 실행에서는 확률에 따라 행동 선택')
panel('결과 동영상 · OpenAI Gym 실행',C/'videos/result_poster.png','결과 영상',['독립 평가의 첫 에피소드 전체','OpenAI Gym 기본 렌더러의 화면\n스텝·원래 보상·종료 여부 표시','학습·평가와 같은 확률 정책으로 실행'])
body('결론과 해석의 범위',[f"선택 정책 독립 평가 평균: {test['mean']:.3f}",f"독립 평가 도달률: {test['success']*100:.3f}%",'강의의 1-step TD · 매 스텝 갱신 방식으로 비교','실행한 후보 중 선택한 결과 · 전역 최적을 뜻하지 않음'])
p.save(O/'RL_week11.pptx');(D/'report_data.json').write_text(json.dumps(dict(groups=RECORDS,base=base,winner=win,combination=combo),ensure_ascii=False,indent=2));(D/'slide_titles.json').write_text(json.dumps([next((sh.text for sh in sl.shapes if sh.name in ['Title','CoverTitle']),'') for sl in p.slides],ensure_ascii=False,indent=2));print('Saved',len(p.slides),'slides')
