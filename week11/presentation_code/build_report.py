"""Generate the A4 design report directly from measured experiment outputs."""
from pathlib import Path
import json, math
import pymupdf as fitz
D=Path('/home/a/ppt_workspace/ppt_info/RL_week11');R=Path('/home/a/ppt_workspace');C=Path('/home/a/Reinforcement_Learning_class/week11/assignment');O=R/'output/RL_week11'
S=json.loads((C/'results/summary.json').read_text());E=json.loads((C/'results/independent_evaluation.json').read_text())
FONT=str(R/'templet/fonts/Pretendard-Bold.ttf');font=fitz.Font(fontfile=FONT);doc=fitz.open();PAGE=None;y=0;blue=(.11,.31,.70);W=595;H=842
source=[]
def lines(txt,size,width):
 out=[]
 for para in txt.split('\n'):
  line=''
  for word in para.split(' '):
   candidate=(line+' '+word) if line else word
   if font.text_length(candidate,fontsize=size)<=width:
    line=candidate
   else:
    if line:out.append(line)
    line=word
    if font.text_length(line,fontsize=size)>width:
     part=''
     for ch in line:
      if font.text_length(part+ch,fontsize=size)>width:out.append(part);part=ch
      else:part+=ch
     line=part
  out.append(line)
 return out
def text(txt,size=10.5,color=(.08,.08,.08),gap=8):
 global y
 ll=lines(txt,size,499)
 assert y+len(ll)*size*1.65 < 793, (len(doc),y,txt[:40])
 for line in ll:
  PAGE.insert_text((48,y+size),line,fontname='pret',fontsize=size,color=color);y+=size*1.65
 y+=gap
 source.append(txt)
def page(title,subtitle='강화학습 week11 · Quiz1 설계 보고서'):
 global PAGE,y
 PAGE=doc.new_page(width=W,height=H);PAGE.insert_font(fontname='pret',fontfile=FONT)
 PAGE.draw_rect(fitz.Rect(0,0,W,65),color=None,fill=blue);PAGE.insert_text((35,39),title,fontname='pret',fontsize=20,color=(1,1,1));PAGE.insert_text((48,89),subtitle,fontname='pret',fontsize=9,color=blue)
 PAGE.insert_text((48,818),f'{len(doc):02}  |  MountainCar Actor-Critic',fontname='pret',fontsize=8,color=(.4,.4,.4));y=112
 source.append('\n'+title+'\n')
def heading(t):text(t,13,blue,6)
def image(path,height):
 global y
 PAGE.insert_image(fitz.Rect(48,y,547,y+height),filename=str(path),keep_proportion=True);y+=height+12

def table(headers,rows,widths=None,rowheight=30):
 global y
 widths=widths or [499/len(headers)]*len(headers)
 allrows=[headers]+rows
 for i,row in enumerate(allrows):
  h=rowheight;x=48
  if i==0:PAGE.draw_rect(fitz.Rect(48,y,547,y+h),color=None,fill=(.92,.95,.99))
  for value,ww in zip(row,widths):
   PAGE.insert_text((x+7,y+h/2+4),str(value),fontname='pret',fontsize=9,color=blue if i==0 else (0,0,0));x+=ww
  PAGE.draw_line((48,y+h),(547,y+h),color=(.83,.85,.9),width=.4);y+=h
 y+=14

def v(name,x):return f'{x*1000:.3f} ×10⁻³' if name.endswith('_lr') else (f'{x:.3f}' if isinstance(x,float) else str(x))
from pptx import Presentation
import hashlib
DECK=O/'RL_week11.pptx';deck=Presentation(DECK);slides={next((sh.text for sh in sl.shapes if sh.name in ['Title','CoverTitle']),''):sl for sl in deck.slides}
def content(title,name='Body'):return next(sh.text for sh in slides[title].shapes if sh.name==name)
def paragraphs(title,name='Body'):return content(title,name).splitlines()
ASSETS=D/'report_assets';ASSETS.mkdir(exist_ok=True)
def current_image(title,stem):
 pic=max([sh for sh in slides[title].shapes if sh.shape_type==13],key=lambda sh:sh.width*sh.height);path=ASSETS/(stem+'.'+pic.image.ext);path.write_bytes(pic.image.blob);return path
ENV=current_image('MountainCar · 목표와 보상','environment');LEARNING=current_image('학습 경과와 독립 평가 분포','learning');POLICY=current_image('정책 맵과 첫 평가 에피소드','policy');VIDEO=current_image('결과 동영상 · OpenAI Gym 실행','video')
DATA=json.loads((D/'report_data.json').read_text());GROUPS=DATA['groups'];test=S['test'];win=DATA['winner'];combo=DATA['combination']
def metric(q):return f"평균 {q['mean']:.3f}, 도달률 {q['success']*100:.3f}%"
NAMES={g['name']:g['title'] for g in GROUPS}
def changes(c):return ' / '.join(NAMES[k]+' '+v(k,x) for k,x in c.items() if k in NAMES and x!=S['base'][k]) or '기준 설정'
page('MountainCar · Actor-Critic 설계')
heading('1. 과제와 실험 조건')
for line in paragraphs('과제와 실험 조건'):text('• '+line,10.5,gap=5)
heading('MountainCar · 목표와 보상')
text('OpenAI Gym 0.26.2의 MountainCar-v0 환경을 직접 실행한다. 상태는 자동차의 위치와 속도이고 행동은 왼쪽 가속·가속 없음·오른쪽 가속이다. 매 스텝 보상은 −1.000, 최대 길이는 200스텝이다. 위치 0.500 이상이고 속도 0.000 이상이면 목표에 도달한다.')
image(ENV,225)
heading('평가와 정책 선택')
text(f"모든 후보를 동일한 검증 초기 상태 {S['metadata']['validation_episodes']}개에서 평가했다. 학습·검증·독립 평가·동영상 모두 행동 확률에 따라 표본 추출하는 정책을 사용한다. 검증 평균 총보상이 가장 높은 설정을 선택하고, 평균이 같으면 표준편차가 작은 설정을 선택했다.")
text(f"선택 이후 별도 초기 상태 100개로 평가했다. {len(S['configs'])}개 고유 설정을 비교했고 그룹별 우수값 조합도 별도로 재학습했다. 모든 점수는 Gym의 원래 총보상이다.")
heading('주요 결과')
text(f"선택 정책의 검증 {metric(win['validation'])}. 독립 평가 평균 {test['mean']:.3f}, 표준편차 {test['std']:.3f}, 목표 도달률 {test['success']*100:.3f}%이다.")

page('핵심 코드 설명')
heading('2. OpenAI Gym과 매 스텝 갱신')
text('gym.make("MountainCar-v0")로 만든 실제 환경에서 상태와 보상을 받는다. env.step으로 한 번 움직인 직후 agent.update를 한 번 수행한다. 환경 전이 수와 Actor·Critic 갱신 횟수가 동일하다.',10)
heading('확률적 행동 선택')
text(content('코드 · 네트워크와 행동 선택','CodeBody').split('\n',1)[1],9.5,gap=8)
heading('강의의 1-step Actor-Critic 코드')
text(content('코드 · Actor와 Critic 갱신','CodeBody').split('\n',1)[1],9.0,gap=8)
text('target과 delta를 unchain으로 분리하고 Critic 손실과 Actor 손실의 기울기를 각각 계산한다. 다음 환경 스텝으로 이동하기 전에 두 Adam optimizer를 갱신한다. ReLU·DeZero 구현을 사용한다.',10)
heading('종료 처리와 입력')
text('강의의 done 마스크를 적용한다. 목표 도달 또는 200스텝 시간 제한에서 다음 상태 가치를 0.000으로 처리한다. 원래 보상을 그대로 사용하며 속도 입력에 곱하는 스케일은 별도 하이퍼파라미터로 비교한다.',10)

page('하이퍼파라미터 · 전체 비교표')
heading('3. 한 조건만 변경한 결과')
text('아래 표는 기본 탐색에서 한 항목만 변경한 결과이다. 학습률은 두 네트워크의 갱신 크기, 노드·층수는 모델 용량, Adam 1차 모멘트는 기울기 누적, 속도 스케일은 입력 크기, 에피소드 수는 학습량을 조절한다. 할인율은 미래 보상의 반영 비율이다.',10)
rows=[]
for g in GROUPS:
 for i,(x,q) in enumerate(zip(g['values'],g['results'])):rows.append([g['title'] if i==0 else '',v(g['name'],x),f"{q['mean']:.3f}",f"{q['std']:.3f}",f"{q['success']*100:.3f}%"])
table(['변경 항목','후보값','평균 보상','표준편차','도달률'],rows,[123,118,86,78,94],rowheight=20)
text('표준편차는 검증 초기 상태와 행동 표본에 따른 에피소드 보상의 차이이다. 실제 실행에서도 확률적으로 행동을 선택하므로 같은 시작 위치에서 다른 경로가 나타날 수 있다.',9.5)

page('하이퍼파라미터 · 결과 해석')
heading('4. 항목별 주요 관찰')
for g in GROUPS:text(g['title']+' · '+g['report_analysis'],10,gap=10)

page('추가 탐색과 최종 정책')
heading('5. 추가 탐색의 후보별 결과')
text('노드 256·속도 스케일 10000.000을 기준으로 할인율·두 학습률·노드·Adam 계수를 각각 변경했다. 기본 탐색 조합 검증 '+metric(combo['validation'])+'. 추가 탐색 조합 검증 '+metric(S['refinement_combination_result']['validation'])+'.',10)
extra_rows=[['추가 탐색 기준','—',result_q] for result_q in [next(r['validation'] for r in S['results'] if r['config']==S['refinement_base'])]]
for name,values in S['refinement_sweeps'].items():
 for value in values:
  if value!=S['refinement_base'][name]:
   cfg=dict(S['refinement_base'],**{name:value});q=next(r['validation'] for r in S['results'] if r['config']==cfg);extra_rows.append([NAMES[name],v(name,value),q])
table(['변경 항목','후보값','평균 보상','표준편차','도달률'],[[label,value,f"{q['mean']:.3f}",f"{q['std']:.3f}",f"{q['success']*100:.3f}%"] for label,value,q in extra_rows],[123,118,86,78,94],rowheight=18)
heading('최종 선택과 독립 평가')
text('선택 설정: '+changes(S['winner'])+'. 전체 후보와 두 조합을 검증 평균으로 비교해 선택했고, 독립 평가 점수로 다시 선택하지 않았다.',10)
table(['평가 항목','결과'],[['검증 평균 / 도달률',f"{win['validation']['mean']:.3f} / {win['validation']['success']*100:.3f}%"],['독립 평균 / 표준편차',f"{test['mean']:.3f} / {test['std']:.3f}"],['독립 도달률 / 평균 스텝',f"{test['success']*100:.3f}% / {-test['mean']:.3f}"],['최소 / 최대 보상',f"{min(test['returns']):.3f} / {max(test['returns']):.3f}"]],[220,279],rowheight=27)
text('실행한 후보 안에서의 결과이며 전역 최적을 증명한 것은 아니다. 목표 미도달은 200스텝 제한의 −200.000점으로 기록했다.',10)

page('정책 동작과 결과 동영상')
heading('6. 확률 정책의 동작')
image(POLICY,185)
text('왼쪽은 각 상태에서 확률이 가장 큰 행동을 색으로 표시한 맵이고, 오른쪽은 독립 평가 첫 에피소드의 실제 궤적이다. 실행에서는 맵의 색을 그대로 고르는 대신 행동 확률에 따라 표본 추출한다.',10)
heading('OpenAI Gym 결과 동영상')
image(VIDEO,230)
text('선택한 정책으로 독립 평가 첫 에피소드를 실행한 OpenAI Gym 기본 렌더러 화면이다. 스텝·누적 보상·위치·속도·목표 도달 여부를 표시한다. 동일한 영상 파일을 보고서 PDF에도 첨부했다.',10)

assert len(doc)==6
source_text='\n\n'.join(source)
for phrase in ['소수 수치는','셋째 자리','작은 학습률은','강의 방법과 구현 범위','손실과 TD 목표','네트워크와 공통 설정','네트워크와 기준값','제출 파일과 재현 방법','소스와 실행','추가 검증 필요','우수성은 미검증','기준 구조인 64','최고 중간점','중간 최고','assignment/','presentation_code/','--workers','--video']:
 assert phrase not in source_text,phrase
doc.embfile_add('result.mp4',(O/'result.mp4').read_bytes(),filename='result.mp4',desc='Native OpenAI Gym execution of the selected categorical policy');doc.subset_fonts();doc.save(O/'RL_week11_design_report.pdf',garbage=4,deflate=True);(D/'report_content.txt').write_text(source_text)
(D/'report_from_edited_deck.json').write_text(json.dumps(dict(source_pptx=str(DECK),source_pptx_sha256=hashlib.sha256(DECK.read_bytes()).hexdigest(),source_slide_count=len(deck.slides),report_pages=len(doc),protocol=S['protocol'],deleted_topics_restored=False),ensure_ascii=False,indent=2));print('Report:',len(doc),'pages')
