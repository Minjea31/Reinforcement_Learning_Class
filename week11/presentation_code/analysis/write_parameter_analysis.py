from pathlib import Path
import json,hashlib
D=Path('/home/a/ppt_workspace/ppt_info/RL_week11');C=Path('/home/a/Reinforcement_Learning_class/week11/assignment');S=json.loads((C/'results/summary.json').read_text());A=json.loads((D/'analysis/policy_diagnostics.json').read_text())
def diag(phase,name=None,value=None):
 c=dict(S['base'] if phase=='basic' else S['refinement_base'])
 if name:c[name]=value
 r=next(r for r in S['results'] if r['config']==c);return A[r['key']]
def pct(a,i):return f"{a['action_fractions'][i]*100:.3f}%"
def conf(a):return f"{a['confident_state_fraction']*100:.3f}%"
b=diag('basic');g9=diag('additional','gamma',.9);g995=diag('additional','gamma',.995);add=diag('additional')
a1=diag('additional','actor_lr',.001);alow=diag('additional','actor_lr',.00005)
cgood=diag('basic','critic_lr',.001);clow=diag('additional','critic_lr',.0001);chigh=diag('additional','critic_lr',.003)
w64=diag('basic','width',64);wa128=diag('additional','width',128);wa64=diag('additional','width',64)
d2=diag('basic','depth',2);d3=diag('basic','depth',3);be8=diag('basic','beta1',.8);be95=diag('basic','beta1',.95);bae8=diag('additional','beta1',.8);bae95=diag('additional','beta1',.95)
v1=diag('basic','velocity_scale',1.);e500=diag('basic','episodes',500);e2000=diag('basic','episodes',2000)
analysis={
'gamma':dict(slides=[
 f'50스텝 뒤 보상의 가중치: γ 0.900 / 0.980 / 0.995\n각각 {.9**50:.3f} / {.98**50:.3f} / {.995**50:.3f} · 미래 반영 정도가 다름',
 '기본 탐색은 세 후보 모두 학습 보상 −200.000\n검증에서도 목표 도달 차이가 드러나지 않음',
 '추가 탐색에서 0.900은 0.980보다 평균 보상 1.500 개선\n도달률은 모두 100.000% · 표준편차는 7.257 대 18.649',
 f"추가 0.995는 오른쪽 가속 {pct(g995,2)} · 최고 위치 {g995['max_position']:.3f}\n먼 보상을 더 반영해도 목표까지 가는 왕복 전략은 유지되지 않음"],
 report='γ가 커지면 50스텝 뒤 보상의 가중치는 0.005→0.364→0.778로 커진다. 추가 탐색의 0.900은 0.980보다 평균 보상이 1.500 높고 표준편차도 7.257로 작았다. 0.995는 오른쪽 가속 99.800%에 편중되어 최고 위치 −0.150에 그쳤다.'),
'actor_lr':dict(slides=[
 'Actor 학습률은 TD 오차가 행동 확률을 바꾸는 크기\n값이 커질수록 한 번의 정책 변화가 커질 수 있음',
 '기본 후보 0.200 / 1.000 / 3.000 ×10⁻³은 모두 −200.000\n추가 탐색에서는 중간값 0.200 ×10⁻³만 검증 도달률 100.000%',
 f"추가 0.050 / 1.000 ×10⁻³의 200스텝 미만 도달은 {alow['train_goal_episodes']} / {a1['train_goal_episodes']}회\n학습 중 도달 경험이 있어도 두 모델의 검증 도달률은 0.000%",
 f"추가 1.000 ×10⁻³의 검증 행동은 {pct(a1,2)} 오른쪽 가속\n확률이 한 행동에 몰리는 것과 빠른 목표 도달은 다름"],
 report='Actor 학습률은 행동 확률의 갱신 크기이다. 추가 탐색에서 0.200 ×10⁻³은 검증 도달률 100.000%였지만 0.050·1.000 ×10⁻³은 0.000%였다. 큰 후보의 검증 행동은 100.000% 오른쪽 가속으로 고정되어 왕복 전략을 형성하지 못했다.'),
'critic_lr':dict(slides=[
 'Critic의 TD 오차가 Actor 손실의 가중치가 됨\n가치 학습이 부정확하면 정책 갱신의 기준도 달라짐',
 f"기본 1.000 ×10⁻³은 {cgood['first_train_goal_episode']}번째에 처음 200스텝 미만 도달\n이후 총 {cgood['train_goal_episodes']}회 · 검증 평균 −130.967 / 도달률 100.000%",
 '기본 0.500 / 3.000 ×10⁻³은 학습 보상도 모두 −200.000\n반면 추가 탐색에서는 0.500 ×10⁻³이 도달률 100.000%',
 f"추가 3.000 ×10⁻³의 최근 학습 100회 평균은 {chigh['last100_train_mean']:.3f}\n학습 중 계속 바뀌는 정책의 기록과 고정된 검증 정책의 성능은 다름"],
 report='Critic의 오차는 Actor 갱신의 기준이다. 기본 1.000 ×10⁻³은 657번째에 처음 200스텝 미만으로 도달했고 검증 도달률도 100.000%였다. 추가 조건에서는 0.500 ×10⁻³이 우수해, 가치 학습률의 효과가 다른 설정과 함께 달라짐을 보여준다.'),
'width':dict(slides=[
 '노드 수는 상태·행동 관계를 표현하는 용량을 바꿈\n용량이 늘어도 목표에 맞는 행동을 학습했는지는 따로 확인해야 함',
 f"기본 64 노드는 검증 행동의 {pct(w64,2)}가 오른쪽 가속\n최고 위치 {w64['max_position']:.3f}로 이동은 했지만 목표 0.500에는 미달",
 '추가 조건의 64 / 128 / 256 노드 도달률\n0.000% / 60.000% / 100.000% · 이 조건에서는 노드 증가와 함께 개선',
 f"추가 128 → 256 노드에서 평균 보상 {add['validation_mean']-wa128['validation_mean']:.3f} 개선\n표준편차도 {wa128['validation_std']:.3f} → {add['validation_std']:.3f}로 감소"],
 report='기본 64 노드는 오른쪽 가속 98.600%, 최고 위치 −0.137로 움직였지만 목표에는 못 갔다. 추가 조건에서는 64·128·256 노드의 도달률이 0.000%·60.000%·100.000%였다. 128→256에서 평균 보상은 54.067 개선되어 용량의 효과가 조건에 따라 드러났다.'),
'depth':dict(slides=[
 '층을 늘리면 상태의 비선형 관계를 더 복잡하게 표현\n동시에 1-step TD로 조정할 가중치도 많아짐',
 f"1 / 2 / 3층의 검증 최고 위치: {b['max_position']:.3f} / {d2['max_position']:.3f} / {d3['max_position']:.3f}\n모두 목표 0.500보다 낮아 오른쪽 언덕까지 가는 궤적이 형성되지 않음",
 f"2 / 3층의 검증 스텝 중 최대 행동확률 ≥ 0.990인 비율\n{conf(d2)} / {conf(d3)} · 행동은 확신적으로 선택됐지만 도달률은 0.000%",
 '따라서 이번 결과는 표현 용량 부족만으로 설명하기 어려움\n정책 확률이 뚜렷해지는 것과 목표에 맞는 전략을 얻는 것은 별개'],
 report='1·2·3층의 검증 최고 위치는 −0.401·−0.371·−0.380이었다. 2·3층은 최대 행동확률이 0.990 이상인 검증 스텝이 80.333%·72.667%였지만 도달률은 0.000%였다. 정책이 확신적으로 바뀌어도 그 선택이 목표에 적합한 것은 아니다.'),
'beta1':dict(slides=[
 'β₁은 과거 기울기 방향을 얼마나 오래 반영할지 조절\n0.950은 0.800보다 이전 방향을 더 오래 유지',
 f"기본 세 후보의 보상은 같아도 가속 없음 행동 비중은 다름\n0.800 / 0.900 / 0.950: {pct(be8,1)} / {pct(b,1)} / {pct(be95,1)}",
 '추가 조건의 0.800 / 0.900 / 0.950 도달률\n66.667% / 100.000% / 0.000% · 가장 큰 값이 우수하지 않았음',
 f"추가 0.950은 200스텝 미만 도달 경험이 {bae95['train_goal_episodes']}회 있었음\n그러나 최근 학습 100회 평균 −200.000 · 좋은 동작이 끝까지 유지되지 않음"],
 report='기본 β₁ 세 후보는 보상이 같았지만 가속 없음 행동 비중은 0.000%·61.717%·65.583%로 달랐다. 추가 조건의 도달률은 66.667%·100.000%·0.000%였다. 0.950은 과거에 도달한 경험이 있어도 최근 학습 100회 보상이 모두 −200.000으로 악화됐다.'),
'velocity_scale':dict(slides=[
 '원래 속도 범위 ±0.070에 배율을 곱해 입력 크기를 바꿈\n신경망이 위치와 속도에 반응하는 정도 및 학습 조건에 영향',
 f"배율 1.000은 검증 행동 {pct(v1,2)}가 오른쪽 가속\n최고 위치 {v1['max_position']:.3f} · 한 방향 가속만으로는 목표에 도달하지 못함",
 '기본 100.000 / 1000.000도 검증 도달률 0.000%\n배율 증가만으로 이동 방향을 바꾸는 전략이 만들어지지는 않음',
 '추가 기준인 노드 256·배율 10000.000에서는 도달률 100.000%\n두 조건이 함께 바뀌었으므로 개선을 배율 하나의 효과로 단정할 수 없음'],
 report='배율 1.000은 오른쪽 가속 100.000%, 최고 위치 −0.150으로 한 방향에 머물렀다. 기본 배율 100.000·1000.000도 검증에서는 실패했다. 추가 조건의 성공은 노드 256과 배율 10000.000을 함께 바꾼 결과이므로 배율 단독 효과로 해석할 수 없다.'),
'episodes':dict(slides=[
 '500 → 2000개는 학습량을 4배로 늘린 비교\n추가 1500회에서도 −200.000보다 좋은 학습 보상은 나오지 않음',
 '검증 보상뿐 아니라 학습 기록도 모두 −200.000\n반복량 증가가 목표 도달 시간을 줄이는 방향으로 작동하지 않았음',
 f"검증 스텝 중 최대 행동확률 ≥ 0.990인 비율은\n500회 {conf(e500)} → 2000회 {conf(e2000)}로 증가했지만 도달률은 0.000%",
 '반복은 기존 행동을 더 확신적으로 만들 수도 있음\n같은 기본 탐색에서는 Critic 학습률 변경에서 목표 도달이 나타남'],
 report='학습량을 500→2000개로 4배 늘려도 학습·검증 보상은 모두 −200.000이었다. 최대 행동확률이 0.990 이상인 검증 스텝은 4.583%→38.100%로 늘었다. 반복이 행동을 더 확신적으로 만들었지만 목표에 맞는 전략으로 개선되지는 않았다.')}
# Definition note prevents calling reward=-200 a definitely unsuccessful training episode.
record=dict(summary_sha256=hashlib.sha256((C/'results/summary.json').read_bytes()).hexdigest(),diagnostics_sha256=hashlib.sha256((D/'analysis/policy_diagnostics.json').read_bytes()).hexdigest(),method='Same 30 native-Gym validation episodes; all return sequences exactly reproduced. Action frequencies are weighted by visited validation steps. Training goal count means reward > -200, i.e. goal strictly before step 200.',parameters=analysis)
(D/'parameter_analysis.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print('Analysis narratives generated for',len(analysis),'parameters')
