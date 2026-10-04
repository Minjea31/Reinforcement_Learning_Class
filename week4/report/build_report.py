"""기존 그래프와 분석 문서에서 Week4 보고서 HTML 생성."""
from pathlib import Path
import html
import re
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
slides = []


def esc(text):
    return html.escape(text)


def page(title, content, section=False):
    slides.append(f'<section class="slide"><header>{esc(title)}</header>{content}'
                  f'<footer><span>{len(slides)+1:02d}</span><b>CBNU</b></footer></section>')


def box(title, paragraphs):
    return '<aside><h3>• &nbsp;'+esc(title)+'</h3><div>'+''.join(
        '<p>'+esc(p.replace('**','').replace('`',''))+'</p>' for p in paragraphs)+'</div></aside>'


def split(title, left, label, paragraphs):
    page(title, '<main class="split"><div class="visual">'+left+'</div>'
         +box(label, paragraphs)+'</main>')


def image(path):
    return f'<img src="../{esc(path)}">'


def code(path, start=None, end=None):
    source = (ROOT / path).read_text()
    if start:
        source = source[source.index(start):]
    if end:
        source = source[:source.index(end)]
    return '<div class="code">'+highlight(source.strip(), PythonLexer(),
                                            HtmlFormatter(style='monokai'))+'</div>'


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+esc(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'


slides.append('''<section class="slide cover"><div class="coverblue"><b>CBNU</b>
<h1>강화학습<br>Week4 HW</h1><p>Quiz 1 · Quiz 2</p></div>
<div class="identity">지능로봇공학과 2021042040 김민재</div></section>''')
page('Contents', '''<div class="contents"><h2>1. Quiz 1 — Policy Evaluation</h2>
<p>a. 코드 분석<br>b. 다섯 정책의 결과 분석<br>c. 정책에 따른 V 변화</p>
<h2>2. Quiz 2 — 5×5 GridWorld</h2><p>a. 정책 반복 과정 및 최종 결과<br>
b. 가치 반복 과정 및 최종 결과</p><h2>3. 전체 결과 비교</h2></div>''')
page('', '<div class="divider">Quiz 1<br><small>Policy Evaluation</small></div>')
split('1. Quiz 1 – 환경 및 계산 규칙', image('week4Q1_graphs/policy_1.png'), 'GridWorld', [
    '3×4 환경 / 시작 (2,0) / 목표 (0,3) / 벽 (1,1)',
    '행동 순서: UP, DOWN, LEFT, RIGHT. 할인율 γ=0.9, 임곗값 0.001.',
    '목표 진입 보상 +1, (1,3) 진입 또는 머무름 보상 -1. 경계·벽에 막히면 현재 칸에 머문다.',
    '목표는 종료 상태로 V=0. 그래프의 숫자는 해당 정책을 따를 때의 기대 할인 보상이다.'
])
split('1. Quiz 1 – 코드 분석', code('policy_eval.py', 'def eval_onestep', 'def policy_eval'), 'One-step Evaluation', [
    '각 상태에서 가능한 행동의 보상과 다음 상태 가치를 계산하고 행동 확률로 가중 평균한다.',
    'V(s) = Σ π(a|s) [r + 0.9V(s′)]',
    'r은 현재 칸의 표시 보상이 아니라 이번 행동으로 다음 칸에 들어갈 때 받는 보상이다.',
    '같은 딕셔너리를 순서대로 갱신하므로 이미 갱신한 이웃의 값을 바로 사용한다.'
])
split('1. Quiz 1 – 코드 분석', code('policy_eval.py', 'def policy_eval'), 'Iterative Evaluation', [
    '현재 정책을 고정하고 one-step evaluation을 반복한다.',
    '갱신 전후 가치 차이의 최댓값 Δ가 0.001 미만이면 종료한다.',
    '다섯 정책을 각각 독립적으로 평가한다. 정책 ①에서 ⑤로 개선하는 과정은 아니다.'
])
q1 = [
    ('① 위로만 이동', '[1, 0, 0, 0]', [
        '대부분의 칸은 위쪽 경계나 벽에 막혀 보상 0만 받으므로 V=0이다.',
        '(1,3)은 바로 목표로 들어가 V=1이다.',
        '(2,3)은 -1을 받은 뒤 목표 보상 +1을 받아 V=-1+0.9×1=-0.1이다.',
        '시작 상태 V(2,0)=0. 목표까지 연결되는 경로가 없다.']),
    ('② 아래로만 이동', '[0, 1, 0, 0]', [
        '모든 유효 상태의 V=0이다. 아래로 이동하다가 벽이나 아래쪽 경계에 머문다.',
        '(1,3)에서 출발해도 아래 칸의 보상이 0이므로 V=0이다.',
        '목표 (0,3)에서 아래로 이동한다고 가정하면 -1이지만, 목표는 종료 상태이므로 추가 행동을 하지 않고 V=0으로 고정한다.']),
    ('③ 왼쪽으로만 이동', '[0, 0, 1, 0]', [
        '목표 반대쪽으로 이동하다가 왼쪽 경계나 벽에 머문다.',
        '목표에도 음수 보상 칸에도 진입하지 않아 모든 유효 상태의 V=0이다.',
        '②와 이동 방향은 다르지만 받을 보상이 모두 0이라 가치 결과는 같다.']),
    ('④ 오른쪽으로만 이동', '[0, 0, 0, 1]', [
        '위 행에서는 목표에 도달한다. 목표에 가까울수록 0.81 → 0.90 → 1.00으로 가치가 커진다.',
        '(1,2)와 (1,3)은 음수 보상 칸에 계속 머물러 -1을 반복해서 받는다.',
        '이론값은 -1/(1-0.9)=-10. 계산값 -9.9914는 종료 임곗값에 따른 근삿값이다.',
        '시작 상태는 아래 행에서 오른쪽 경계에 머물러 V=0이다.']),
    ('⑤ 위쪽 선호 확률적 이동', '[0.7, 0.1, 0.1, 0.1]', [
        '위 70%, 나머지 방향 각 10%. 그래프는 가장 확률이 높은 위쪽 화살표만 표시한다.',
        '오른쪽으로도 이동할 수 있어 왼쪽 칸에서도 목표 보상을 얻을 가능성이 생긴다. 시작 상태 V≈0.0392.',
        '(1,3)은 70% 확률로 목표에 진입해 V≈0.6458이다.',
        '(2,3)은 음수 보상 칸에 들어갈 가능성이 높아 V≈-0.3449이다.'])
]
for i, (title, probs, paragraphs) in enumerate(q1, 1):
    split('1. Quiz 1 – 결과 분석', image(f'week4Q1_graphs/policy_{i}.png'), title+' '+probs, paragraphs)
page('1. Quiz 1 – 정책에 따른 V 비교', '<div class="full">'+table(
    ['상태', '① 위', '② 아래', '③ 왼쪽', '④ 오른쪽', '⑤ 위쪽 선호'], [
        ['(0,2)', '0','0','0','1','0.3513'],
        ['(1,2)', '0','0','0','-9.9914','0.2085'],
        ['(1,3)', '1','0','0','-9.9914','0.6458'],
        ['(2,3)', '-0.1','0','0','0','-0.3449'],
        ['시작 (2,0)', '0','0','0','0','0.0392'],
    ])+box('V가 변하는 이유', [
        '정책이 바뀌면 목표까지의 경로·이동 시간·음수 보상 횟수가 달라진다.',
        '같은 칸의 보상 표시는 고정되어 있어도 미래 보상의 기대값 V는 달라진다.',
        '⑤가 시작 상태에서 다섯 정책 중 가장 높다. 전체 가능한 정책 중 최적이라는 의미는 아니다.'
    ])+'</div>')
split('1. Quiz 1 – V 계산 예시', image('week4Q1_graphs/policy_5.png'), '(0,2)와 (2,3)의 변화', [
    '(0,2): ④는 즉시 목표에 들어가 V=1. ⑤는 즉시 오른쪽으로 갈 확률이 10%이고 경계에 머무르는 시간이 길어 V≈0.3513으로 감소한다.',
    '⑤의 V(0,2) = 0.7[0.9V(0,2)] + 0.1[0.9V(1,2)] + 0.1[0.9V(0,1)] + 0.1×1.',
    '(2,3): ①은 -1+0.9×1=-0.1. ⑤에서는 다음 상태 (1,3)의 가치도 0.6458로 낮아져 최종 V≈-0.3449가 된다.',
    '위쪽 이동 확률을 줄였다고 반드시 V가 높아지지는 않는다. 다음 상태의 미래 보상도 함께 달라진다.'
])
page('', '<div class="divider">Quiz 2<br><small>Policy Iteration · Value Iteration</small></div>')
split('2. Quiz 2 – 5×5 환경', image('week4Q2_graphs/value_iteration/step_00.png'), '환경 설정', [
    '시작 (4,0), 목표 (0,4), 벽 (2,1)·(2,2). 좌표는 (행, 열), 0부터 시작한다.',
    '폭탄 (0,3)·(3,4)에 진입하면 -1, 목표에 진입하면 +1. 나머지는 0.',
    '폭탄은 Q1과 같은 비종료 칸으로 해석했다. 목표만 종료 상태이며 V=0이다.',
    '할인율 0.9, 임곗값 0.001. 동일한 환경에서 두 알고리즘을 비교한다.'
])
split('2. Quiz 2 – 정책 반복 코드 분석', code('policy_iter.py', 'def policy_iter', '# 가치 함수'), 'Policy Iteration', [
    '균등 무작위 정책으로 시작한다.',
    '현재 정책을 임곗값까지 평가한 다음, r+0.9V(s′)가 가장 큰 행동으로 정책을 개선한다.',
    '개선 전후 정책이 같으면 종료한다. 그래프는 평가 직후의 가치와 평가에 사용한 현재 정책을 표시한다.',
    '하나의 step 안에 정책 평가를 위한 여러 순회가 포함된다.'
])
split('2. Quiz 2 – 가치 반복 코드 분석', code('value_iter.py', 'def value_iter_onestep', 'def value_iter('), 'Value Iteration', [
    '정책의 확률 평균 대신 모든 행동 중 최댓값으로 V를 갱신한다.',
    'V(s)=max [r+0.9V(s′)]',
    '전체 상태를 한 번 갱신한 결과를 매 단계 저장한다. 최대 변화량이 0.001 미만이면 종료한다.',
    'V가 수렴한 뒤 탐욕 정책을 한 번 계산한다. 동일 순회에서 최신 값을 사용한다.'
])
md = (ROOT / 'week4Q2_graphs/week4Q2_analysis.md').read_text()
parts = md.split('## 정책 반복과 가치 반복의 차이')[0]
for title, folder in [('정책 반복', 'policy_iteration'), ('가치 반복', 'value_iteration')]:
    section = parts.split('## '+title+'\n', 1)[1].split('\n## ', 1)[0]
    for name, body in re.findall(r'### (step_\d+|final)\n(.*?)(?=\n### |\Z)', section, flags=re.S):
        paragraphs = [p.strip().replace('\n',' ') for p in body.strip().split('\n\n')
                      if p.strip() and not p.startswith('![') and not p.startswith('[가치')]
        split(f'2. Quiz 2 – {title} 과정', image(f'week4Q2_graphs/{folder}/{name}.png'), name, paragraphs)
page('2. Quiz 2 – 최종 가치 및 정책', '<main class="pair">'
     '<div><h3>정책 반복</h3>'+image('week4Q2_graphs/policy_iteration/final.png')+'</div>'
     '<div><h3>가치 반복</h3>'+image('week4Q2_graphs/value_iteration/final.png')+'</div>'
     '</main><p class="caption">두 방법의 최종 가치·정책이 일치하며, 시작 상태 V(4,0) ≈ 0.47830이다.</p>')
page('3. 정책 반복과 가치 반복 비교', '<div class="full">'+table(
    ['비교 항목', '정책 반복', '가치 반복'], [
        ['가치 갱신', '정책 확률로 가중 평균', '행동별 값 중 최댓값'],
        ['한 step', '정책 평가 수렴 + 정책 개선', '전체 상태 한 번 갱신'],
        ['정책 계산', '바깥 반복마다 개선', '가치 수렴 후 최종 계산'],
        ['종료 기준', '정책이 더 이상 변하지 않음', '최대 가치 변화 < 0.001'],
        ['이번 실행', '평가·개선 5회', '갱신 7회 (마지막은 변화 확인)'],
    ])+box('단계 수와 계산량', [
        '정책 반복은 초기의 음수 가치가 정책 개선에 따라 양수로 바뀐다.',
        '가치 반복은 목표 주변의 양수 가치가 왼쪽·시작 상태로 점차 전달된다.',
        '한 step의 계산량이 다르므로 그래프 수만으로 실행 속도를 비교할 수 없다.'
    ])+'</div>')
split('3. 전체 결과 분석', image('week4Q2_graphs/value_iteration/final.png'), '결과 해석', [
    'Quiz 1: 같은 환경에서도 어떤 정책을 따르는지에 따라 V가 달라진다. 목표 접근, 보상 지연, 반복 손실을 함께 고려해야 한다.',
    'Quiz 2: 두 알고리즘은 다른 계산 과정으로 동일한 최적 가치와 정책에 도달한다.',
    '시작 상태에서 폭탄을 피하는 8번의 이동으로 목표에 도달한다. 마지막 이동에서만 +1을 받아 V=0.9⁷≈0.47830.',
    '폭탄 칸 자체의 V도 양수일 수 있다. 그 칸에서 출발해 앞으로 받을 보상을 평가하기 때문이다. 목표 V=0은 종료 규칙이다.'
])
css = '''
@page {size: 960pt 540pt; margin:0}
*{box-sizing:border-box} body{margin:0;background:#ddd;font-family:"NanumGothic",sans-serif;color:#080808}
.slide{width:1280px;height:720px;position:relative;background:white;overflow:hidden;break-after:page}
header{height:82px;background:#1c50b3;color:white;font-size:42px;font-weight:800;padding:15px 27px}
footer{position:absolute;bottom:0;left:24px;right:0;height:32px;display:flex;justify-content:space-between;align-items:end}
footer span{font-size:15px;color:#777;padding-bottom:7px} footer b{font-family:sans-serif;color:#be1551;font-size:30px;font-weight:900}
.coverblue{height:596px;background:#1c50b3;color:white}.coverblue>b{display:block;color:#be1551;font:900 54px sans-serif;border-bottom:2px solid white;padding:12px 18px}
h1{font-size:72px;margin:60px 48px 15px;line-height:1.24}.coverblue>p{font-size:29px;margin-left:50px}.identity{padding:46px 28px;font-size:25px;font-weight:bold}
.contents{margin:75px 0 0 490px}.contents h2{font-size:31px;margin:27px 0 10px}.contents p{font-size:24px;line-height:1.4;margin:0 0 24px 25px}
.divider{text-align:center;margin-top:215px;font-size:67px;font-weight:bold}.divider small{font-size:37px;line-height:2}
.split{display:grid;grid-template-columns:560px 600px;gap:46px;padding:45px 36px 20px;height:590px;align-items:center}
.visual{width:560px;max-height:530px;display:flex;align-items:center;justify-content:center}.visual img{max-width:560px;max-height:510px;object-fit:contain}
aside{border:9px solid #1c50b3;border-radius:20px;overflow:hidden;background:white;width:100%}
aside h3{background:#1c50b3;color:white;margin:0;padding:7px 16px 15px;font-size:23px}aside>div{padding:0 19px 6px}
aside p{font-size:22px;line-height:1.55;font-weight:700;margin:15px 0;overflow-wrap:anywhere}
.code{background:#272822;width:100%;padding:16px;border-radius:2px}.code pre{margin:0;white-space:pre-wrap;word-break:break-all;font-family:"NanumGothicCoding",monospace;font-size:17px;line-height:1.3}
.full{padding:40px 50px}.full aside{margin-top:27px}.full aside p{font-size:22px;margin:9px 0}
table{width:100%;border-collapse:collapse;font-size:23px}th{background:#1c50b3;color:white}th,td{padding:13px 15px;border:1px solid #c1cbdc;text-align:center}tr:nth-child(odd) td{background:#eef3fb}
.pair{display:flex;gap:60px;margin:25px 45px 0;justify-content:center}.pair h3{font-size:25px;text-align:center;margin:0 0 8px}.pair img{height:475px;max-width:540px;object-fit:contain}.caption{text-align:center;font-size:24px;margin:20px 0}
@media screen{.slide{margin:20px auto;box-shadow:0 3px 18px #aaa}}@media print{body{background:white}}
'''
css += HtmlFormatter(style='monokai').get_style_defs('.highlight')
doc = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>강화학습 Week4 HW</title><style>'+css+'</style></head><body>'+''.join(slides)+'</body></html>'
(OUT / '강화학습_week4_2021042040.html').write_text(doc, encoding='utf-8')
print(f'{len(slides)} slides generated')
