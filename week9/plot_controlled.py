"""한 조건씩 변경한 실제 학습 함수와 원래 sin 함수를 비교. 곡선=seed0, 통계=3seeds."""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from format_figures import format_figure
S=json.loads((OUT/'controlled_summary.json').read_text());G={g['name']:g for g in S['groups']}
FIG=OUT/'figures_v4';FIG.mkdir(exist_ok=True)
X=np.linspace(0,1,501).reshape(-1,1)
def prediction(r,x):
    c=r['config'];m=RegressionNet(**{k:c[k] for k in ['width','depth','activation','init']},seed=r['seed'])
    z=np.load(OUT/(r['key']+'.npz'))
    for i,p in enumerate(m.params()):p.data=z['p'+str(i)]
    with no_grad():return m(x).data
names={'std1':'std=1','std001':'std=0.010','xavier':'Xavier','he':'He','MomentumSGD':'Momentum'}
def label(g,r):
    v=r['config'][g['key']]
    if isinstance(v,float):return f'{v:.0e}' if v and v<.001 else f'{v:.3f}'
    return names.get(str(v),str(v))
def draw(ax,name):
    g=G[name];ax.plot(X,target(X),'k--',lw=2,label='True sin(6πx)')
    for i,r in enumerate(g['rows']):
        rr=next(t for t in r['runs'] if t['seed']==0)
        if rr['status']!='ok':continue
        ax.plot(X,prediction(rr,X),lw=1.5,label=label(g,r),ls=['-',':','-.','--'][i%4])
    ax.set_xlabel('x');ax.set_ylabel('y');ax.tick_params(labelsize=9)
    ax.legend(fontsize=8,ncol=2);ax.spines[['top','right']].set_visible(False)
    ax.set_title({'width':'Hidden nodes','depth':'Hidden layers','optimizer':'Optimizer','activation':'Activation','initialization':'Initialization','learning_rate':'Learning rate','decay':'Weight decay λ'}[name],fontsize=11)
for name in ['optimizer','activation','initialization']:
    fig,ax=plt.subplots(figsize=(6,5.1));draw(ax,name);format_figure(fig);fig.tight_layout();fig.savefig(FIG/f'controlled_{name}.png',dpi=180);plt.close(fig)
for name,groups in [('structure',['width','depth']),('other',['learning_rate','decay'])]:
    fig,axs=plt.subplots(2,1,figsize=(6,5.1))
    for ax,g in zip(axs,groups):draw(ax,g)
    format_figure(fig);fig.tight_layout();fig.savefig(FIG/f'controlled_{name}.png',dpi=180);plt.close(fig)
print('Saved five function comparison graphs')
