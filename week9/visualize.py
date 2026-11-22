"""저장된 실험 결과만 사용하여 그래프와 분석용 통계를 생성."""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from format_figures import format_figure
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
S=json.loads((OUT/'summary.json').read_text())
def save(name):
    format_figure(plt.gcf());plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=180);plt.close()
def plot_net(r,xx):
    c=r['config'];net=RegressionNet(**{k:c[k] for k in ['width','depth','activation','init']},seed=r['seed'])
    z=np.load(OUT/(r['key']+'.npz'))
    for i,p in enumerate(net.params()):p.data=z['p'+str(i)]
    with no_grad():return net(xx).data
xx=np.linspace(0,1,1001).reshape(-1,1)
for name,row in [('baseline',S['baseline']),('final',S['final'])]:
    plt.figure(figsize=(6,5.1));x,y=data(0);plt.scatter(x,y,s=13,c='#999999',alpha=.65,label='Training data (n=100)')
    plt.plot(xx,target(xx),'k--',lw=2,label='True sin(6πx)')
    for r in row['runs']:plt.plot(xx,plot_net(r,xx),lw=2 if r['seed']==0 else 1,alpha=1 if r['seed']==0 else .5,label='Prediction seed '+str(r['seed']))
    plt.xlabel('x');plt.ylabel('y');plt.legend(fontsize=9);save(name)
    plt.figure(figsize=(6,5.1))
    for j,label in [(1,'Training MSE (noisy)'),(2,'Validation MSE (clean)')]:
        h=np.array([r['history'] for r in row['runs']]);v=h[:,:,j]
        plt.plot(h[0,:,0],v.mean(0),label=label);plt.fill_between(h[0,:,0],np.maximum(v.mean(0)-v.std(0),1e-6),v.mean(0)+v.std(0),alpha=.15)
    plt.yscale('log');plt.xlabel('Updates');plt.ylabel('MSE (mean ± SD, 3 seeds)');plt.legend(fontsize=9);save(name+'_loss')
for stage in S['stages']:
    name=stage['name'];rows=stage['rows']
    if name=='optimizer':
        plt.figure(figsize=(6,5.1))
        for o in ['SGD','MomentumSGD','AdaGrad','Adam']:
            rr=[r for r in rows if r['config']['optimizer']==o];rr.sort(key=lambda r:r['config']['lr'])
            plt.plot([r['config']['lr'] for r in rr],[r['mean'] if r['mean'] is not None else np.nan for r in rr],marker='o',label=o.replace('SGD','SGD'))
        plt.xscale('log');plt.yscale('log');plt.xlabel('Learning rate');plt.ylabel('Clean validation MSE');plt.legend(fontsize=9)
    else:
        keys={'width':'width','depth':'depth','activation':'activation','activation_recheck':'activation','decay':'decay'}
        labels=[str(r['config'][keys[name]]) if name in keys else r['config']['activation']+'\n'+r['config']['init'] for r in rows]
        plt.figure(figsize=(6,5.1));v=[r['mean'] if r['mean'] is not None else np.nan for r in rows]
        plt.scatter(np.arange(len(rows)),v,s=65,color=['#1c50b3' if r['id']==stage['winner'] else '#a8bddf' for r in rows],zorder=3)
        plt.errorbar(np.arange(len(rows)),v,yerr=[r['std'] or 0 for r in rows],fmt='none',ecolor='#333333',capsize=3)
        plt.xticks(range(len(rows)),labels,fontsize=8 if len(rows)>5 else 11)
        for i,r in enumerate(rows):
            if r['mean'] is None: plt.text(i,.02,'diverged',rotation=90,ha='center',transform=plt.gca().get_xaxis_transform())
        plt.yscale('log');plt.ylabel('Clean validation MSE (mean ± SD)')
    save(name)
# Full-budget final candidate learning curve includes selected common stopping point.
row=S['stages'][-1]['rows'];row=next(r for r in row if r['id']==S['stages'][-1]['winner'])
h=np.array([r['history'] for r in row['runs']]);plt.figure(figsize=(6,5.1))
for j,l in [(1,'Training (noisy)'),(2,'Validation (clean)')]:plt.plot(h[0,:,0],h[:,:,j].mean(0),label=l)
plt.axvline(S['chosen_step'],c='#be1551',ls='--',label='Selected update')
plt.yscale('log');plt.xlabel('Updates');plt.ylabel('MSE (mean of 3 seeds)');plt.legend(fontsize=9);save('stopping')
print('FIGURES',len(list(FIG.glob('*.png'))))
