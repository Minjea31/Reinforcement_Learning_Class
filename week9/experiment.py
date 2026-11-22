"""Quiz Q1: PDF pp.49–54 기반. 이론/예제 외부 참조 없음.
공식 DeZero vendor는 실행 의존성. python experiment.py 로 전체 재현.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, time, hashlib, warnings, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/deep-learning-from-scratch-3'))
from dezero import Model, no_grad
import dezero.layers as L
import dezero.functions as F
from dezero import optimizers
warnings.filterwarnings('ignore',category=RuntimeWarning)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
VALID=np.linspace(0,1,401).reshape(-1,1)
TEST=(np.arange(1,1001)/1001).reshape(-1,1)
def target(x): return np.sin(6*np.pi*x)
def data(seed):
    rng=np.random.RandomState(seed)
    x=rng.rand(100,1)
    y=target(x)+rng.randn(100,1)  # 강의 52쪽: noise std=1
    assert x.shape==y.shape==(100,1) and np.all((x>=0)&(x<=1))
    return x,y
class RegressionNet(Model):
    def __init__(self,width,depth,activation,init,seed):
        super().__init__()
        self.activation=activation
        rng=np.random.RandomState(seed+100)
        sizes=[1]+[width]*depth+[1]
        self.layers=[]
        for i,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
            layer=L.Linear(m,in_size=n,dtype=np.float64)
            scale={'xavier':np.sqrt(1/n),'he':np.sqrt(2/n),'std1':1.,'std001':.01}[init]
            layer.W.data=rng.randn(n,m)*scale
            setattr(self,'layer'+str(i),layer)
            self.layers.append(layer)
    def params(self):
        for layer in self.layers:
            yield layer.W
            yield layer.b
    def forward(self,x):
        for layer in self.layers[:-1]:
            x=getattr(F,self.activation)(layer(x))
        return self.layers[-1](x)  # 회귀 출력: 선형

def identifier(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
def run(task):
    c,seed=task; key=identifier(c)+'_'+str(seed); path=OUT/(key+'.json')
    if path.exists(): return json.loads(path.read_text())
    x,y=data(seed); model=RegressionNet(c['width'],c['depth'],c['activation'],c['init'],seed)
    opt=getattr(optimizers,c['optimizer'])(**({'alpha':c['lr']} if c['optimizer']=='Adam' else {'lr':c['lr']}))
    opt.setup(model)
    history=[]; start=time.perf_counter(); status='ok'
    best=(float('inf'),0,None)
    for step in range(c['steps']+1):
        if step%100==0 or step==c['steps']:
            with no_grad():
                train=float(np.mean((model(x).data-y)**2))
                valid=float(np.mean((model(VALID).data-target(VALID))**2))
            if not np.isfinite(train+valid) or train>1e12:
                status='diverged'; break
            history.append([step,train,valid])
            if valid<best[0]: best=(valid,step,[p.data.copy() for p in model.params()])
        if step==c['steps']: break
        pred=model(x)
        loss=F.mean_squared_error(y,pred)
        if c['decay']:
            for layer in model.layers: loss=loss+.5*c['decay']*F.sum(layer.W**2)
        model.cleargrads()
        loss.backward()
        opt.update()
    # 모든 조건은 마지막 시점 기준 비교. 체크포인트 선택은 별도 마지막 단계에서만.
    result=dict(config=c,seed=seed,key=key,status=status,seconds=time.perf_counter()-start,
                params=sum(p.data.size for p in model.params()),history=history,
                train=history[-1][1] if status=='ok' else None,
                valid=history[-1][2] if status=='ok' else None,best_step=best[1],best_valid=best[0])
    np.savez(OUT/(key+'.npz'),**{'p'+str(i):p.data for i,p in enumerate(model.params())})
    np.savez(OUT/(key+'_best.npz'),**{'p'+str(i):p for i,p in enumerate(best[2])})
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

def evaluate(pool,configs):
    unique={identifier(c):c for c in configs}
    runs=list(pool.map(run,[(c,s) for c in unique.values() for s in [0,1,2]]))
    rows=[]
    for k,c in unique.items():
        rr=[r for r in runs if identifier(r['config'])==k]; good=all(r['status']=='ok' for r in rr)
        rows.append(dict(config=c,id=k,mean=float(np.mean([r['valid'] for r in rr])) if good else None,
                         std=float(np.std([r['valid'] for r in rr])) if good else None,runs=rr))
    return rows

def main():
    base=dict(width=10,depth=1,activation='sigmoid',init='xavier',optimizer='SGD',lr=.2,steps=10000,decay=0.)
    stages=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context("fork")) as pool:
        baseline=evaluate(pool,[base])[0]
        current=base.copy()
        def stage(name,configs):
            nonlocal current
            rows=evaluate(pool,configs)
            win=min((r for r in rows if r['mean'] is not None),key=lambda r:(r['mean'],r['runs'][0]['params']))
            current=win['config'].copy()
            stages.append(dict(name=name,rows=rows,winner=win['id']))
            (OUT/'progress.json').write_text(json.dumps(dict(baseline=baseline,stages=stages),indent=2))
            print(name, 'selected', current,'MSE',round(win['mean'],6),flush=True)
        stage('width',[dict(current,width=w) for w in [10,50,100]])
        stage('depth',[dict(current,depth=d) for d in [1,2,3]])
        stage('activation',[dict(current,activation=a) for a in ['sigmoid','tanh','relu']])
        stage('optimizer',[dict(current,optimizer=o,lr=l) for o in ['SGD','MomentumSGD','AdaGrad','Adam'] for l in [.0001,.001,.01,.1]])
        # sigmoid default SGD may hide expressive power; joint activation/optimizer revisit controls order effect.
        stage('activation_recheck',[dict(current,activation=a) for a in ['sigmoid','tanh','relu']])
        stage('initialization',[dict(current,activation=a,init=i) for a in ['tanh','relu'] for i in ['std1','std001','xavier','he']])
        stage('decay',[dict(current,decay=d) for d in [0.,.0001,.001,.01]])
        # common stopping point chosen on mean validation, same epoch across seeds
        finalrows=evaluate(pool,[current]); fr=finalrows[0]
        hs=[np.array(r['history']) for r in fr['runs']]
        idx=int(np.argmin(np.mean([h[:,2] for h in hs],axis=0)))
        chosen_step=int(hs[0][idx,0])
        # retrain exactly chosen updates so every seed uses same validation-selected step
        final=evaluate(pool,[dict(current,steps=chosen_step)])[0]
        tests=[]
        for label,row in [('baseline',baseline),('final',final)]:
            for r in row['runs']:
                net=RegressionNet(**{k:r['config'][k] for k in ['width','depth','activation','init']},seed=r['seed'])
                weights=np.load(OUT/(r['key']+'.npz'))
                for i,p in enumerate(net.params()): p.data=weights['p'+str(i)]
                with no_grad(): pred=net(TEST).data
                tests.append(dict(label=label,seed=r['seed'],mse=float(np.mean((pred-target(TEST))**2)),edge_mse=float(np.mean((pred-target(TEST))[[*range(100),*range(900,1000)]]**2))))
    record=dict(baseline=baseline,stages=stages,final=final,chosen_step=chosen_step,tests=tests,
                validation='401 inclusive grid points',test='1000 interior grid points i/1001, disjoint from validation',seeds=[0,1,2],noise_std=1.)
    (OUT/'summary.json').write_text(json.dumps(record,indent=2))
    print('DONE',chosen_step,tests,flush=True)
if __name__=='__main__': main()
