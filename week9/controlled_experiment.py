"""최종 후보 주변 한 번에 한 조건만 변경. 그룹별 모든 후보를 실제 DeZero 학습.
선택은 검증 3시드 평균. 최종 조합은 실제 실행된 단일 후보 중 선택해 미실험 조합 주장 방지.
"""
from experiment import *

def main():
    original=json.loads((OUT/'summary.json').read_text())
    reference=original['final']['config'].copy()
    definitions=[('width','width',[10,50,100]),('depth','depth',[1,2,3]),
      ('optimizer','optimizer',['SGD','MomentumSGD','AdaGrad','Adam']),
      ('activation','activation',['sigmoid','tanh','relu']),
      ('initialization','init',['std1','std001','xavier','he']),
      ('learning_rate','lr',[.0001,.001,.01,.1]),
      ('decay','decay',[0.,.0001,.001,.01]),('updates','steps',[1000,3000,6000,9900])]
    groups=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as pool:
        for name,key,values in definitions:
            rows=evaluate(pool,[dict(reference,**{key:v}) for v in values])
            best=min((r for r in rows if r['mean'] is not None),key=lambda r:(r['mean'],r['runs'][0]['params']))
            groups.append(dict(name=name,key=key,rows=rows,winner=best['id']))
            print(name,[(r['config'][key],r['mean']) for r in rows],flush=True)
            (OUT/'controlled_progress.json').write_text(json.dumps(groups,indent=2))
    # 항목별 최저 조건을 실제 조합해 다시 학습하고, 단일 변경 후보와 비교한다.
    combined_config=reference.copy()
    for g in groups:
        winner=next(r for r in g['rows'] if r['id']==g['winner'])
        combined_config[g['key']]=winner['config'][g['key']]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as pool:
        combined=evaluate(pool,[combined_config])[0]
    candidates=[r for g in groups for r in g['rows'] if r['mean'] is not None]
    if combined['mean'] is not None:candidates.append(combined)
    best=min(candidates,key=lambda r:(r['mean'],r['runs'][0]['params']))
    tests=[]
    for label,row in [('baseline',original['baseline']),('final',best)]:
        for r in row['runs']:
            net=RegressionNet(**{k:r['config'][k] for k in ['width','depth','activation','init']},seed=r['seed'])
            z=np.load(OUT/(r['key']+'.npz'))
            for i,p in enumerate(net.params()):p.data=z['p'+str(i)]
            with no_grad(): pred=net(TEST).data
            tests.append(dict(label=label,seed=r['seed'],mse=float(np.mean((pred-target(TEST))**2)),edge_mse=float(np.mean((pred-target(TEST))[[*range(100),*range(900,1000)]]**2))))
    record=dict(reference=reference,groups=groups,combined=combined,final=best,tests=tests)
    (OUT/'controlled_summary.json').write_text(json.dumps(record,indent=2))
    report=original.copy();report.update(final=best,chosen_step=best['config']['steps'],tests=tests,controlled_groups=groups,reference=reference)
    (OUT/'summary_v4.json').write_text(json.dumps(report,indent=2))
    print('FINAL',best['config'],best['mean'],tests,flush=True)
if __name__=='__main__':main()
