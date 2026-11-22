"""결과 기록과 체크포인트의 실제 예측값 대조. 학습 데이터 100개 조건 확인."""
from experiment import *
def main():
    assert not np.any(np.isclose(VALID[:,None,:],TEST[None,:,:],atol=1e-12,rtol=0))
    summary=json.loads((OUT/'summary_v4.json').read_text())
    for label in ['baseline','final']:
        for r in summary[label]['runs']:
            c=r['config'];net=RegressionNet(**{k:c[k] for k in ['width','depth','activation','init']},seed=r['seed'])
            z=np.load(OUT/(r['key']+'.npz'))
            for i,p in enumerate(net.params()):p.data=z['p'+str(i)]
            x,y=data(r['seed'])
            assert len(x)==100 and x.shape==y.shape==(100,1)
            with no_grad():
                pred=net(TEST).data
                valid=float(np.mean((net(VALID).data-target(VALID))**2))
            expected=next(t for t in summary['tests'] if t['label']==label and t['seed']==r['seed'])
            assert pred.shape==(1000,1) and np.isfinite(pred).all()
            assert np.isclose(np.mean((pred-target(TEST))**2),expected['mse'],rtol=1e-10)
            assert np.isclose(valid,r['valid'],rtol=1e-10)
    controlled=json.loads((OUT/'controlled_summary.json').read_text())
    for group in controlled['groups']:
        for row in group['rows']:
            changed=[k for k,v in row['config'].items() if v!=controlled['reference'][k]]
            assert all(k==group['key'] for k in changed)
            for rec in row['runs']:
                if rec['status']!='ok':continue
                c=rec['config'];net=RegressionNet(**{k:c[k] for k in ['width','depth','activation','init']},seed=rec['seed'])
                z=np.load(OUT/(rec['key']+'.npz'))
                for i,p in enumerate(net.params()):p.data=z['p'+str(i)]
                with no_grad():actual=float(np.mean((net(VALID).data-target(VALID))**2))
                assert np.isclose(actual,rec['valid'],rtol=1e-10)
    # 강의 45쪽 자동미분 예제 확인
    from dezero import Variable
    x=Variable(np.array(5.)); y=3*x**2;y.backward();assert x.grad.data==30
    print('PASS: final/baseline + all controlled checkpoints, one-factor changes, 100 training samples, metrics and autograd')
if __name__=='__main__':main()
