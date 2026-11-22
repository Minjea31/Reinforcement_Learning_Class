"""최종 설정으로 100개 학습점을 새로 학습하고 결과를 저장하는 제출용 진입점."""
from experiment import *
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    c=json.loads((OUT/'summary_v4.json').read_text())['final']['config']
    x,y=data(args.seed)
    model=RegressionNet(**{k:c[k] for k in ['width','depth','activation','init']},seed=args.seed)
    optimizer=optimizers.Adam(alpha=c['lr']).setup(model)
    for step in range(c['steps']):
        pred=model(x)
        loss=F.mean_squared_error(y,pred)
        for layer in model.layers: loss=loss+.5*c['decay']*F.sum(layer.W**2)
        model.cleargrads();loss.backward();optimizer.update()
        if step%1000==0: print(f'update={step} objective={float(loss.data):.6f}')
    with no_grad():
        pred=model(TEST).data
        print('final clean test MSE:',float(np.mean((pred-target(TEST))**2)))
    np.savez(OUT/f'final_seed{args.seed}.npz',**{'p'+str(i):p.data for i,p in enumerate(model.params())})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.scatter(x,y,s=12,label='100 noisy training samples');plt.plot(TEST,target(TEST),'k--',label='sin(6πx)')
    plt.plot(TEST,pred,label='DeZero prediction');plt.xlabel('x');plt.ylabel('y');plt.legend();plt.tight_layout()
    plt.savefig(OUT/f'final_seed{args.seed}.png',dpi=180)
if __name__=='__main__':main()
