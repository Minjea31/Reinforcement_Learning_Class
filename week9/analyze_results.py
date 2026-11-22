"""저장 모델의 구간·극값·시드·오차 지표 분석. 추가 학습 없음."""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from format_figures import format_figure
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
FIG=OUT/'figures';S=json.loads((OUT/'summary.json').read_text())
def prediction(row,x):
 c=row['config'];m=RegressionNet(**{k:c[k] for k in ['width','depth','activation','init']},seed=row['seed'])
 z=np.load(OUT/(row['key']+'.npz'))
 for i,p in enumerate(m.params()):p.data=z['p'+str(i)]
 with no_grad():return m(x).data
record={};peaks=((np.arange(6)*2+1)/12).reshape(-1,1)
for label in ['baseline','final']:
 rows=S[label]['runs'];pred=np.array([prediction(r,TEST) for r in rows]);errors=(pred-target(TEST))[:,:,0]
 periods=np.array([[np.mean(e[(TEST[:,0]>=i/3)&(TEST[:,0]<(i+1)/3)]**2) for i in range(3)] for e in errors])
 ext=np.array([prediction(r,peaks)[:,0] for r in rows])
 record[label]=dict(mse=(errors**2).mean(axis=1).tolist(),mae=np.abs(errors).mean(axis=1).tolist(),max_abs=np.abs(errors).max(axis=1).tolist(),period_mse=periods.tolist(),extreme_pred=ext.tolist(),extreme_errors=(ext-target(peaks)[:,0]).tolist(),train=[r['train'] for r in rows],valid=[r['valid'] for r in rows],seconds=[r['seconds'] for r in rows],params=rows[0]['params'])
 for i,e in enumerate(errors):
  if label=='final':plt.plot(TEST[:,0],e,label=f'Seed {i}',lw=1.6)
 if label=='final':
  plt.axhline(0,c='k',ls='--');plt.xlabel('x');plt.ylabel('Prediction - true target');plt.legend();format_figure(plt.gcf());plt.tight_layout();plt.savefig(FIG/'residual.png',dpi=180);plt.close()
plt.figure(figsize=(6,5.1))
for i,label in enumerate(['baseline','final']):
 vals=np.array(record[label]['period_mse']);plt.bar(np.arange(3)+(i-.5)*.3,vals.mean(0),.3,yerr=vals.std(0),capsize=4,label=label.title())
plt.xticks(range(3),['[0, 1/3)','[1/3, 2/3)','[2/3, 1]']);plt.xlabel('Period interval');plt.ylabel('Clean test MSE (mean ± SD)');plt.legend();format_figure(plt.gcf());plt.tight_layout();plt.savefig(FIG/'period_error.png',dpi=180);plt.close()
plt.figure(figsize=(6,5.1))
for label in ['baseline','final']:plt.plot(range(3),record[label]['mse'],'o-',label=label.title())
plt.xticks(range(3),['0','1','2']);plt.xlabel('Seed');plt.ylabel('Clean test MSE');plt.legend();format_figure(plt.gcf());plt.tight_layout();plt.savefig(FIG/'seed_error.png',dpi=180);plt.close()
plt.figure(figsize=(6,5.1))
for label in ['baseline','final']:
 for kind in ['train','valid']:
  rows=S[label]['runs'];h=np.array([r['history'] for r in rows]);j=1 if kind=='train' else 2
  plt.plot(h[0,:,0],h[:,:,j].mean(0),label=f'{label.title()} {kind}',ls='--' if kind=='train' else '-')
plt.yscale('log');plt.xlabel('Updates');plt.ylabel('MSE (3-seed mean)');plt.legend(fontsize=9);format_figure(plt.gcf());plt.tight_layout();plt.savefig(FIG/'loss_compare.png',dpi=180);plt.close()
record['extreme_x']=peaks[:,0].tolist();record['extreme_target']=target(peaks)[:,0].tolist()
(OUT/'detailed_analysis.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
