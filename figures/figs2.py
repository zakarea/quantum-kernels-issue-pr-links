import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'.')
C=['#2a78d6','#eb6834','#1baf7a','#eda100']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e5e0','axes.axisbelow':True,'axes.edgecolor':'#8a8980','savefig.dpi':200})
cb=pd.read_csv('../in_cb/results_codebert/raw_results.csv'); tf=pd.read_csv('results/raw_results.csv')
mods=['QSVC-Z','QSVC-ZZ1','SVC-RBF','LogReg','XGB']
fig,ax=plt.subplots(1,2,figsize=(10,3.6),sharey=False)
for k,met in enumerate(['auc','mrr']):
    x=np.arange(len(mods))
    for i,(nm,df) in enumerate([('TF-IDF',tf),('CodeBERT',cb)]):
        m=df[df.d==8].groupby('model')[met].agg(['mean','std']).reindex(mods)
        ax[k].bar(x+(i-.5)*.38,m['mean'],.38,yerr=m['std'],color=C[i],label=nm,edgecolor='white',lw=1.5,error_kw=dict(lw=1,ecolor='#5a5950'))
    ax[k].set_xticks(x); ax[k].set_xticklabels(mods,fontsize=9); ax[k].set_title(met.upper()+' at d = 8 qubits')
    ax[k].set_ylim(0.5 if met=='auc' else 0.2, None)
ax[0].legend(frameon=False)
fig.suptitle('Feature robustness: TF-IDF vs CodeBERT (7 seeds, repo-level split; bars = mean, whiskers = SD)'); fig.tight_layout(); fig.savefig('figs/fig4_codebert.png')
