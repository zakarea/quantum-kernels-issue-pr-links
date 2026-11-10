import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e5e0','axes.axisbelow':True,'axes.edgecolor':'#8a8980','savefig.dpi':200})
C={'Z':'#2a78d6','ZZ1':'#eb6834','Pauli':'#e87ba4','PQK-Z':'#1baf7a','QSVC-Z':'#2a78d6','QSVC-ZZ1':'#eb6834'}
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
c=pd.read_csv('results_extra/concentration.csv').groupby(['fmap','d','lam']).mean(numeric_only=True).reset_index()
for fm in ['Z','ZZ1','Pauli']:
    g=c[(c.fmap==fm)&(c.lam==1.0)]; ax[0].plot(g.d,g.k_std,marker='o',lw=2,ms=6,color=C[fm],label=f'{fm}, λ = 1')
    g=c[(c.fmap==fm)&(c.lam==0.1)]; ax[0].plot(g.d,g.k_std,marker='s',lw=2,ms=5,color=C[fm],ls=':',label=f'{fm}, λ = 0.1')
ax[0].set_yscale('log'); ax[0].set_xlabel('Qubits'); ax[0].set_title('Kernel spread (SD of off-diagonal values)'); ax[0].legend(frameon=False,fontsize=8,ncol=2)
q=pd.read_csv('results_extra/qubits_results.csv').groupby(['d','model']).mrr.mean().reset_index()
for mo,col,ls in [('QSVC-Z','#2a78d6','--'),('PQK-Z','#1baf7a','--'),('SVC-RBF','#eda100','-'),('LogReg','#5a5950','-')]:
    g=q[q.model==mo]; ax[1].plot(g.d,g.mrr,marker='o',lw=2,ms=6,color=col,ls=ls,label=mo)
ax[1].set_xticks([4,8,12,16,20,24]); ax[1].set_xlabel('Qubits (Z map, exact)'); ax[1].set_title('MRR vs qubits (10 seeds)'); ax[1].legend(frameon=False,fontsize=8)
s=pd.read_csv('results_extra/shots_results.csv'); s=s[s.condition.str.startswith('torino')]
s=s.groupby(['model','shots']).mrr.mean().reset_index()
for mo in ['QSVC-Z','QSVC-ZZ1','PQK-Z']:
    g=s[s.model==mo]; ax[2].plot(g.shots,g.mrr,marker='o',lw=2,ms=6,color=C[mo],ls='--',label=mo)
ax[2].axhline(0.766,color='#eda100',lw=2,label='SVC-RBF (exact)'); ax[2].set_xscale('log'); ax[2].minorticks_off()
ax[2].set_xticks([1024,4096,16384]); ax[2].set_xticklabels(['1,024','4,096','16,384']); ax[2].set_xlabel('Shots per circuit (ibm_torino noise)')
ax[2].set_title('MRR under shots + device noise (d = 12)'); ax[2].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig6_quantum_extra.png')
