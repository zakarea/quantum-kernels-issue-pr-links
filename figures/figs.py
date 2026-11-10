import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e5e0','axes.axisbelow':True,'axes.edgecolor':'#8a8980','savefig.dpi':200})
df=pd.read_csv('results/raw_results.csv'); m=df.groupby(['d','model']).mean(numeric_only=True).reset_index()
# Fig1: quantum vs classical by d
mods=['QSVC-Z','QSVC-ZZ1','SVC-RBF','LogReg','XGB']
fig,ax=plt.subplots(1,3,figsize=(11,3.4))
for k,met in enumerate(['f1','auc','mrr']):
    for i,mo in enumerate(mods):
        s=m[m.model==mo]; ax[k].plot(s.d,s[met],marker='o',ms=6,lw=2,color=C[i],label=mo,ls='--' if mo.startswith('QSVC') else '-')
    ax[k].set_title(met.upper()); ax[k].set_xticks([4,8,12]); ax[k].set_xlabel('Reduced dimension d (= qubits)')
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,fontsize=9); fig.suptitle('RQ1: Quantum kernels vs classical at equal dimension (7 seeds, repo-level split)'); fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig('figs/fig1_rq1.png')
# Fig2: feature maps
fm=['QSVC-Z','QSVC-Pauli','QSVC-ZZ1','QSVC-ZZ2']
fig,ax=plt.subplots(figsize=(6.5,3.4)); w=.2
for i,f in enumerate(fm):
    s=m[m.model==f]; ax.bar(s.d+(i-1.5)*0.8,s.mrr,width=0.8,color=C[i],label=f.replace('QSVC-',''),edgecolor='white',lw=2)
ax.set_xticks([4,8,12]); ax.set_xlabel('Qubits'); ax.set_ylabel('MRR'); ax.legend(frameon=False,title='Feature map',ncol=4,fontsize=8)
ax.set_title('RQ2: Entangling feature maps hurt ranking, and more qubits make it worse'); fig.tight_layout(); fig.savefig('figs/fig2_rq2.png')
# Fig3: scaling
rows=[]
for n in [200,400,800,1600,3200]:
    r=pd.read_csv(f'scaling/n{n}/raw_results.csv'); r['n']=n; rows.append(r)
s=pd.concat(rows).groupby(['n','model']).mean(numeric_only=True).reset_index()
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
for i,mo in enumerate(['QSVC-Z','QSVC-ZZ1','SVC-RBF','XGB']):
    t=s[s.model==mo]
    if t.empty: continue
    ls='--' if mo.startswith('QSVC') else '-'
    ax[0].plot(t.n,t.auc,marker='o',lw=2,ms=6,color=C[[0,1,2,4][i]],ls=ls,label=mo)
    if mo!='XGB': ax[1].plot(t.n,t.fit_s,marker='o',lw=2,ms=6,color=C[[0,1,2,4][i]],ls=ls,label=mo)
for a in ax: a.set_xscale('log'); a.set_xticks([200,400,800,1600,3200]); a.set_xticklabels([200,400,800,1600,3200]); a.minorticks_off(); a.set_xlabel('Training pairs')
ax[0].set_title('AUC vs training size (d=8)'); ax[1].set_yscale('log'); ax[1].minorticks_off(); ax[1].set_title('Fit time, seconds (log)'); ax[0].legend(frameon=False,fontsize=8)
fig.suptitle('RQ4: Scaling (3 seeds)'); fig.tight_layout(); fig.savefig('figs/fig3_rq4.png')
print(s[['n','model','f1','auc','fit_s']].round(3).to_string())
