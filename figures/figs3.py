import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C={'QSVC-Z(untuned)':'#8a8980','QSVC-Z':'#2a78d6','PQK-Z':'#1baf7a','QSVC-ZZ1':'#e87ba4','SVC-RBF':'#eb6834','LogReg':'#eda100','XGB':'#5a5950'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e5e0','axes.axisbelow':True,'axes.edgecolor':'#8a8980','savefig.dpi':200})
df=pd.read_csv('results_enh/raw_results_all.csv'); m=df.groupby(['setting','d','model']).mean(numeric_only=True).reset_index()
fig,ax=plt.subplots(1,3,figsize=(12,3.8))
for k,(s,t) in enumerate([('PCA','Main features, tuned'),('PLS','PLS reduction'),('PLS+meta','PLS + open-time metadata')]):
    for mo,c in C.items():
        g=m[(m.setting==s)&(m.model==mo)]
        ax[k].plot(g.d,g.mrr,marker='o',ms=6,lw=2,color=c,ls='--' if mo.startswith(('QSVC','PQK')) else '-',label=mo)
    ax[k].set_title(t); ax[k].set_xticks([4,8,12]); ax[k].set_xlabel('Qubits / dimension d')
ax[0].set_ylabel('MRR (ranking among 20 PRs)')
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=7,frameon=False,fontsize=9)
fig.suptitle('Enhancements: with tuned bandwidth, quantum kernels match the best classical models (dashed = quantum, 7 seeds)')
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig('figs/fig5_enhanced.png')
