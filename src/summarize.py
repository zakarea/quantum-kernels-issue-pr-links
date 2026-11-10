import ast,re,pandas as pd,numpy as np
from scipy.stats import wilcoxon
rows=[]
for l in open('results.log'):
    l=l.strip()
    if l.startswith("{'seed'"):
        rows.append(ast.literal_eval(re.sub(r'np\.float64\(([^)]*)\)',r'\1',l)))
df=pd.DataFrame(rows); df=df[df.seed<=6]
df.to_csv('results/raw_results.csv',index=False)
m=['precision','recall','f1','auc','hit1','hit5','mrr','kernel_mae','kernel_corr','fit_s','infer_s']
m=[c for c in m if c in df]
s=df.groupby(['d','model'])[m].agg(['mean','std']).round(3)
s.to_csv('results/summary.csv')
pd.set_option('display.width',250,'display.max_rows',200)
print(df.groupby(['d','model'])[['f1','auc','mrr','hit1','hit5','fit_s']].mean().round(3))
print(df.groupby(['d','model']).size().unstack(0))
def cd(a,b):
    a,b=np.array(a),np.array(b); return ((a[:,None]>b).sum()-(a[:,None]<b).sum())/(len(a)*len(b))
out=[]
for d in sorted(df.d.unique()):
    sub=df[df.d==d]; q=sub[sub.model.str.startswith('QSVC')&~sub.model.str.contains('RQ3')]
    if q.empty: continue
    for metric in ['f1','auc','mrr']:
        best=q.groupby('model')[metric].mean().idxmax()
        for base in ['SVC-RBF','LogReg','XGB','SVC-RBF(full)']:
            a=sub[sub.model==best].sort_values('seed')[metric].values; b=sub[sub.model==base].sort_values('seed')[metric].values
            if len(b)!=len(a) or len(a)==0: continue
            p=wilcoxon(a,b).pvalue if np.any(a!=b) else 1
            out.append(dict(d=d,metric=metric,quantum=best,baseline=base,q_mean=a.mean().round(3),c_mean=b.mean().round(3),p=round(p,4),cliffs=round(cd(a,b),3)))
st=pd.DataFrame(out); st.to_csv('results/stat_tests.csv',index=False); print(st.to_string())
