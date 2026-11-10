import sys, hashlib, json; sys.path.insert(0,'.')
import numpy as np
from qml_pilink import load_links
from splits import get_split, flat, temporal_links
ref={}
for name,data,split in [('pilink','pilink_pairs_meta.csv','repo'),('ghpr','ghpr_pairs_meta.csv','temporal')]:
    l=load_links(data); lt=temporal_links(l) if split=='temporal' else None; ref[name]={}
    for seed in range(7):
        tr,te,rs,rtr,_=get_split(l,seed,split,links_t=lt); iv,pv=flat(rs)
        ref[name][seed]=hashlib.md5(np.concatenate([te.iss.values,te.pr.values,iv,pv]).astype(np.int64).tobytes()).hexdigest()[:12]
        json.dump(ref,open('split_fingerprints.json','w'),indent=1)
print(ref)
