"""Issue/PR timestamps, in the same row order as flatten.py, merged into pilink_pairs_meta.csv."""
import json, sys
import pandas as pd
src, pairs_csv, dst = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for it in json.load(open(src)):
    prs = [n["pull_info"] for n in (it.get("timelineItems") or {}).get("nodes", []) if n.get("pull_info")]
    if not prs:
        continue
    pr = prs[0]
    rows.append(dict(t_issue_open=it.get("created_at"), t_issue_close=it.get("closed_at"),
                     t_pr_open=pr.get("createdAt"), t_pr_close=pr.get("closedAt"), _pr=pr.get("pull_no"), _iss=it["issue_no"]))
t = pd.DataFrame(rows)
df = pd.read_csv(pairs_csv, low_memory=False)
assert len(df) == len(t) and (df.issue_no.values == t._iss.values).all() and (df.pr_number.values == t._pr.values).all()
for c in [c for c in t if c.startswith("t_")]:
    df[c] = (pd.to_datetime(t[c], utc=True, errors="coerce") - pd.Timestamp("2011-01-01", tz="UTC")).dt.total_seconds().values / 86400
df.to_csv(dst, index=False)
print(len(df), df[[c for c in df if c.startswith("t_")]].isna().mean().round(3).to_dict())
d = df.t_pr_open - df.t_issue_open
print("PR open - issue open (days): median", d.median().round(2), "share PR after issue", (d > 0).mean().round(3))
print("|PR close - issue close| < 1h share", ((df.t_pr_close - df.t_issue_close).abs() < 1/24).mean().round(3))
