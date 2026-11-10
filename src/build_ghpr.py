"""Build GHPR (CNCF projects) links in the same schema as pilink_pairs_meta.csv.

One row per (issue, first linking PR). Same leakage cleaning as flatten.py
(#N references, URLs, closing keywords). Timestamps in days since 2011-01-01.
Note: GHPR raw data has PR title + description but no commit messages.
  python -I build_ghpr.py /home/claude/ext/ghpr-raw/repos ghpr_pairs_meta.csv
"""
import json, re, sys
from pathlib import Path
import pandas as pd

LEAK = re.compile(r"https?://\S+|\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|ref(?:s)?|see)\b\s*#\d+|#\d+|\bgh-\d+\b", re.I)
clean = lambda s: re.sub(r"\s+", " ", LEAK.sub(" ", s or "")).strip()
T0 = pd.Timestamp("2011-01-01", tz="UTC")
days = lambda s: (pd.Timestamp(s) - T0).total_seconds() / 86400 if s else float("nan")

root = Path(sys.argv[1]); rows = []
for repo_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
    repo = f"{repo_dir.parent.name}/{repo_dir.name}"
    pulls = {}
    for f in repo_dir.glob("pull-*.json"):
        p = json.load(open(f))
        for i in p.get("linked_issue_numbers") or []:
            if i not in pulls or p["number"] < pulls[i]["number"]:
                pulls[i] = p  # first (lowest-numbered) PR that fixed issue i
    for i, p in pulls.items():
        f = repo_dir / f"issue-{i}.json"
        if not f.exists():
            continue
        it = json.load(open(f))
        rows.append(dict(repo=repo, issue_no=i, pr_number=p["number"], created_at=it.get("created_at"),
                         issue_title=clean(it.get("title")), issue_body=clean(it.get("body"))[:3000],
                         pr_title=clean(p.get("title")), pr_body=clean(p.get("body"))[:3000],
                         t_issue_open=days(it.get("created_at")), t_issue_close=days(it.get("closed_at")),
                         t_pr_open=days(p.get("created_at")), t_pr_close=days(p.get("merged_at") or p.get("closed_at"))))
df = pd.DataFrame(rows).sort_values(["repo", "pr_number", "issue_no"]).reset_index(drop=True)
df.to_csv(sys.argv[2], index=False)
d = df.t_pr_open - df.t_issue_open
print(len(df), "links,", df.repo.nunique(), "repos;", df.groupby("repo").size().to_dict())
print("PR opened after issue:", round((d > 0).mean(), 3), "median gap days", round(d.median(), 2),
      "| PR merged within 1h of issue close:", round(((df.t_pr_close - df.t_issue_close).abs() < 1 / 24).mean(), 3))
