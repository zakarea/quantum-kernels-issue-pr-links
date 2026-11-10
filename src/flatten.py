"""Flatten PI-Link JSON -> one row per (issue, first linked PR), with leakage cues removed."""
import json, re, sys
import pandas as pd

LEAK = re.compile(r"https?://\S+|\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|ref(?:s)?|see)\b\s*#\d+|#\d+|\bgh-\d+\b",
                  re.I)
clean = lambda s: re.sub(r"\s+", " ", LEAK.sub(" ", s or "")).strip()

src, dst = sys.argv[1], sys.argv[2]
rows = []
for it in json.load(open(src)):
    prs = [n["pull_info"] for n in (it.get("timelineItems") or {}).get("nodes", []) if n.get("pull_info")]
    if not prs:
        continue
    pr = prs[0]
    commits = " ".join(c.get("commit", {}).get("commit_message", "") for c in (pr.get("commits") or {}).get("nodes", []))
    rows.append(dict(
        repo=it["issue_repository"]["nameWithOwner"],
        issue_no=it["issue_no"], pr_number=pr.get("pull_no"),
        created_at=it.get("created_at"),
        issue_title=clean(it.get("issue_title")), issue_body=clean(it.get("issue_body"))[:3000],
        pr_title=clean(pr.get("pull_title")),
        pr_body=(clean(pr.get("pull_body"))[:2000] + " " + clean(commits)[:1000]).strip(),
        n_linked_prs=len(prs)))
df = pd.DataFrame(rows)
df.to_csv(dst, index=False)
print(len(df), "links,", df.repo.nunique(), "repos;", (df.groupby("repo").size() >= 3).sum(), "repos with >=3 links")
