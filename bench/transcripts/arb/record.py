import json
from pathlib import Path

from bench import arb
from bench.run import Usage
from repo2txt import rank
from repo2txt.api import repo_select

data = Path("/srv/dev/arb/data")
rows = [json.loads(line) for line in open("bench/results/arb/repo2txt-llm.jsonl")]
pick = []
for task in ("code2test", "comment2context"):
    misses = [r for r in rows if r["task_type"] == task and r["metrics"]["Recall@5"] == 0]
    pick += sorted(misses, key=lambda r: r["sample_id"])[:2]
samples = {}
for subset in arb.SUBSETS:
    for line in (data / "benchmark" / subset / "samples.jsonl").read_text().splitlines():
        s = json.loads(line)
        samples[s["id"]] = s
corpora = arb.Corpora(data)
for r in pick:
    s = samples[r["sample_id"]]
    repo, _ = corpora.load(s["repo"], s["base_commit"])
    usage, log = Usage(), []
    inner = usage.complete()

    def complete(messages, tools, inner=inner, log=log):
        reply = inner(messages, tools)
        log.append(
            {
                "messages_in": len(messages),
                "reply": reply.text,
                "calls": [{"tool": c.name, "args": c.arguments} for c in reply.calls],
            }
        )
        return reply

    query = arb.query_text(s)
    result = repo_select(repo, query, arb.SELECT_BUDGET, method="llm", complete=complete)
    shortlist = [p for p, _ in rank.bm25(repo, query)][:30]
    out = {
        "sample": s["id"],
        "task_type": s["task_type"],
        "repo": s["repo"],
        "query": query[:3000],
        "gold": r["gold_files"],
        "selected": [f.path for f in result.files],
        "gold_in_shortlist": [g for g in r["gold_files"] if g in shortlist],
        "turns": log,
        "input_tokens": usage.input,
        "output_tokens": usage.output,
    }
    Path(f"bench/transcripts/arb/{s['task_type']}-{s['id'][:8]}.json").write_text(
        json.dumps(out, indent=1)
    )
    print(
        s["task_type"],
        s["repo"],
        "gold",
        r["gold_files"][:3],
        "in shortlist",
        out["gold_in_shortlist"],
        "selected",
        out["selected"][:6],
    )
