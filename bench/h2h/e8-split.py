"""E8 splits (see docs/EXPERIMENTS.md, E8). Writes local, gitignored dataset
files for the LongMemEval runner's --dataset-path:

- .cache/lme/e8-dev.json: 6 of the 45 questions the earlier retrieval spike
  already saw (results/lme-seen-ids.json), the first 2 by question id of
  each of their three types.
- .cache/lme/e8-dev-q1.json: the first of those, for a one-question cost probe.
- .cache/lme/e8-test-pool.json: every question outside those 45 questions'
  families. TEST is the runner's own stratified sample over this file
  (--per-type K, seed 42). A family is a question and its "_abs" rewrite,
  which share the same history, so a sibling of a seen question is seen too.

    python3 bench/h2h/e8-split.py
"""

import json
import os
from collections import defaultdict

LME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "lme")
data = json.load(open(os.path.join(LME, "longmemeval_s_cleaned.json")))
seen = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "lme-seen-ids.json"))))
assert len(seen) == 45, len(seen)

by_type = defaultdict(list)
for q in data:
    if q["question_id"] in seen:
        by_type[q["question_type"]].append(q)
dev = [q for t in sorted(by_type) for q in sorted(by_type[t], key=lambda q: q["question_id"])[:2]]
family = lambda qid: qid.removesuffix("_abs")
seen_families = {family(i) for i in seen}
pool = [q for q in data if family(q["question_id"]) not in seen_families]
assert len(dev) == 6, len(dev)

for name, rows in [("e8-dev.json", dev), ("e8-dev-q1.json", dev[:1]), ("e8-test-pool.json", pool)]:
    json.dump(rows, open(os.path.join(LME, name), "w"))
    print(name, len(rows), sorted({q["question_type"] for q in rows}))
print("dev ids", [q["question_id"] for q in dev])
