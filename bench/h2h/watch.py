"""Live view of a running E7 split: prints each question as the runner judges
it (system, question, gold answer, the answer at each cutoff, the verdict).

    python3 bench/h2h/watch.py [test|dev] [system ...]   # all systems if none named
    python3 bench/h2h/watch.py test --last 20 mem0       # also replay the 20 newest

Read-only; polls .cache/h2h/<split>/predicted_*/ every second.
"""

import glob
import json
import os
import sys
import textwrap
import time

args = sys.argv[1:]
split = args.pop(0) if args and args[0] in ("dev", "test") else "test"
last = 0
if "--last" in args:
    i = args.index("--last")
    last = int(args[i + 1])
    del args[i : i + 2]
only = set(args)
root = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "h2h", split)
C = {"CORRECT": "\033[32m✓\033[0m", "WRONG": "\033[31m✗\033[0m"}


def files():
    out = {}
    for f in glob.glob(os.path.join(root, "predicted_*", "conv*_q*.json")):
        name = os.path.basename(os.path.dirname(f))[len("predicted_") :]
        if not only or name in only:
            out[f] = os.path.getmtime(f)
    return out


def show(f):
    try:
        q = json.load(open(f))
    except (OSError, json.JSONDecodeError):
        return False  # still being written; retry next poll
    name = os.path.basename(os.path.dirname(f))[len("predicted_") :]
    print(f"\n\033[1m{name}\033[0m  {q['question_id']}  [{q.get('category_name')}]")
    print(f"  Q    {q['question']}")
    print(f"  gold {q.get('ground_truth_answer')}")
    for cut, r in (q.get("cutoff_results") or {}).items():
        ans = (r.get("generated_answer") or "").strip() or "\033[33m(empty: call failed)\033[0m"
        mark = C.get(r.get("judgment"), "?")
        print(f"  {mark} @{cut[4:]:<4}" + textwrap.shorten(ans, 160))
    return True


seen = files()
for f in sorted(seen, key=seen.get)[-last:] if last else []:
    show(f)
print(f"\nwatching {root} ({len(seen)} judged so far) — Ctrl-C to stop", flush=True)
while True:
    time.sleep(1)
    now = files()
    for f in sorted((f for f in now if seen.get(f) != now[f]), key=now.get):
        if show(f):
            seen[f] = now[f]
    sys.stdout.flush()
