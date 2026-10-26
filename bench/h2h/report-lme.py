"""E8 report: score one LongMemEval split of the head-to-head from the runner's
per-question JSON files (.cache/h2h/e8-<split>/predicted_<system>/<qid>.json).

Per system and cutoff: judged accuracy (PASS), per-type accuracy, abstention
accuracy (`_abs` questions), context the answerer was given (≈ tokens, chars /
4, of the top-k memories), and search latency p50/p90. Paired exact McNemar
(two-sided) of R2-A against every other arm at each cutoff, the pre-registered
matched-context comparison, and, when H2H_LEDGER points at a spend ledger
(JSON lines with `phase` and `usd`, one phase per system named
e8-<split>-<system>), ingest spend per system.

    python3 bench/h2h/report-lme.py [test|dev] [--json results/e8-lme-test.json]
"""

import glob
import json
import math
import os
import statistics
import sys

args = sys.argv[1:]
split = args.pop(0) if args and args[0] in ("dev", "test") else "test"
out = args[args.index("--json") + 1] if "--json" in args else None
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "h2h", f"e8-{split}")
LEDGER = os.getenv("H2H_LEDGER")
REF = "ogm-r2a"
PUBLIC = ["mem0", "graphiti", "graphiti-combined", "cognee"]
CUTS = ["top_10", "top_50", "top_200"]
# which ledger phase paid for each system's ingest (combined searches basic's ingest)
PHASE = {"graphiti-combined": "graphiti"}


def load(system):
    qs = {}
    for f in glob.glob(os.path.join(ROOT, f"predicted_{system}", "*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        q = json.load(open(f))
        qs[q["question_id"]] = q
    return qs


def mcnemar(b, c):
    """Exact two-sided McNemar on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def ok(q, cut):
    r = (q.get("cutoff_results") or {}).get(cut) or {}
    return r.get("judgment") == "PASS"


def tokens(q, cut):
    k = int(cut.split("_")[1])
    return sum(len(r.get("memory", "")) for r in q["retrieval"]["search_results"][:k]) / 4


def pct(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else None


systems = sorted(d[len(os.path.join(ROOT, "predicted_")) :] for d in glob.glob(os.path.join(ROOT, "predicted_*")))
data = {s: load(s) for s in systems}
data = {s: d for s, d in data.items() if d}

empty = {s: [qid for qid, q in d.items() for c in CUTS if not ((q["cutoff_results"].get(c) or {}).get("generated_answer") or "").strip()] for s, d in data.items()}
if any(empty.values()):
    sys.exit(f"empty answers, refusing to score: { {s: v for s, v in empty.items() if v} }")

spend = {}
for line in open(LEDGER) if LEDGER else []:
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    p = r.get("phase") or ""
    if p.startswith(f"e8-{split}-"):
        spend[p[len(f"e8-{split}-") :]] = spend.get(p[len(f"e8-{split}-") :], 0) + r.get("usd", 0)

types = sorted({q["question_type"] for d in data.values() for q in d.values()})
report = {"split": split, "systems": {}, "vs_ref": {}, "matched_context": {}, "spend_usd": {k: round(v, 2) for k, v in spend.items()}}
for s, d in data.items():
    qs = list(d.values())
    lat = sorted(q["retrieval"]["search_latency_ms"] for q in qs)
    row = {"n": len(qs), "latency_ms_p50": round(statistics.median(lat)), "latency_ms_p90": round(lat[int(0.9 * (len(lat) - 1))])}
    for c in CUTS:
        row[c] = {
            "accuracy": pct([ok(q, c) for q in qs]),
            "context_tokens_mean": round(statistics.mean(tokens(q, c) for q in qs)),
            "abstention": pct([ok(q, c) for q in qs if q.get("is_abstention")]),
            "abstention_n": sum(1 for q in qs if q.get("is_abstention")),
            **{t: pct([ok(q, c) for q in qs if q["question_type"] == t]) for t in types},
        }
    ph = PHASE.get(s, s)
    row["ingest_usd_per_question"] = round(spend.get(ph, 0) / len(qs), 3) if LEDGER else None
    report["systems"][s] = row

ref = data[REF]
for s, d in data.items():
    if s == REF:
        continue
    ids = sorted(set(ref) & set(d))
    report["vs_ref"][s] = {}
    for c in CUTS:
        win = sum(1 for i in ids if ok(ref[i], c) and not ok(d[i], c))
        loss = sum(1 for i in ids if ok(d[i], c) and not ok(ref[i], c))
        report["vs_ref"][s][c] = {"n": len(ids), f"{REF}_wins": win, f"{s}_wins": loss, "p": float(f"{mcnemar(win, loss):.3g}")}
    # matched context: R2-A at the smallest cutoff whose mean context >= X's top-10 context
    target = report["systems"][s]["top_10"]["context_tokens_mean"]
    cut = next((c for c in CUTS if report["systems"][REF][c]["context_tokens_mean"] >= target), None)
    use = cut or CUTS[-1]
    win = sum(1 for i in ids if ok(ref[i], use) and not ok(d[i], "top_10"))
    loss = sum(1 for i in ids if ok(d[i], "top_10") and not ok(ref[i], use))
    report["matched_context"][s] = {
        "rival_top10_tokens": target,
        "ref_cutoff": use,
        "ref_tokens": report["systems"][REF][use]["context_tokens_mean"],
        "reached": cut is not None,
        "ref_accuracy": pct([ok(ref[i], use) for i in ids]),
        "rival_accuracy": pct([ok(d[i], "top_10") for i in ids]),
        f"{REF}_wins": win,
        f"{s}_wins": loss,
        "p": float(f"{mcnemar(win, loss):.3g}"),
    }

S = report["systems"]
order = sorted(S, key=lambda s: -S[s]["top_10"]["accuracy"])
print(f"| system | n | acc@10 | acc@50 | acc@200 | tokens@10 | tokens@200 | abstention@10 | search p50 | ingest $/q |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in order:
    r = S[s]
    print(f"| {s} | {r['n']} | {r['top_10']['accuracy']} | {r['top_50']['accuracy']} | {r['top_200']['accuracy']} | {r['top_10']['context_tokens_mean']:,} | {r['top_200']['context_tokens_mean']:,} | {r['top_10']['abstention']} ({r['top_10']['abstention_n']}) | {r['latency_ms_p50']:,} ms | {r['ingest_usd_per_question']} |")
print(f"\n| system | " + " | ".join(types) + " |")
print("|---|" + "---|" * len(types))
for s in order:
    print(f"| {s} | " + " | ".join(str(S[s]["top_10"][t]) for t in types) + " |")
print("\nR2-A vs each arm at top 10 (R2-A wins / losses, p):")
for s, v in report["vs_ref"].items():
    t = v["top_10"]
    print(f"  {s}: {t[f'{REF}_wins']}/{t[f'{s}_wins']} p={t['p']}")
print("\nMatched context:")
for s, m in report["matched_context"].items():
    print(f"  {s}: R2-A@{m['ref_cutoff']} ({m['ref_tokens']:,} tok{'' if m['reached'] else ', NOT reached'}) {m['ref_accuracy']} vs {s}@10 ({m['rival_top10_tokens']:,} tok) {m['rival_accuracy']}  {m[f'{REF}_wins']}/{m[f'{s}_wins']} p={m['p']}")
print("\nspend:", report["spend_usd"])
if out:
    json.dump(report, open(out, "w"), indent=2)
    print("wrote", out)
