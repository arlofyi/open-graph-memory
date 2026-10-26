"""Terminal browser for a head-to-head split: one row per question, a ✓/✗ per
system, and the selected question's answers from every system underneath.
Picks up newly judged files while a run is still going.

    python3 bench/h2h/browse.py [test|dev|e8-dev|e8-test]   # E7 LoCoMo or E8 LongMemEval

Keys
  j/k ↓/↑ move    PgDn/PgUp page    g/G top/bottom    ]/[ scroll the answers
  s  sort: question · hardest first · category · newest judged
  f  filter: all · systems disagree · ogm-r2a wrong · ogm-r2a only right
  c  category filter        t  cutoff: top 10 · 50 · 200
  r  show the judge's reasons        p  autoplay (+/- slower/faster)
  q  quit

Read-only. Stdlib only (curses).
"""

import curses
import glob
import json
import locale
import os
import re
import sys
import textwrap
import time

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "h2h", SPLIT)
REF = "ogm-r2a"
ABBR = {"ogm-r2a": "R2", "ogm-denseX": "DX", "ogm-dense": "Dn", "mem0": "M0", "graphiti": "Gb",
        "graphiti-combined": "Gc", "cognee": "Cg", "arlo-memory": "Am"}
ORDER = list(ABBR)
CUTS = ["top_10", "top_50", "top_200"]
SORTS = ["question", "hardest first", "category", "newest judged"]
FILTERS = ["all", "systems disagree", f"{REF} wrong", f"{REF} only right"]


class Data:
    def __init__(self):
        self.cache = {}  # path -> (mtime, parsed)
        self.q = {}  # qid -> {system: parsed}
        self.mtime = {}  # qid -> newest judged time
        self.systems = []

    def refresh(self):
        seen = set()
        for f in glob.glob(os.path.join(ROOT, "predicted_*", "*.json")):
            if os.path.basename(f).startswith("_"):
                continue  # runner checkpoints
            seen.add(f)
            try:
                m = os.path.getmtime(f)
            except OSError:
                continue
            if f in self.cache and self.cache[f][0] == m:
                continue
            try:
                self.cache[f] = (m, json.load(open(f)))
            except (OSError, json.JSONDecodeError):
                pass  # still being written; next refresh
        for f in set(self.cache) - seen:
            del self.cache[f]
        self.q, self.mtime, systems = {}, {}, set()
        for f, (m, d) in self.cache.items():
            s = os.path.basename(os.path.dirname(f))[len("predicted_"):]
            if s.startswith("smoke"):
                continue
            systems.add(s)
            self.q.setdefault(d["question_id"], {})[s] = d
            self.mtime[d["question_id"]] = max(m, self.mtime.get(d["question_id"], 0))
        self.systems = sorted(systems, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))
        self.cats = ["all"] + sorted({category(d) for by in self.q.values() for d in by.values()})


def category(d):
    """LoCoMo: category_name. LongMemEval: question_type, abstention marked."""
    c = d.get("category_name") or d.get("question_type", "")
    return c + " (abs)" if d.get("is_abstention") else c


def verdict(d, cut):
    r = (d.get("cutoff_results") or {}).get(cut) or {}
    if not (r.get("generated_answer") or "").strip():
        return None  # call failed or not judged yet
    return r.get("judgment") in ("CORRECT", "PASS")  # LoCoMo / LongMemEval judges


def qkey(qid):
    m = re.fullmatch(r"conv(\d+)_q(\d+)", qid)
    return (int(m[1]), int(m[2]), "") if m else (0, 0, qid)


class App:
    def __init__(self, scr):
        self.scr, self.data = scr, Data()
        self.sel = self.top = self.dscroll = 0
        self.sort = self.filt = self.cat = 0
        self.cut, self.reasons = 0, False
        self.play, self.speed, self.last_step = False, 4.0, time.time()
        self.rows, self.status = [], ""
        curses.curs_set(0)
        curses.use_default_colors()
        for i, c in enumerate([curses.COLOR_GREEN, curses.COLOR_RED, curses.COLOR_YELLOW, curses.COLOR_CYAN], 1):
            curses.init_pair(i, c, -1)
        scr.timeout(250)

    def build(self):
        cut, ss = CUTS[self.cut], self.data.systems
        cur = self.rows[self.sel] if self.rows else None
        rows = []
        for qid, by in self.data.q.items():
            any_d = next(iter(by.values()))
            cats = self.data.cats
            self.cat %= len(cats)
            if cats[self.cat] != "all" and category(any_d) != cats[self.cat]:
                continue
            v = {s: verdict(by[s], cut) if s in by else None for s in ss}
            known = [x for x in v.values() if x is not None]
            ref = v.get(REF)
            if self.filt == 1 and len(set(known)) < 2:
                continue
            if self.filt == 2 and ref is not False:
                continue
            if self.filt == 3 and not (ref and all(x is False for s, x in v.items() if s != REF and x is not None)):
                continue
            rows.append(qid)
        key = [
            qkey,
            lambda q: (sum(1 for d in self.data.q[q].values() if verdict(d, cut)), qkey(q)),
            lambda q: (category(next(iter(self.data.q[q].values()))), qkey(q)),
            lambda q: -self.data.mtime[q],
        ][self.sort]
        self.rows = sorted(rows, key=key)
        if cur in self.rows:
            self.sel = self.rows.index(cur)
        self.sel = max(0, min(self.sel, len(self.rows) - 1))

    def put(self, y, x, s, attr=0):
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and x < w:
            try:
                self.scr.addnstr(y, x, s, w - x - 1, attr)
            except curses.error:
                pass

    def draw(self):
        scr = self.scr
        scr.erase()
        h, w = scr.getmaxyx()
        cut, ss = CUTS[self.cut], self.data.systems
        list_h = max(5, (h - 3) * 2 // 5)
        # header
        acc = []
        for s in ss:
            vs = [verdict(self.data.q[q][s], cut) for q in self.rows if s in self.data.q[q]]
            vs = [x for x in vs if x is not None]
            acc.append(f"{ABBR.get(s, s[:2])} {100 * sum(vs) / len(vs):4.1f}" if vs else f"{ABBR.get(s, s[:2])}  n/a")
        self.put(0, 0, f" {SPLIT} · {len(self.rows)} questions · @{cut[4:]} · sort: {SORTS[self.sort]} · filter: {FILTERS[self.filt]} · cat: {self.data.cats[self.cat % len(self.data.cats)]}" + ("  ▶ autoplay %.1fs" % self.speed if self.play else ""), curses.A_REVERSE)
        self.put(1, 0, " acc  " + "  ".join(acc), curses.color_pair(4))
        cols = "".join(f"{ABBR.get(s, s[:2]):>3}" for s in ss)
        self.put(2, 0, f" {'question':<16}{'category':<20}{cols}", curses.A_BOLD)
        # list
        if self.sel < self.top:
            self.top = self.sel
        if self.sel >= self.top + list_h:
            self.top = self.sel - list_h + 1
        for i, qid in enumerate(self.rows[self.top:self.top + list_h]):
            y, idx = 3 + i, self.top + i
            by = self.data.q[qid]
            cat = category(next(iter(by.values())))[:19]
            base = curses.A_REVERSE if idx == self.sel else 0
            self.put(y, 0, f" {qid[:15]:<16}{cat:<20}", base)
            x = 37
            for s in ss:
                v = verdict(by[s], cut) if s in by else None
                mark, col = ("✓", 1) if v else ("✗", 2) if v is False else ("·", 3)
                self.put(y, x, f"  {mark}", base | curses.color_pair(col))
                x += 3
            q = by.get(REF) or next(iter(by.values()))
            self.put(y, x + 2, q.get("question", ""), base | curses.A_DIM)
        # detail
        y0 = 3 + list_h
        self.put(y0, 0, "─" * (w - 1), curses.A_DIM)
        lines = self.detail(w - 2) if self.rows else [("Nothing matches this filter.", 0)]
        self.dscroll = max(0, min(self.dscroll, max(0, len(lines) - (h - y0 - 2))))
        for i, (text, attr) in enumerate(lines[self.dscroll:self.dscroll + h - y0 - 2]):
            self.put(y0 + 1 + i, 1, text, attr)
        self.put(h - 1, 0, " j/k move · ]/[ scroll answers · s sort · f filter · c category · t cutoff · r reasons · p play · q quit " + self.status, curses.A_REVERSE)
        scr.refresh()

    def detail(self, w):
        qid = self.rows[self.sel]
        by, cut = self.data.q[qid], CUTS[self.cut]
        any_d = by.get(REF) or next(iter(by.values()))
        ev = any_d.get("evidence") or any_d.get("answer_session_ids") or []
        asked = any_d.get("reference_date") or any_d.get("question_date", "")
        out = [(f"{qid}  [{category(any_d)}]  evidence {', '.join(ev)}  asked {asked}", curses.A_DIM)]
        for ln in textwrap.wrap("Q  " + any_d.get("question", ""), w):
            out.append((ln, curses.A_BOLD))
        for ln in textwrap.wrap("gold  " + str(any_d.get("ground_truth_answer")), w):
            out.append((ln, curses.color_pair(4)))
        for s in self.data.systems:
            out.append(("", 0))
            if s not in by:
                out.append((f"{s}: not judged yet", curses.color_pair(3)))
                continue
            r = (by[s].get("cutoff_results") or {}).get(cut) or {}
            v = verdict(by[s], cut)
            mark, col = ("✓", 1) if v else ("✗", 2) if v is False else ("· call failed", 3)
            ctx = sum(len(m.get("memory", "")) for m in by[s]["retrieval"]["search_results"][:int(cut[4:])]) // 4
            out.append((f"{mark} {s}   (~{ctx} context tokens)", curses.A_BOLD | curses.color_pair(col)))
            for ln in textwrap.wrap(r.get("generated_answer") or "", w - 2):
                out.append(("  " + ln, 0))
            if self.reasons and r.get("reason"):
                for ln in textwrap.wrap("judge: " + r["reason"], w - 2):
                    out.append(("  " + ln, curses.A_DIM))
        return out

    def run(self):
        last_refresh = 0
        self.put(0, 0, f" loading {ROOT} …", curses.A_REVERSE)
        self.scr.refresh()
        while True:
            if time.time() - last_refresh > 3:
                self.data.refresh()
                self.build()
                last_refresh = time.time()
            if self.play and time.time() - self.last_step >= self.speed:
                if self.sel < len(self.rows) - 1:
                    self.sel += 1
                    self.dscroll = 0
                self.last_step = time.time()
            self.draw()
            k = self.scr.getch()
            if k == -1:
                continue
            ch = chr(k) if 0 <= k < 256 else ""
            h = self.scr.getmaxyx()[0]
            page = max(5, (h - 3) * 2 // 5)
            moved = True
            if ch == "q":
                return
            elif ch == "j" or k == curses.KEY_DOWN:
                self.sel = min(self.sel + 1, len(self.rows) - 1)
            elif ch == "k" or k == curses.KEY_UP:
                self.sel = max(self.sel - 1, 0)
            elif k == curses.KEY_NPAGE:
                self.sel = min(self.sel + page, len(self.rows) - 1)
            elif k == curses.KEY_PPAGE:
                self.sel = max(self.sel - page, 0)
            elif ch == "g":
                self.sel = 0
            elif ch == "G":
                self.sel = max(len(self.rows) - 1, 0)
            else:
                moved = False
                if ch == "]":
                    self.dscroll += 3
                elif ch == "[":
                    self.dscroll = max(0, self.dscroll - 3)
                elif ch in "sfct":
                    if ch == "s":
                        self.sort = (self.sort + 1) % len(SORTS)
                    elif ch == "f":
                        self.filt = (self.filt + 1) % len(FILTERS)
                    elif ch == "c":
                        self.cat = (self.cat + 1) % len(self.data.cats)
                    else:
                        self.cut = (self.cut + 1) % len(CUTS)
                    self.build()
                elif ch == "r":
                    self.reasons = not self.reasons
                elif ch == "p":
                    self.play, self.last_step = not self.play, time.time()
                elif ch in "+=":
                    self.speed = min(60, self.speed + 1)
                elif ch == "-":
                    self.speed = max(1, self.speed - 1)
            if moved:
                self.dscroll = 0
                self.last_step = time.time()


def main(scr):
    App(scr).run()


if __name__ == "__main__":
    if not os.path.isdir(ROOT):
        sys.exit(f"no results at {ROOT}")
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(main)
