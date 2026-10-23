"""Score cached OCR outputs against the manifest answers -> table + ocr_bench/results.json

    python ocr_bench/score.py

Unit = one number (make_sets.NUM_RE). Exact match = identical after NFKC; no tolerance.

Readings:
  boxed engines (tesseract, vision): on pages with positions, the reading of an answer number is
    the engine's characters whose centers fall inside the answer's box. On pb_notext (no positions)
    numbers are aligned in reading order (difflib).
  lighton: no word/cell boxes, so its numbers are aligned in reading order inside each of its
    blocks (answers assigned to the block that contains them); leftovers are aligned page-wide.
Flags (fixed before the run, not tuned on it):
  tesseract  word conf < 90      vision  line conf < 0.9      lighton  no confidence -> never flags
  pair (lighton + boxed): a number is accepted only if both engines read the same number at the
    same place (same lighton block, aligned in order); everything else in that block is flagged,
    and a number both missed inside a disagreeing block counts as flagged. Partner text outside
    every lighton block is flagged (nothing confirmed it).
Missing numbers from a single engine are never flagged: nothing signals them.
"""
import html
import json
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from make_sets import nums

BENCH = Path(__file__).resolve().parent
TAU = {"tesseract": 90.0, "vision": 0.9, "vision-auto": 0.9}
BOXED = ("tesseract", "vision", "vision-auto")
LIGHTON = ("lighton-1b",)  # 0.8B: 24 min/page on MPS (no fast linear-attention kernels), not run
PAD = 1.0  # pt, slack around answer boxes
RUN_LEVEL = True  # v2 reading (see positional_reading); False reproduces v1
RULES = True  # new OCR flag rules: LightOn empty/cut guard, sign-uncertain dash, collapsed boxes


def lighton_guard(out):
    """A LightOn page whose output cannot be trusted at all: empty (the VIN page: one token, no
    blocks) or cut at the token limit. Every number on it is flagged; nothing can be confirmed."""
    return out is not None and (not out["blocks"] or out["truncated"])


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def inside(pt, b, pad=PAD):
    return b[0] - pad <= pt[0] <= b[2] + pad and b[1] - pad <= pt[1] <= b[3] + pad


def reading_order(objs, key=lambda o: o["bbox"]):
    """Sort boxes into lines (y-centers within half a median height), then left to right."""
    objs = [o for o in objs if key(o)]
    if not objs:
        return []
    h = statistics.median(key(o)[3] - key(o)[1] for o in objs) or 1.0
    objs.sort(key=lambda o: center(key(o))[1])
    lines, cur, y0 = [], [], None
    for o in objs:
        y = center(key(o))[1]
        if cur and y - y0 > 0.5 * h:
            lines.append(cur)
            cur = []
        if not cur:
            y0 = y
        cur.append(o)
    lines.append(cur)
    return [o for line in lines for o in sorted(line, key=lambda o: key(o)[0])]


def align(gt, pred):
    """gt/pred: lists of strings -> list (per gt index) of (pred index or None, equal?)."""
    out = [(None, False)] * len(gt)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = (j1 + k, True)
        elif tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                out[i1 + k] = (j1 + k, False)
    return out


# ---------------------------------------------------------------- engine outputs -> chars / runs
SIGN_UNCERTAIN = re.compile(r"^\(?[-−–]")  # OCR cannot tell hyphen / minus / en dash apart
DEGENERATE = 0.12  # a char box narrower than this share of its height is collapsed


def boxed_chars(out, engine):
    """-> chars [{c, bbox, conf}] (NFKC-expanded) and runs [{t, bbox, conf, chars:[idx]}]."""
    chars, runs = [], []
    for line in out["lines"]:
        pieces = []
        for ch in line["chars"]:
            for c in nfkc(ch["c"]):
                pieces.append({"c": c, "bbox": ch["bbox"], "conf": ch["conf"]})
        base = len(chars)
        chars += pieces
        text = "".join(p["c"] for p in pieces)
        for s, e, run in nums(text):
            idx = [base + k for k in range(s, e) if chars[base + k]["bbox"]]
            if not idx:
                continue
            bb = [min(chars[i]["bbox"][0] for i in idx), min(chars[i]["bbox"][1] for i in idx),
                  max(chars[i]["bbox"][2] for i in idx), max(chars[i]["bbox"][3] for i in idx)]
            # collapsed char boxes (Tesseract jpn puts `A4(` at one x with ~0 width): position unreliable
            degenerate = any((chars[i]["bbox"][2] - chars[i]["bbox"][0]) < DEGENERATE * (chars[i]["bbox"][3] - chars[i]["bbox"][1])
                             for i in idx)
            runs.append({"t": run, "bbox": bb, "conf": min(chars[i]["conf"] for i in idx), "chars": idx,
                         "sign_uncertain": RULES and (bool(SIGN_UNCERTAIN.match(run)) or degenerate)})
    return chars, runs


def lighton_runs(out):
    """-> blocks [{bbox, runs:[str]}] in the model's own order."""
    blocks = []
    for b in out["blocks"]:
        t = re.sub(r"</t[dh]>", " │ ", b["text"])
        t = re.sub(r"</tr>|<br\s*/?>", "\n", t)
        t = nfkc(html.unescape(re.sub(r"<[^>]+>", " ", t)))
        blocks.append({"bbox": b["bbox"], "runs": [r for _, _, r in nums(t)]})
    return blocks


def positional_reading(gt, chars, runs=None):
    """The engine's reading of one answer number. v2 (run level): the engine's whole numbers whose
    center lies inside the answer box, joined left to right. v1 sliced loose characters by their
    centers, and engine char boxes are a little off, so a neighbour's char slipped in (`3` -> `53`) or
    one of ours fell out (`(2025` -> `(202`) although the engine read the number right."""
    if runs is not None:
        hit_runs = sorted((r for r in runs if inside(center(r["bbox"]), gt["bbox"])), key=lambda r: r["bbox"][0])
        idx = [k for r in hit_runs for k in r["chars"]]
        return "".join(r["t"] for r in hit_runs), idx
    hit = [k for k, c in enumerate(chars) if c["bbox"] and inside(center(c["bbox"]), gt["bbox"])]
    hit.sort(key=lambda k: chars[k]["bbox"][0])
    return "".join(chars[k]["c"] for k in hit), hit


# ---------------------------------------------------------------- per page scoring
def score_boxed(item, chars, runs, char_flag, miss_flag=lambda gt: False):
    """char_flag(k) -> bool. Returns per-answer records (ok, flagged, missing, minconf)."""
    recs = []
    if item["ordered_only"]:
        ordered = reading_order(runs)
        for g, (j, eq) in zip(item["gt"], align([g["t"] for g in item["gt"]], [r["t"] for r in ordered])):
            r = ordered[j] if j is not None else None
            flagged = bool(r) and any(char_flag(k) for k in r["chars"])
            recs.append({"ok": eq, "flagged": flagged, "missing": r is None, "minconf": r["conf"] if r else None})
        return recs
    for g in item["gt"]:
        text, hit = positional_reading(g, chars, runs if RUN_LEVEL else None)
        flagged = any(char_flag(k) for k in hit) or (not hit and miss_flag(g))
        recs.append({"ok": text == g["t"], "flagged": flagged, "missing": not hit,
                     "minconf": min((chars[k]["conf"] for k in hit), default=None)})
    return recs


def score_lighton(item, blocks, page_w):
    gt = item["gt"]
    ok = [False] * len(gt)
    used = [[False] * len(b["runs"]) for b in blocks]
    if not item["ordered_only"]:
        pad = 0.01 * page_w
        owner = {}
        for i, g in enumerate(gt):
            for bi, b in enumerate(blocks):
                if inside(center(g["bbox"]), b["bbox"], pad):
                    owner.setdefault(bi, []).append(i)
                    break
        for bi, idx in owner.items():
            order = [o["i"] for o in reading_order([dict(gt[i], i=i) for i in idx])]
            for gi, (j, eq) in zip(order, align([gt[i]["t"] for i in order], blocks[bi]["runs"])):
                if eq:
                    ok[gi], used[bi][j] = True, True
    left_gt = [i for i in range(len(gt)) if not ok[i]]
    left_pred = [(bi, j) for bi, b in enumerate(blocks) for j in range(len(b["runs"])) if not used[bi][j]]
    for gi, (j, eq) in zip(left_gt, align([gt[i]["t"] for i in left_gt], [blocks[bi]["runs"][j] for bi, j in left_pred])):
        ok[gi] = ok[gi] or eq
    return [{"ok": o, "flagged": False, "missing": None, "minconf": None} for o in ok]


def guarded(recs, flag_all):
    return [dict(r, flagged=True) for r in recs] if (flag_all and RULES) else recs


def score_pair(item, chars, runs, blocks, page_w):
    """LightOn + boxed partner: accept a partner number only if LightOn read the same number in
    the same block at the same place in order."""
    pad = 0.01 * page_w
    run_flag = [True] * len(runs)
    disagree = [False] * len(blocks)
    for bi, b in enumerate(blocks):
        mine = reading_order([dict(r, i=i) for i, r in enumerate(runs)
                              if inside(center(r["bbox"]), b["bbox"], pad)])
        mine = [m for m in mine if run_flag[m["i"]]]  # a run belongs to the first block holding it
        ops = SequenceMatcher(None, [m["t"] for m in mine], b["runs"], autojunk=False).get_opcodes()
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                for k in range(i1, i2):
                    run_flag[mine[k]["i"]] = mine[k]["sign_uncertain"]  # agreement cannot fix a shared sign misread
            else:
                disagree[bi] = True
    in_run = {}
    for i, r in enumerate(runs):
        for k in r["chars"]:
            in_run[k] = i

    def where(pt):
        for bi, b in enumerate(blocks):
            if inside(pt, b["bbox"], pad):
                return bi
        return None

    def char_flag(k):
        if k in in_run:
            return run_flag[in_run[k]]
        bi = where(center(chars[k]["bbox"]))
        return bi is None or disagree[bi]

    def miss_flag(g):
        bi = where(center(g["bbox"]))
        return bi is not None and disagree[bi]

    return score_boxed(item, chars, runs, char_flag, miss_flag)


# ---------------------------------------------------------------- aggregate
def load(engine, item):
    f = BENCH / "out" / engine / f"{item['id']}.json"
    return json.loads(f.read_text()) if f.exists() else None


def summarize(recs, secs, n_pages):
    n = sum(len(r) for r in recs)
    flat = [x for r in recs for x in r]
    wrong = [x for x in flat if not x["ok"]]
    return {"pages": n_pages, "numbers": n,
            "exact_pct": round(100 * sum(x["ok"] for x in flat) / n, 2) if n else None,
            # one dense page (e.g. a VIN list) can dominate the pooled number, so also per page
            "page_median_exact_pct": round(statistics.median(100 * sum(x["ok"] for x in r) / len(r) for r in recs if r), 2),
            "wrong": len(wrong),
            "wrong_unflagged": sum(not x["flagged"] for x in wrong),
            "missing": sum(bool(x["missing"]) for x in wrong),
            "review_pct": round(100 * sum(x["flagged"] for x in flat) / n, 2) if n else None,
            "pages_with_review_pct": round(100 * sum(any(x["flagged"] for x in r) for r in recs) / n_pages, 1) if n_pages else None,
            "sec_per_page_mean": round(statistics.mean(secs), 2) if secs else None,
            "sec_per_page_p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 2) if secs else None}


def tau_needed(recs, engine):
    """Single boxed engine: lowest threshold that flags every wrong number it actually read,
    and the review share at that threshold. Missing numbers can't be flagged by any threshold."""
    flat = [x for r in recs for x in r]
    read_wrong = [x["minconf"] for x in flat if not x["ok"] and x["minconf"] is not None]
    if not read_wrong:
        return {"tau_for_zero": None, "review_pct_at_tau": None}
    tau = max(read_wrong) + (1 if engine == "tesseract" else 0.001)
    flagged = sum(x["minconf"] is not None and x["minconf"] < tau for x in flat)
    return {"tau_for_zero": round(tau, 3), "review_pct_at_tau": round(100 * flagged / len(flat), 2),
            "unflaggable_missing": sum(not x["ok"] and x["minconf"] is None for x in flat)}


def main():
    items = json.loads((BENCH / "sets" / "manifest.json").read_text())
    systems = list(BOXED) + list(LIGHTON) + [f"{l}+{b}" for l in LIGHTON for b in BOXED]
    results, runs_meta = {}, {}
    for e in list(BOXED) + list(LIGHTON):
        f = BENCH / "out" / e / "_run.json"
        runs_meta[e] = json.loads(f.read_text()) if f.exists() else None
    for sysname in systems:
        parts = sysname.split("+")
        per_set = {}
        for set_name in ("gold_ocr", "pb_notext", "real_render", "ALL"):
            sel = [i for i in items if set_name == "ALL" or i["set"] == set_name]
            recs, secs = [], []
            for it in sel:
                outs = [load(p, it) for p in parts]
                if any(o is None for o in outs):
                    continue
                if parts[0] in BOXED:
                    chars, runs = boxed_chars(outs[0], parts[0])
                    tau = TAU[parts[0]]
                    signk = {k for r_ in runs if r_["sign_uncertain"] for k in r_["chars"]}
                    r = score_boxed(it, chars, runs, lambda k, c=chars, t=tau, sk=signk: k in sk or (c[k]["conf"] is not None and c[k]["conf"] < t))
                elif len(parts) == 1:
                    r = guarded(score_lighton(it, lighton_runs(outs[0]), it["width"]), lighton_guard(outs[0]))
                else:
                    chars, runs = boxed_chars(outs[1], parts[1])
                    r = guarded(score_pair(it, chars, runs, lighton_runs(outs[0]), it["width"]), lighton_guard(outs[0]))
                recs.append(r)
                secs.append(sum(o["seconds"] for o in outs))  # pair runs both engines
            if not recs:
                continue
            s = summarize(recs, secs, len(recs))
            if parts[0] in BOXED and set_name == "ALL":
                s.update(tau_needed(recs, parts[0]))
            per_set[set_name] = s
        if per_set:
            mem = [runs_meta.get(p) for p in parts]
            per_set["memory"] = {p: m for p, m in zip(parts, mem)}
            results[sysname] = per_set
    (BENCH / "results.json").write_text(json.dumps(results, indent=1))
    hdr = f"{'system':28s} {'set':12s} {'pages':>5s} {'nums':>5s} {'exact%':>7s} {'pgMed%':>7s} {'wrong':>6s} {'UNFLAG':>6s} {'miss':>5s} {'review%':>7s} {'pgRev%':>6s} {'s/page':>7s}"
    print(hdr)
    for name, per in results.items():
        for set_name, s in per.items():
            if set_name == "memory":
                continue
            print(f"{name:28s} {set_name:12s} {s['pages']:5d} {s['numbers']:5d} {s['exact_pct']:7.2f} {s['page_median_exact_pct']:7.2f} {s['wrong']:6d} "
                  f"{s['wrong_unflagged']:6d} {s['missing']:5d} {s['review_pct']:7.2f} {s['pages_with_review_pct']:6.1f} {s['sec_per_page_mean']:7.2f}")
    print("\nmemory (MB): ", json.dumps({e: m and {k: round(v) for k, v in m.items() if isinstance(v, float) and "mb" in k}
                                         for e, m in runs_meta.items()}))
    for name in BOXED:
        if name in results:
            a = results[name]["ALL"]
            print(f"{name}: threshold for 0 unflagged (read numbers) = {a.get('tau_for_zero')}, review% at it = "
                  f"{a.get('review_pct_at_tau')}, missing numbers no threshold can flag = {a.get('unflaggable_missing')}")


def _selftest():
    assert align(["1", "2", "3"], ["1", "9", "3"]) == [(0, True), (1, False), (2, True)]
    assert align(["1", "2", "3"], ["1", "3"]) == [(0, True), (None, False), (1, True)]
    item = {"ordered_only": False, "gt": [{"t": "2,979", "bbox": [10, 10, 30, 20]}]}
    chars = [{"c": c, "bbox": [10 + 4 * i, 10, 14 + 4 * i, 20], "conf": 99.0} for i, c in enumerate("2,979")]
    runs = lambda t: [{"t": t, "bbox": [10, 10, 30, 20], "conf": 99.0, "chars": list(range(5))}]
    assert score_boxed(item, chars, runs("2,979"), lambda k: False)[0]["ok"]
    r = score_boxed(item, chars, runs("2,978"), lambda k: chars[k]["conf"] < 90)[0]
    assert not r["ok"] and not r["flagged"]
    # a neighbour's run whose center is outside the answer box is not part of the reading
    near = runs("2,979") + [{"t": "5", "bbox": [31, 10, 35, 20], "conf": 99.0, "chars": []}]
    assert score_boxed(item, chars, near, lambda k: False)[0]["ok"]
    assert lighton_guard({"blocks": [], "truncated": False}) and lighton_guard({"blocks": [1], "truncated": True})


if __name__ == "__main__":
    _selftest()
    main()
