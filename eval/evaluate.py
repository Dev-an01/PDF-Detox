"""Table-extraction evaluation: predicted JSON vs gold JSON (schema in design.md §3).

Usage:
    python evaluate.py GOLD_DIR PRED_DIR [--out report.json] [--no-teds]
    python evaluate.py                      # runs the self-check

Gold and pred files pair by filename (doc_id.json). Metrics:
  - table detection  P/R/F1 at IoU >= 0.5 (first region of each table); gold without
    regions (ParseBench) is matched by cell-text overlap instead
  - TEDS, TEDS-struct (PubTabNet definition; tree edit distance via `apted`)
  - cell exact match    raw_text equal after whitespace collapse only
  - numeric exact match normalized `value` equal as Decimal, zero tolerance; gold without
    `value` (ParseBench) counts every cell with a digit, compared on exact raw_text
  - unflagged numeric errors  wrong/missing numbers in a cell, table or page NOT flagged for
    review (pred cell `flags` non-empty, table `needs_review`, or the number's page has
    non-empty `flags` in pred `pages`). Target: 0
  - column shifts       gold value missing from its column but found in another column of the
    same row (a cell that holds the right text is never a shift, even if the row repeats it)
  - missing / extra rows (rows aligned with difflib, so one dropped row
    does not cascade into every row below it)
Dependency: apted (pip install apted). Everything else is stdlib.
"""
import json
import sys
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

from apted import APTED, Config

IOU_MATCH = 0.5
TEXT_MATCH = 0.3  # min cell-text overlap when matching tables without bboxes


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def dec(v):
    try:
        return Decimal(v) if v is not None else None
    except InvalidOperation:
        return None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------- TEDS ----------
class Node:
    def __init__(self, tag, span=(1, 1), text="", children=None):
        self.tag, self.span, self.text, self.children = tag, span, text, children or []


class TedsConfig(Config):
    def __init__(self, struct_only):
        self.struct_only = struct_only

    def rename(self, a, b):
        if a.tag != b.tag or a.span != b.span:
            return 1.0
        if a.tag == "td" and not self.struct_only:
            return lev(a.text, b.text) / max(len(a.text), len(b.text), 1)
        return 0.0


def to_tree(table):
    rows = {}
    for c in table["cells"]:
        rows.setdefault(c["row"], []).append(c)
    trs = [
        Node("tr", children=[
            Node("td", (c.get("row_span", 1), c.get("col_span", 1)), norm(c["raw_text"]))
            for c in sorted(rows[r], key=lambda c: c["col"])
        ])
        for r in sorted(rows)
    ]
    return Node("table", children=trs)


def count(n):
    return 1 + sum(count(c) for c in n.children)


def teds(gold, pred, struct_only=False):
    g, p = to_tree(gold), to_tree(pred)
    d = APTED(g, p, TedsConfig(struct_only)).compute_edit_distance()
    return 1.0 - d / max(count(g), count(p))


# ---------- cell-level ----------
def grid(table):
    """row index -> {col: cell}, anchor cells only, rows in order."""
    rows = {}
    for c in table["cells"]:
        rows.setdefault(c["row"], {})[c["col"]] = c
    return [rows[r] for r in sorted(rows)]


def row_sig(row):
    return "\x1f".join(norm(row[c]["raw_text"]) for c in sorted(row))


def is_num(gc):
    if "value" in gc:
        return dec(gc["value"]) is not None
    return any(ch.isdigit() for ch in gc["raw_text"])


def num_ok(gc, pc):
    if pc is None:
        return False
    if "value" in gc:
        return dec(pc.get("value")) == dec(gc["value"])
    return norm(pc["raw_text"]) == norm(gc["raw_text"])


def gold_page(gold_table, cell=None):
    if cell and cell.get("page"):
        return cell["page"]
    return gold_table["regions"][0]["page"] if gold_table.get("regions") else 1


def cell_metrics(gold, pred, flagged_pages=frozenset(), seen=None):
    g_rows, p_rows = grid(gold), grid(pred)
    table_flagged = bool(pred.get("needs_review"))
    page_flagged = lambda gc: gold_page(gold, gc) in flagged_pages
    m = dict(cells=0, cells_exact=0, nums=0, nums_exact=0, nums_flagged=0, num_errors_unflagged=0, col_shifts=0,
             rows_gold=len(g_rows), rows_missing=0, rows_extra=0)
    pairs = []
    sm = SequenceMatcher(None, [row_sig(r) for r in g_rows], [row_sig(r) for r in p_rows], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        n = min(i2 - i1, j2 - j1) if tag == "replace" else (i2 - i1 if tag == "equal" else 0)
        pairs += [(g_rows[i1 + k], p_rows[j1 + k]) for k in range(n)]
        m["rows_missing"] += (i2 - i1) - n
        m["rows_extra"] += (j2 - j1) - n
    for g, p in pairs:
        p_by_raw = {}
        for col, c in p.items():
            p_by_raw.setdefault(norm(c["raw_text"]), set()).add(col)
        for col, gc in g.items():
            pc = p.get(col)
            m["cells"] += 1
            m["cells_exact"] += pc is not None and norm(pc["raw_text"]) == norm(gc["raw_text"])
            if not is_num(gc):
                continue
            m["nums"] += 1
            flagged = table_flagged or page_flagged(gc) or bool(pc is not None and pc.get("flags"))
            m["nums_flagged"] += flagged
            if num_ok(gc, pc):
                m["nums_exact"] += 1
                if seen is not None:
                    seen["ok"].add(id(gc))
                continue
            if not flagged:
                m["num_errors_unflagged"] += 1
            elif seen is not None:
                seen["flagged"].add(id(gc))
            holds_it = pc is not None and norm(pc["raw_text"]) == norm(gc["raw_text"])
            if not holds_it and p_by_raw.get(norm(gc["raw_text"]), set()) - {col}:
                m["col_shifts"] += 1
    # Cells in gold rows that were never aligned count as wrong, not ignored.
    aligned = {id(g) for g, _ in pairs}
    for g in g_rows:
        if id(g) not in aligned:
            n_num = sum(is_num(c) for c in g.values())
            m["cells"] += len(g)
            m["nums"] += n_num
            silent = 0 if table_flagged else sum(is_num(c) and not page_flagged(c) for c in g.values())
            m["num_errors_unflagged"] += silent
            m["nums_flagged"] += n_num - silent
            if seen is not None:
                seen["flagged"] |= {id(c) for c in g.values() if is_num(c) and (table_flagged or page_flagged(c))}
    return m


# ---------- document / corpus ----------
def first_region(t):
    r = t["regions"][0]
    return r["page"], r["bbox"]


def text_overlap(a, b):
    ca = Counter(norm(c["raw_text"]) for c in a["cells"] if c["raw_text"].strip())
    cb = Counter(norm(c["raw_text"]) for c in b["cells"] if c["raw_text"].strip())
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 0.0


def match_tables(gold_tables, pred_tables):
    cands = []
    for i, g in enumerate(gold_tables):
        for j, p in enumerate(pred_tables):
            if g.get("regions") and p.get("regions"):
                (gp, gb), (pp, pb) = first_region(g), first_region(p)
                s = iou(gb, pb) if gp == pp else 0.0
                ok = s >= IOU_MATCH
            else:
                s = text_overlap(g, p)
                ok = s >= TEXT_MATCH
            if ok:
                cands.append((s, i, j))
    used_g, used_p, out = set(), set(), []
    for s, i, j in sorted(cands, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i), used_p.add(j), out.append((i, j))
    return out


def eval_doc(gold, pred, with_teds=True):
    gt, pt = gold["tables"], pred["tables"]
    flagged_pages = frozenset(pg["page"] for pg in pred.get("pages", []) if pg.get("flags"))
    matches = match_tables(gt, pt)
    tot = dict(tables_gold=len(gt), tables_pred=len(pt), tables_matched=len(matches),
               teds=[], teds_struct=[])
    seen = {"ok": set(), "flagged": set()}
    for i, j in matches:
        if with_teds:  # pure-Python TED is slow on 1000+ cell tables; skip with --no-teds
            tot["teds"].append(teds(gt[i], pt[j]))
            tot["teds_struct"].append(teds(gt[i], pt[j], struct_only=True))
        for k, v in cell_metrics(gt[i], pt[j], flagged_pages, seen).items():
            tot[k] = tot.get(k, 0) + v
    unmatched = set(range(len(gt))) - {i for i, _ in matches}
    for i in unmatched:  # missed tables: all their cells/rows count as lost
        if with_teds:
            tot["teds"].append(0.0)
            tot["teds_struct"].append(0.0)
        g = grid(gt[i])
        tot["rows_gold"] = tot.get("rows_gold", 0) + len(g)
        tot["rows_missing"] = tot.get("rows_missing", 0) + len(g)
        n_num = sum(is_num(c) for r in g for c in r.values())
        silent = sum(is_num(c) and gold_page(gt[i], c) not in flagged_pages for r in g for c in r.values())
        tot["cells"] = tot.get("cells", 0) + sum(len(r) for r in g)
        tot["nums"] = tot.get("nums", 0) + n_num
        tot["num_errors_unflagged"] = tot.get("num_errors_unflagged", 0) + silent
        tot["nums_flagged"] = tot.get("nums_flagged", 0) + n_num - silent
        seen["flagged"] |= {id(c) for r in g for c in r.values() if is_num(c) and gold_page(gt[i], c) in flagged_pages}
    # Split-only diagnostic: a gold row whose text, joined without spaces, equals some output row on
    # the page (each output row used once, any table, any order) holds the right numbers; cells are
    # only split differently (`EUR 20,000,000` vs `EUR | 20,000,000`). Wrong numbers in such rows are
    # counted apart so the real error share is visible. Existing metrics are unchanged.
    joined = lambda r: "".join(norm(c["raw_text"]) for _, c in sorted(r.items())).replace(" ", "")
    pool = Counter(joined(r) for t in pt for r in grid(t))
    so_wrong = so_silent = 0
    for t in gt:
        for r in grid(t):
            key = joined(r)
            if key and pool[key] > 0:
                pool[key] -= 1
                for c in r.values():
                    if is_num(c) and id(c) not in seen["ok"]:
                        so_wrong += 1
                        so_silent += id(c) not in seen["flagged"]
    tot["num_errors_split_only"] = so_wrong
    tot["num_errors_unflagged_split_only"] = so_silent
    return tot


def summarize(docs):
    s = {}
    for d in docs.values():
        for k, v in d.items():
            s[k] = s.get(k, []) + v if isinstance(v, list) else s.get(k, 0) + v
    ratio = lambda a, b: round(a / b, 4) if b else None
    p, r = ratio(s.get("tables_matched", 0), s.get("tables_pred", 0)), ratio(s.get("tables_matched", 0), s.get("tables_gold", 0))
    return {
        "docs": len(docs),
        "table_precision": p, "table_recall": r,
        "table_f1": round(2 * p * r / (p + r), 4) if p and r else 0.0,
        "teds": ratio(sum(s.get("teds", [])), len(s.get("teds", []))),
        "teds_struct": ratio(sum(s.get("teds_struct", [])), len(s.get("teds_struct", []))),
        "cell_exact_acc": ratio(s.get("cells_exact", 0), s.get("cells", 0)),
        "numeric_exact_acc": ratio(s.get("nums_exact", 0), s.get("nums", 0)),
        "numeric_errors": s.get("nums", 0) - s.get("nums_exact", 0),
        "numeric_errors_unflagged": s.get("num_errors_unflagged", 0),
        "numeric_errors_split_only": s.get("num_errors_split_only", 0),
        "numeric_errors_unflagged_split_only": s.get("num_errors_unflagged_split_only", 0),
        "numbers": s.get("nums", 0),
        "numbers_flagged": s.get("nums_flagged", 0),
        "review_pct": round(100 * s.get("nums_flagged", 0) / s["nums"], 2) if s.get("nums") else None,
        "column_shifts": s.get("col_shifts", 0),
        "rows_missing": s.get("rows_missing", 0),
        "rows_extra": s.get("rows_extra", 0),
        "rows_gold": s.get("rows_gold", 0),
    }


def run(gold_dir, pred_dir, with_teds=True):
    docs, tags = {}, {}
    for gf in sorted(Path(gold_dir).glob("*.json")):
        pf = Path(pred_dir) / gf.name
        pred = json.loads(pf.read_text()) if pf.exists() else {"tables": []}
        gold = json.loads(gf.read_text())
        docs[gf.stem] = eval_doc(gold, pred, with_teds)
        for t in gold.get("tags", []):
            tags.setdefault(t, []).append(gf.stem)
    return {"summary": summarize(docs),
            "by_tag": {t: summarize({d: docs[d] for d in ids}) for t, ids in sorted(tags.items())},
            "per_doc": docs}


def _selftest():
    def cell(r, c, raw, val=None):
        return {"row": r, "col": c, "raw_text": raw, "value": val}

    def table(rows):
        cells = [cell(r, c, raw, val) for r, row in enumerate(rows) for c, (raw, val) in enumerate(row)]
        return {"regions": [{"page": 1, "bbox": [0, 0, 100, 100]}], "cells": cells}

    gold = table([[("item", None), ("2025", None)],
                  [("cash", None), ("2,002,429", "2002429")],
                  [("bad debt", None), ("△2,979", "-2979")],
                  [("total", None), ("5,863,684", "5863684")]])
    assert summarize({"d": eval_doc({"tables": [gold]}, {"tables": [gold]})})["numeric_exact_acc"] == 1.0
    assert teds(gold, gold) == 1.0

    # dropped row: only that row is lost, the rows after it still align
    dropped = table([[("item", None), ("2025", None)],
                     [("cash", None), ("2,002,429", "2002429")],
                     [("total", None), ("5,863,684", "5863684")]])
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": [dropped]})})
    assert s["rows_missing"] == 1 and s["numeric_exact_acc"] == round(2 / 3, 4), s

    # column shift: value moved into column 2
    shifted = table([[("item", None), ("2025", None), ("", None)],
                     [("cash", None), ("", None), ("2,002,429", "2002429")],
                     [("bad debt", None), ("△2,979", "-2979"), ("", None)],
                     [("total", None), ("5,863,684", "5863684"), ("", None)]])
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": [shifted]})})
    assert s["column_shifts"] == 1 and s["teds"] < 1.0, s

    assert s["numeric_errors_unflagged"] == 1, s
    flagged = dict(shifted, needs_review=True)  # same error, but sent to review -> not silent
    assert summarize({"d": eval_doc({"tables": [gold]}, {"tables": [flagged]})})["numeric_errors_unflagged"] == 0

    # missed table entirely
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": []})})
    assert s["table_recall"] == 0.0 and s["numeric_exact_acc"] == 0.0 and s["numeric_errors_unflagged"] == 3, s

    # gold without regions/values (ParseBench): matched by text, digits compared on raw text
    pb = {"cells": [{k: v for k, v in c.items() if k != "value"} for c in gold["cells"]]}
    pred = dict(gold, cells=[dict(c, raw_text="2,002,428") if c["raw_text"] == "2,002,429" else c for c in gold["cells"]])
    s = summarize({"d": eval_doc({"tables": [pb]}, {"tables": [pred]})})
    # digit cells: header "2025" + 3 values; one changed digit -> 1 error out of 4
    assert s["table_recall"] == 1.0 and s["numeric_errors"] == 1 and s["numeric_exact_acc"] == 0.75, s

    # same value in two columns, both in place: never a shift (even when the value is wrong)
    dup = table([[("item", None), ("prior", None), ("current", None)],
                 [("shares", None), ("1,123", "1123"), ("1,123", "1123")]])
    no_values = dict(dup, cells=[{k: v for k, v in c.items() if k != "value"} for c in dup["cells"]])
    s = summarize({"d": eval_doc({"tables": [dup]}, {"tables": [no_values]})})
    assert s["column_shifts"] == 0 and s["numeric_errors"] == 2, s
    moved = table([[("item", None), ("prior", None), ("current", None)],
                   [("shares", None), ("", None), ("1,123", "1123")]])  # col 1 really lost it
    assert summarize({"d": eval_doc({"tables": [dup]}, {"tables": [moved]})})["column_shifts"] == 1

    # a page flagged as a whole: its numbers are flagged even with no table output
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": [], "pages": [{"page": 1, "flags": ["no_text_layer"]}]})})
    assert s["numeric_errors"] == 3 and s["numeric_errors_unflagged"] == 0, s
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": [], "pages": [{"page": 2, "flags": ["x"]}]})})
    assert s["numeric_errors_unflagged"] == 3, s  # a flag on another page does not count

    # review share: every numeric cell that is flagged, right or wrong
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": [dict(gold, needs_review=True)]})})
    assert s["numbers"] == 3 and s["numbers_flagged"] == 3 and s["review_pct"] == 100.0, s
    one = dict(gold, cells=[dict(c, flags=["x"]) if c["raw_text"] == "△2,979" else c for c in gold["cells"]])
    s = summarize({"d": eval_doc({"tables": [gold]}, {"tables": [one]})})
    assert s["numbers_flagged"] == 1 and s["numeric_errors"] == 0, s

    # split-only: same row text, cells split differently -> wrong at cell level, but split-only
    g2 = {"cells": [{"row": 0, "col": 0, "raw_text": "EUR 20,000,000"}, {"row": 0, "col": 1, "raw_text": "51.00 Fincantieri"}]}
    p2 = {"cells": [{"row": 0, "col": k, "raw_text": t} for k, t in enumerate(["EUR", "20,000,000", "51.00", "Fincantieri"])]}
    s = summarize({"d": eval_doc({"tables": [g2]}, {"tables": [p2]})})
    assert s["numeric_errors"] == 2 and s["numeric_errors_split_only"] == 2 and s["numeric_errors_unflagged_split_only"] == 2, s
    p3 = {"cells": [dict(c, raw_text="20,000,001") if c["raw_text"] == "20,000,000" else c for c in p2["cells"]]}
    s = summarize({"d": eval_doc({"tables": [g2]}, {"tables": [p3]})})
    assert s["numeric_errors_split_only"] == 0, s  # a changed digit is a real error, never split-only
    print("self-check ok")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _selftest()
        sys.exit(0)
    report = run(sys.argv[1], sys.argv[2], with_teds="--no-teds" not in sys.argv)
    out = json.dumps(report, ensure_ascii=False, indent=2)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(out)
    print(json.dumps(report["summary"], indent=2))
