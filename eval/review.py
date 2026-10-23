"""Check the parser by eye.

    python eval/review.py PDF [--page N] [--gold GOLD.json] [--pred PRED.json] [--out page_review.html]
        One page: the page image on the left, the parsed tables on the right. Flagged cells are amber;
        with --gold, wrong cells are red (wrong and flagged: red with amber border). Click a cell to see
        its crop from the page next to the parsed text and value.

    python eval/review.py --sample50 [--split test] [--seed 7] [--pred-dir preds/final_pb] [--out numbers_check.html]
        50 random numbers from ParseBench (test by default), fixed seed: the crop of each number from the
        page next to the parsed value, so a person can tick right / wrong. The page counts the ticks.

Self-contained HTML (images embedded), no libraries.
"""
import base64
import html
import json
import random
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
import evaluate as ev  # noqa: E402

DPI = 110

CSS = """
:root { --bg:#fbfbf9; --fg:#1d1d1b; --muted:#6b6b66; --line:#dcdbd5; --card:#fff; --flag:#fff1cc; --flagline:#d9a23a;
  --wrong:#fde2e1; --wrongline:#d64541; --sel:#2f6fb3; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --fg:#ecebe6; --muted:#a3a29c; --line:#34342f; --card:#1d1d1b;
  --flag:#4a3a12; --flagline:#e3b453; --wrong:#4d1d1b; --wrongline:#f0625d; --sel:#5b9be0; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 -apple-system, "Hiragino Sans", sans-serif; }
header { padding:14px 16px; border-bottom:1px solid var(--line); } h1 { font-size:17px; margin:0 0 4px; } .muted { color:var(--muted); }
.split { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; padding:12px 16px; }
@media (max-width:900px) { .split { grid-template-columns:1fr; } }
.pane { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; overflow:auto; max-height:88vh; }
.pagewrap { position:relative; } .pagewrap img { width:100%; display:block; }
.box { position:absolute; border:2px solid var(--sel); pointer-events:none; display:none; }
table { border-collapse:collapse; margin:6px 0 14px; } td, th { border:1px solid var(--line); padding:2px 6px; cursor:pointer; }
th { background:rgba(127,127,127,.12); } td.num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
.flag { background:var(--flag); } .wrong { background:var(--wrong); outline:2px solid var(--wrongline); outline-offset:-2px; }
.wrong.flag { outline-color:var(--flagline); }
#detail { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line); padding:6px 0 8px; margin-bottom:6px; }
.crop { background-repeat:no-repeat; border:1px solid var(--line); display:inline-block; vertical-align:middle; }
.legend span { display:inline-block; padding:1px 6px; margin-right:6px; border-radius:3px; }
.grid50 { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:10px; padding:12px 16px; }
.item { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; }
.item img { max-width:100%; display:block; margin:4px 0; border:1px solid var(--line); background:#fff; }
.item code { font-size:14px; } .tick label { margin-right:10px; cursor:pointer; }
"""


def b64png(pix):
    return "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()


def wrong_cells(gold, pred):
    """{(table index, row, col)} of predicted cells whose gold number they fail (same alignment as evaluate.py)."""
    out = set()
    for i, j in ev.match_tables(gold["tables"], pred["tables"]):
        G, P = ev.grid(gold["tables"][i]), ev.grid(pred["tables"][j])
        pred_rows = sorted({c["row"] for c in pred["tables"][j]["cells"]})
        sm = ev.SequenceMatcher(None, [ev.row_sig(r) for r in G], [ev.row_sig(r) for r in P], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            n = min(i2 - i1, j2 - j1) if tag == "replace" else (i2 - i1 if tag == "equal" else 0)
            for k in range(n):
                g, p = G[i1 + k], P[j1 + k]
                for col, gc in g.items():
                    if ev.is_num(gc) and not ev.num_ok(gc, p.get(col)) and col in p:
                        out.add((j, pred_rows[j1 + k], col))
    return out


def page_review(pdf, page_no, gold_path, pred_path, out):
    import parser as P
    pred = json.loads(Path(pred_path).read_text()) if pred_path else P.parse(str(pdf))
    gold = json.loads(Path(gold_path).read_text()) if gold_path else None
    wrong = wrong_cells(gold, pred) if gold else set()
    page = pymupdf.open(pdf)[page_no - 1]
    pix = page.get_pixmap(dpi=DPI)
    img, s = b64png(pix), DPI / 72.0
    pflags = {p["page"]: p["flags"] for p in pred["pages"]}
    parts, e = [], html.escape
    for ti, t in enumerate(pred["tables"]):
        if all(r["page"] != page_no for r in t["regions"]):
            continue
        parts.append(f"<h3>{e(t['table_id'])} <span class='muted'>{t['n_rows']}x{t['n_cols']} {e(t.get('title', ''))}"
                     f"{' | needs review: ' + e(', '.join(i['check'] for i in t['issues'] if i['check'] not in ('totals',))) if t['needs_review'] else ''}</span></h3><table>")
        rows = {}
        for c in t["cells"]:
            rows.setdefault(c["row"], []).append(c)
        for r in sorted(rows):
            parts.append("<tr>")
            for c in sorted(rows[r], key=lambda c: c["col"]):
                flags = list(c["flags"]) + ([i["check"] for i in t["issues"] if t["needs_review"] and i["check"] not in ("totals",)])
                flags += pflags.get(c.get("page"), [])
                cls = [x for x, on in (("num", c.get("value") is not None), ("flag", bool(flags) and c.get("value") is not None),
                                      ("wrong", (ti, c["row"], c["col"]) in wrong)) if on]
                bb = c.get("bbox") if c.get("page") == page_no else None
                data = (f" data-bb='{json.dumps([round(v * s, 1) for v in bb])}'" if bb else "")
                info = json.dumps({"text": c["raw_text"], "value": c.get("value"), "flags": sorted(set(flags)),
                                   "wrong": (ti, c["row"], c["col"]) in wrong, "source": c.get("source")}, ensure_ascii=False)
                tag = "th" if c["is_header"] else "td"
                span = (f" rowspan={c['row_span']}" if c["row_span"] > 1 else "") + (f" colspan={c['col_span']}" if c["col_span"] > 1 else "")
                parts.append(f"<{tag}{span} class='{' '.join(cls)}'{data} data-info='{e(info)}' onclick='pick(this)'>{e(c['raw_text'])}</{tag}>")
            parts.append("</tr>")
        parts.append("</table>")
    js = f"""<script>
const IMG = "{img}", W = {pix.width}, H = {pix.height};
function pick(el) {{
  const info = JSON.parse(el.dataset.info), box = document.getElementById('box'), d = document.getElementById('detail');
  let crop = '<span class="muted">no box (cell not located on this page)</span>';
  if (el.dataset.bb) {{
    const [x0, y0, x1, y1] = JSON.parse(el.dataset.bb), pad = 6, w = x1 - x0 + 2 * pad, h = y1 - y0 + 2 * pad, z = 2;
    crop = `<span class="crop" style="width:${{w * z}}px;height:${{h * z}}px;background-image:url(${{IMG}});` +
           `background-size:${{W * z}}px ${{H * z}}px;background-position:-${{(x0 - pad) * z}}px -${{(y0 - pad) * z}}px"></span>`;
    const img = document.querySelector('.pagewrap img'), k = img.clientWidth / W;
    Object.assign(box.style, {{display: 'block', left: x0 * k + 'px', top: y0 * k + 'px', width: (x1 - x0) * k + 'px', height: (y1 - y0) * k + 'px'}});
    box.scrollIntoView({{block: 'nearest'}});
  }} else box.style.display = 'none';
  d.innerHTML = crop + ` &nbsp; parsed <code>${{info.text.replace(/</g, '&lt;')}}</code> value <code>${{info.value}}</code>` +
    (info.flags.length ? ` &nbsp; ⚠ ${{info.flags.join(', ')}}` : '') + (info.wrong ? ' &nbsp; <b style="color:var(--wrongline)">wrong vs gold</b>' : '') +
    (info.source ? ` &nbsp; <span class="muted">${{info.source}}</span>` : '');
}}
</script>"""
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Page Review</title><style>{CSS}</style></head><body>
<header><h1>{e(Path(pdf).name)}, page {page_no}</h1><div class="muted legend">source: {e(next((p['text_source'] for p in pred['pages'] if p['page'] == page_no), '?'))}
 {('| page flags: ' + e(', '.join(pflags.get(page_no, [])))) if pflags.get(page_no) else ''} &nbsp;
 <span class="flag">flagged</span>{'<span class="wrong">wrong vs gold</span>' if gold else ''} click a cell to see its crop</div></header>
<div class="split"><div class="pane"><div class="pagewrap"><img src="{img}" alt="page {page_no}"><div id="box" class="box"></div></div></div>
<div class="pane"><div id="detail" class="muted">click a cell</div>{''.join(parts) or '<p class="muted">no tables on this page</p>'}</div></div>
{js}</body></html>"""
    Path(out).write_text(doc, encoding="utf-8")
    return out


def sample50(split, seed, pred_dir, out, n=50):
    stems = json.loads((ROOT / "eval" / "parsebench_split.json").read_text())[split]
    pool = []
    for s in stems:
        f = Path(pred_dir) / f"{s}.json"
        if not f.exists():
            continue
        pred = json.loads(f.read_text())
        gold = json.loads((ROOT / "eval" / "parsebench_gold" / f"{s}.json").read_text())
        wrong = wrong_cells(gold, pred)
        pflags = {p["page"]: p["flags"] for p in pred["pages"]}
        for ti, t in enumerate(pred["tables"]):
            for c in t["cells"]:
                if c.get("value") is not None and c.get("bbox"):
                    flags = sorted(set(c["flags"] + ([i["check"] for i in t["issues"] if t["needs_review"] and i["check"] != "totals"])
                                       + pflags.get(c["page"], [])))
                    pool.append((s, c, flags, (ti, c["row"], c["col"]) in wrong))
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    e, items = html.escape, []
    for k, (s, c, flags, auto_wrong) in enumerate(pick, 1):
        page = pymupdf.open(ROOT / "parsebench" / "docs" / "table" / f"{s}.pdf")[c["page"] - 1]
        r = pymupdf.Rect(c["bbox"]) + (-30, -6, 30, 6)
        img = b64png(page.get_pixmap(dpi=200, clip=r & page.rect))
        items.append(f"""<div class="item"><div class="muted">#{k} · {e(s[:46])}</div><img src="{img}" alt="crop {k}">
<div>parsed <code>{e(c['raw_text'])}</code> → value <code>{e(str(c['value']))}</code></div>
<div class="muted">{'⚠ ' + e(', '.join(flags)) if flags else 'not flagged'} · eval vs ParseBench gold: {'<b>mismatch</b>' if auto_wrong else 'ok or not compared'}</div>
<div class="tick"><label><input type="radio" name="q{k}" value="ok" onchange="count()"> right</label>
<label><input type="radio" name="q{k}" value="bad" onchange="count()"> wrong</label></div></div>""")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Numbers Check</title><style>{CSS}</style></head><body>
<header><h1>{len(pick)} random numbers from ParseBench {e(split)} (seed {seed})</h1>
<div class="muted">The crop is the page around the number's box; the parsed text must match it character for character,
and the value is the number as printed (signs, no scale). Tick each one. <b id="tally">0 right, 0 wrong, {len(pick)} left</b></div></header>
<div class="grid50">{''.join(items)}</div>
<script>function count() {{ const v = [...document.querySelectorAll('input:checked')].map(x => x.value);
const ok = v.filter(x => x == 'ok').length, bad = v.length - ok;
document.getElementById('tally').textContent = `${{ok}} right, ${{bad}} wrong, ${{{len(pick)} - v.length}} left`; }}</script>
</body></html>"""
    Path(out).write_text(doc, encoding="utf-8")
    return out, len(pool)


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    if "--sample50" in sys.argv:
        out, n = sample50(_arg("--split", "test"), int(_arg("--seed", 7)), _arg("--pred-dir", str(ROOT / "preds" / "final_pb")),
                          _arg("--out", "numbers_check.html"))
        print(f"{out}  (50 numbers sampled from {n} parsed numbers with a box)")
    elif len(sys.argv) > 1:
        print(page_review(sys.argv[1], int(_arg("--page", 1)), _arg("--gold"), _arg("--pred"), _arg("--out", "page_review.html")))
    else:
        print(__doc__)
