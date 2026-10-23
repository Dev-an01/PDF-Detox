"""docs/scores.jsonl -> docs/scores.html: one page to see whether the parser gets safer, milestone by milestone.

    python eval/dashboard.py        (also run automatically by eval/scoreboard.py)

Every number is a share of the numbers in the gold tables of that set, so gold, ParseBench dev
and ParseBench test compare directly:
  correct      exact value as printed
  caught       wrong, but flagged for review (a person will see it)
  slipped      wrong and NOT flagged (the dangerous one; target 0)
  sent to review  every flagged number, right or wrong
Plain HTML + inline SVG, no libraries.
"""
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = [("gold all", "Gold (23 docs)"), ("PB dev", "ParseBench dev"), ("PB test", "ParseBench test")]
LINE_SETS = [("gold all", "Gold", "gold"), ("PB dev", "Dev", "dev"), ("PB test", "Test", "test")]


def shares(r):
    n = r.get("numbers") or 0
    if not n:
        return None
    wrong, slip = r.get("numeric_errors") or 0, r.get("numeric_errors_unflagged") or 0
    so = r.get("numeric_errors_unflagged_split_only")
    return {"correct": 100 * (n - wrong) / n, "caught": 100 * (wrong - slip) / n, "slipped": 100 * slip / n,
            "real_slipped": None if so is None else 100 * (slip - so) / n,
            "review": 100 * (r.get("numbers_flagged") or 0) / n}


def load():
    rows = [json.loads(l) for l in (ROOT / "docs" / "scores.jsonl").read_text().splitlines() if l.strip()]
    order, data, notes = [], {}, {}
    for r in rows:
        if r["milestone"] not in order:
            order.append(r["milestone"])
        data[(r["milestone"], r["set"])] = shares(r)
        notes[r["milestone"]] = r.get("note", "")
    return order, data, notes


def move(before, after, higher_is_better=True):
    d = after - before
    if abs(d) < 0.5:
        return "flat", 0
    good = (d > 0) == higher_is_better
    return f"{'up' if d > 0 else 'down'} {abs(d):.1f} pts ({'better' if good else 'worse'})", 1 if good else -1


def verdict(dev, test):
    if dev == 0 and test == 0:
        return "no change"
    if dev == test:
        return "dev and test together"
    if test == 0:
        return "dev only, test flat"
    if dev == 0:
        return "test only, dev flat"
    return "dev and test OPPOSITE"


def changes(order, data, key, sets, higher_is_better=True):
    """One plain line per milestone: how `key` moved against the milestone before it, and whether
    ParseBench dev and test moved the same way."""
    lines = []
    for prev, cur in zip(order, order[1:]):
        parts, dirs = [], {}
        for s, label, *_ in sets:
            a, b = data.get((prev, s)), data.get((cur, s))
            if a and b:
                text, sign = move(a[key], b[key], higher_is_better)
                parts.append(f"{label} {text}")
                dirs[s] = sign
        v = verdict(dirs.get("PB dev", 0), dirs.get("PB test", 0))
        lines.append(f"<b>{html.escape(cur)}</b>: " + ", ".join(parts) + f" <i>[{v}]</i>")
    return lines


def pct(v, cls=""):
    return f'<td class="num {cls}">{"–" if v is None else f"{v:.1f}%"}</td>'


def summary_table(order, data, notes):
    head = ('<tr><th class="l">milestone</th><th class="l">set</th><th>numbers correct</th><th>wrong, but flagged</th>'
            '<th class="bad">wrong and NOT flagged</th><th class="bad">…not counting split-only</th><th>sent to review</th></tr>')
    body = []
    for m in order:
        for k, (s, label) in enumerate(SETS):
            d = data.get((m, s)) or {}
            slip = d.get("slipped")
            first = k == 0
            ms = (f'<th class="ms" rowspan="{len(SETS)}">{html.escape(m)}<div class="note">{html.escape(notes[m])}</div></th>'
                  if first else "")
            body.append(f'<tr class="{"first" if first else ""}">{ms}<td class="set">{label}</td>' + pct(d.get("correct")) +
                        pct(d.get("caught")) + pct(slip, "bad" if slip and slip >= 0.05 else "zero") +
                        pct(d.get("real_slipped"), "bad" if (d.get("real_slipped") or 0) >= 0.05 else "zero") +
                        pct(d.get("review"), "over" if (d.get("review") or 0) > 50 else "") + "</tr>")
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def stacked_bars(order, data, s, title):
    W, H, pad, top = 300, 190, 30, 10
    bw = (W - pad - 10) / max(len(order), 1)
    out = [f'<svg viewBox="0 0 {W} {H + 40}" role="img" aria-label="{title}: correct, caught, slipped per milestone">']
    for y in (0, 50, 100):
        yy = top + H - H * y / 100
        out.append(f'<line x1="{pad}" x2="{W - 5}" y1="{yy}" y2="{yy}" class="grid"/>'
                   f'<text x="{pad - 4}" y="{yy + 3}" class="axis" text-anchor="end">{y}%</text>')
    for i, m in enumerate(order):
        d = data.get((m, s))
        x = pad + i * bw + bw * 0.15
        if d:
            y = top + H
            for k in ("correct", "caught", "slipped"):
                h = H * d[k] / 100
                y -= h
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.7:.1f}" height="{max(h, 0):.1f}" class="{k}">'
                           f'<title>{m} {k}: {d[k]:.1f}%</title></rect>')
        out.append(f'<text x="{x + bw * 0.35:.1f}" y="{top + H + 14}" class="axis" text-anchor="middle">{html.escape(m)}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(order, data):
    W, H, pad, top = 640, 200, 40, 10
    step = (W - pad - 70) / max(len(order) - 1, 1)  # room on the right for the line labels
    xs = [pad + i * step for i in range(len(order))]
    out = [f'<svg viewBox="0 0 {W} {H + 40}" role="img" aria-label="numbers correct per milestone, gold vs dev vs test">']
    for y in (0, 25, 50, 75, 100):
        yy = top + H - H * y / 100
        out.append(f'<line x1="{pad}" x2="{W - 10}" y1="{yy}" y2="{yy}" class="grid"/>'
                   f'<text x="{pad - 6}" y="{yy + 3}" class="axis" text-anchor="end">{y}%</text>')
    for x, m in zip(xs, order):
        out.append(f'<text x="{x:.1f}" y="{top + H + 16}" class="axis" text-anchor="middle">{html.escape(m)}</text>')
    for s, label, cls in LINE_SETS:
        pts = [(x, data[(m, s)]["correct"]) for x, m in zip(xs, order) if data.get((m, s))]
        if not pts:
            continue
        path = " ".join(f"{x:.1f},{top + H - H * v / 100:.1f}" for x, v in pts)
        out.append(f'<polyline points="{path}" class="ln {cls}"/>')
        for x, v in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{top + H - H * v / 100:.1f}" r="3.5" class="dot {cls}"><title>{label}: {v:.1f}%</title></circle>')
        lx, lv = pts[-1]
        out.append(f'<text x="{lx + 6:.1f}" y="{top + H - H * lv / 100 + 4:.1f}" class="lbl {cls}">{label}</text>')
    out.append("</svg>")
    return "".join(out)


CSS = """
:root { --bg:#fbfbf9; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --card:#ffffff;
  --ok:#2f8f5b; --caught:#d9a23a; --slip:#d64541; --gold:#9a7b18; --dev:#2f6fb3; --test:#7a4fb5; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#151514; --fg:#ecebe6; --muted:#a3a29c; --line:#2e2e2b; --card:#1d1d1b;
  --ok:#48b37a; --caught:#e3b453; --slip:#f0625d; --gold:#d1b04a; --dev:#5b9be0; --test:#a682e0; } }
:root[data-theme="dark"] { --bg:#151514; --fg:#ecebe6; --muted:#a3a29c; --line:#2e2e2b; --card:#1d1d1b;
  --ok:#48b37a; --caught:#e3b453; --slip:#f0625d; --gold:#d1b04a; --dev:#5b9be0; --test:#a682e0; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, "Hiragino Sans", "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; padding:28px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:32px 0 8px; }
.sub, .note, .caption li { color:var(--muted); } .sub { margin:0 0 20px; font-size:13px; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:6px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th.ms { text-align:left; vertical-align:top; min-width:150px; } .note { font-weight:400; font-size:11px; white-space:normal; max-width:220px; }
td.set, th.l { text-align:left; } td.set { color:var(--muted); } tr.first td, tr.first th { border-top:2px solid var(--line); }
.linecard { max-width:760px; } .captitle { margin:10px 0 0; font-size:13px; font-weight:600; } .caption i { color:var(--muted); }
th.grp { text-align:center; border-left:1px solid var(--line); }
td.num { font-variant-numeric:tabular-nums; } th.bad { color:var(--slip); }
td.bad { color:var(--slip); font-weight:700; } td.zero { color:var(--ok); } td.over { color:var(--caught); font-weight:700; }
.charts { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
.card h3 { font-size:14px; margin:0 0 6px; }
svg { width:100%; height:auto; display:block; }
.grid { stroke:var(--line); } .axis { fill:var(--muted); font-size:10px; }
rect.correct { fill:var(--ok); } rect.caught { fill:var(--caught); } rect.slipped { fill:var(--slip); }
.ln { fill:none; stroke-width:2.5; } .ln.gold { stroke:var(--gold); } .ln.dev { stroke:var(--dev); } .ln.test { stroke:var(--test); }
.dot.gold { fill:var(--gold); } .dot.dev { fill:var(--dev); } .dot.test { fill:var(--test); }
.lbl { font-size:11px; font-weight:600; } .lbl.gold { fill:var(--gold); } .lbl.dev { fill:var(--dev); } .lbl.test { fill:var(--test); }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin:6px 0 0; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.caption { margin:8px 0 0; padding-left:18px; font-size:13px; } .caption b { color:var(--fg); }
"""


def main():
    order, data, notes = load()
    bars = "".join(f'<div class="card"><h3>{label}</h3>{stacked_bars(order, data, s, label)}</div>' for s, label in SETS)
    legend = ('<div class="legend"><span><span class="sw" style="background:var(--ok)"></span>correct</span>'
              '<span><span class="sw" style="background:var(--caught)"></span>wrong, flagged (caught)</span>'
              '<span><span class="sw" style="background:var(--slip)"></span>wrong, NOT flagged (slipped through)</span></div>')
    bar_lines = changes(order, data, "slipped", [(s, l) for s, l, _ in LINE_SETS], higher_is_better=False)
    line_lines = changes(order, data, "correct", [(s, l) for s, l, _ in LINE_SETS])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Parser Scores</title>
<style>{CSS}</style></head><body><main>
<h1>Parser scores per milestone</h1>
<p class="sub">Shares of all numbers in the gold tables of each set. Generated {datetime.now():%Y-%m-%d %H:%M} from docs/scores.jsonl.
ParseBench compares printed text (its ground truth has no values). Until m4 every table is flagged, so nothing can slip
through on gold. "Not counting split-only" leaves out wrong numbers whose whole row text matches an output row on the
page (ParseBench just splits cells differently); it is lenient, so the real share lies between the two red columns.
Sent to review over 50% is shown in amber.</p>
{summary_table(order, data, notes)}
<h2>Where the numbers end up</h2>{legend}
<div class="charts">{bars}</div>
<p class="captitle">Wrong and NOT flagged (red), lower is better:</p>
<ul class="caption">{"".join(f"<li>{l}</li>" for l in bar_lines)}</ul>
<h2>Numbers correct: do dev and test move together?</h2>
<div class="card linecard">{line_chart(order, data)}</div>
<p class="captitle">Numbers correct, higher is better:</p>
<ul class="caption">{"".join(f"<li>{l}</li>" for l in line_lines)}</ul>
</main></body></html>"""
    (ROOT / "docs" / "scores.html").write_text(page)
    return ROOT / "docs" / "scores.html"


if __name__ == "__main__":
    print(main())
