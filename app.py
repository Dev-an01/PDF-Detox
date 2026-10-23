"""Terminal app for the table parser.

    python app.py                              menu: 1 single file, 2 several files / folder, 3 exit
    python app.py FILE_OR_FOLDER ... [--format md|html|json]    same, without the menu

Each PDF is saved next to itself as <name>_parsed.md (default), .html, or .json (the full parser
output, design.md §2 schema: every cell with raw_text, value, bbox, flags).
Flagged cells (anything not verified: OCR not confirmed by two engines, totals that do not add up,
possible column shift, ...) are marked ⚠ in the terminal, the md and the html, with a flag list at the end.
OCR pages take about 2 minutes each; their progress is shown.
"""
import html
import json
import sys
import threading
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import parser as P  # noqa: E402  (src/parser.py)

WARN = "⚠"
MAX_CELL = 28  # display columns per cell in the terminal (files keep the full text)


# ---------------------------------------------------------------- flags
def cell_flags(cell, table, page_flags):
    """Why a cell needs review: its own flags, its table's hard issues, its page's flags."""
    out = list(cell.get("flags", []))
    if table.get("needs_review"):
        out += [i["check"] for i in table["issues"] if i["check"] in P.TABLE_HARD]
    out += page_flags.get(cell.get("page"), [])
    return sorted(set(out))


def table_grid(table, page_flags):
    """-> rows of display strings (spanned positions empty), flag list [(row, col, text, reasons)]."""
    grid = [["" for _ in range(table["n_cols"])] for _ in range(table["n_rows"])]
    flagged = []
    for c in table["cells"]:
        reasons = cell_flags(c, table, page_flags)
        text = c["raw_text"]
        if reasons and (c.get("value") is not None or c["flags"]):  # numbers, and cells with their own flags
            flagged.append((c["row"], c["col"], text, reasons))
            text = f"{WARN} {text}"
        if c["row"] < table["n_rows"] and c["col"] < table["n_cols"]:
            grid[c["row"]][c["col"]] = text
    return grid, flagged


def page_flag_map(doc):
    return {p["page"]: list(p["flags"]) for p in doc["pages"]}


# ---------------------------------------------------------------- terminal
def width(s):
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def clip(s, n):
    out, w = "", 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > n - 1 and width(s) > n:
            return out + "…"
        out, w = out + ch, w + cw
    return out


def print_table(table, page_flags):
    grid, _ = table_grid(table, page_flags)
    grid = [[clip(" ".join(t.split()), MAX_CELL) for t in r] for r in grid]
    widths = [max([width(r[k]) for r in grid] + [1]) for k in range(table["n_cols"])]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    for i, r in enumerate(grid):
        print("| " + " | ".join(t + " " * (widths[k] - width(t)) for k, t in enumerate(r)) + " |")
        if i == table["header_rows"] - 1:
            print(line.replace("-", "="))
    print(line)


def describe(table):
    bits = [f"page {', '.join(str(r['page']) for r in table['regions'])}", f"{table['n_rows']} x {table['n_cols']}"]
    if table.get("title"):
        bits.append(table["title"])
    if table.get("unit"):
        u = table["unit"]
        bits.append(f"unit: {u['raw_text']} (scale {u['scale']}, {u['currency']}, not applied)")
    if table.get("needs_review"):
        bits.append(f"{WARN} table needs review")
    return " | ".join(bits)


def page_sources(doc):
    return [(p["page"], p["text_source"], p["flags"]) for p in doc["pages"]]


def all_flags(doc):
    pf = page_flag_map(doc)
    out = []
    for t in doc["tables"]:
        _, fl = table_grid(t, pf)
        out += [(t["table_id"], r, c, text, reasons) for r, c, text, reasons in fl]
    return out


def page_items(doc):
    """-> [(page, [block...])], every table appearing once at its first block (later pages of a
    table that continues across pages get a 'continued' marker block)."""
    tables = {t["table_id"]: t for t in doc["tables"]}
    placed, out = set(), []
    for pg in doc["pages"]:
        blocks = []
        for b in pg.get("blocks", []):
            if b["type"] == "table":
                tid = b["table_id"]
                if tid in placed or b.get("continued"):
                    blocks.append({"type": "continued", "table_id": tid})
                    continue
                placed.add(tid)
                blocks.append(dict(b, table=tables[tid]))
            else:
                blocks.append(b)
        out.append((pg, blocks))
    rest = [t for tid, t in tables.items() if tid not in placed]  # safety: no table is ever dropped
    if rest and out:
        out[-1][1].extend({"type": "table", "table_id": t["table_id"], "table": t} for t in rest)
    return out


LABEL = {"unit": "unit", "caption": "caption", "footnote": "note", "furniture": "page furniture", "figure": "figure"}


def block_line(b):
    """One non-table block as plain text for the terminal."""
    mark = f"{WARN} " if b.get("flags") else ""
    if b["type"] == "heading":
        return f"{'#' * b.get('level', 1)} {mark}{b['text']}"
    if b["type"] in LABEL:
        return f"[{LABEL[b['type']]}] {mark}{b['text']}"
    return f"{mark}{b['text']}"


def print_doc(doc):
    print("\npages:")
    for pno, src, fl in page_sources(doc):
        print(f"  page {pno}: {src}" + (f"   {WARN} {', '.join(fl)}" if fl else ""))
    pf = page_flag_map(doc)
    for pg, blocks in page_items(doc):
        print(f"\n---- page {pg['page']} ({pg['text_source']}) ----")
        for b in blocks:
            if b["type"] == "table":
                t = b["table"]
                print(f"\n{t['table_id']}: {describe(t)}")
                print_table(t, pf)
                print()
            elif b["type"] == "continued":
                print(f"({b['table_id']} continues here; its rows are shown with the table above)")
            else:
                print(block_line(b))
    flags = all_flags(doc)
    print(f"\nflagged cells: {len(flags)}")
    for tid, r, c, text, reasons in flags[:40]:
        print(f"  {WARN} {tid} row {r} col {c}: {text!r}  ({', '.join(reasons)})")
    if len(flags) > 40:
        print(f"  ... {len(flags) - 40} more in the saved file")


# ---------------------------------------------------------------- files
def esc_md(s):
    return " ".join(s.split()).replace("|", "\\|")


def to_markdown(doc):
    pf = page_flag_map(doc)
    out = [f"# {doc['doc_id']}", "", f"Source: `{doc['source_file']}`", "", "## Pages", ""]
    out += [f"- page {p}: {src}" + (f" {WARN} {', '.join(fl)}" if fl else "") for p, src, fl in page_sources(doc)]
    for pg, blocks in page_items(doc):
        out += ["", f"## Page {pg['page']} ({pg['text_source']})", ""]
        for b in blocks:
            if b["type"] == "table":
                t = b["table"]
                grid, _ = table_grid(t, pf)
                out += ["", f"**{t['table_id']}**: {describe(t)}", ""]
                head = grid[: max(t["header_rows"], 1)]
                # one markdown header row: stacked header rows joined with " / " (html keeps the real spans)
                hdr = [" / ".join(x for x in (row[k] for row in head) if x) for k in range(t["n_cols"])]
                out.append("| " + " | ".join(esc_md(h) for h in hdr) + " |")
                out.append("|" + "---|" * t["n_cols"])
                for row in grid[max(t["header_rows"], 1):]:
                    out.append("| " + " | ".join(esc_md(x) for x in row) + " |")
                out.append("")
            elif b["type"] == "continued":
                out += [f"*({b['table_id']} continues on this page; its rows are in the table above)*", ""]
            elif b["type"] == "heading":
                out += [f"{'#' * min(2 + b.get('level', 1), 6)} {(WARN + ' ') if b.get('flags') else ''}{esc_md(b['text'])}", ""]
            elif b["type"] == "text":
                out += [f"{(WARN + ' OCR: ') if b.get('flags') else ''}{esc_md(b['text'])}", ""]
            else:
                out += [f"*[{LABEL.get(b['type'], b['type'])}]* {(WARN + ' ') if b.get('flags') else ''}{esc_md(b['text'])}", ""]
    flags = all_flags(doc)
    out += ["", "## Flagged cells", "", f"{len(flags)} cells need review." if flags else "None."]
    out += [f"- {WARN} {tid} row {r} col {c}: `{esc_md(text)}` ({', '.join(reasons)})" for tid, r, c, text, reasons in flags]
    return "\n".join(out) + "\n"


CSS = """
:root { --bg:#ffffff; --fg:#1d1d1b; --muted:#6b6b66; --line:#d9d8d2; --flag:#fff1cc; --flagline:#d9a23a; --head:#f3f2ee; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29c; --line:#3a3a36;
  --flag:#4a3a12; --flagline:#e3b453; --head:#222220; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, "Hiragino Sans", "Segoe UI", sans-serif;
  margin:0; padding:24px 16px; } main { max-width:1100px; margin:0 auto; }
h1 { font-size:20px; } h2 { font-size:16px; margin-top:28px; } .muted { color:var(--muted); }
.wrap { overflow-x:auto; } table { border-collapse:collapse; margin:8px 0; }
td, th { border:1px solid var(--line); padding:4px 8px; vertical-align:top; } th { background:var(--head); text-align:center; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.flag { background:var(--flag); outline:2px solid var(--flagline); outline-offset:-2px; }
section.page { border-top:1px solid var(--line); margin-top:18px; } .blk { margin:4px 0; }
.tag { font-size:11px; color:var(--muted); border:1px solid var(--line); border-radius:3px; padding:0 4px; margin-right:4px; }
.furniture { color:var(--muted); font-size:12px; }
"""


def to_html(doc):
    pf = page_flag_map(doc)
    e = html.escape
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>{e(doc['doc_id'])} parsed</title><style>{CSS}</style></head><body><main>",
           f"<h1>{e(doc['doc_id'])}</h1><p class='muted'>Source: {e(doc['source_file'])}. Cells marked {WARN} need review.</p>",
           "<h2>Pages</h2><ul>"]
    out += [f"<li>page {p}: {e(src)}" + (f" {WARN} {e(', '.join(fl))}" if fl else "") + "</li>" for p, src, fl in page_sources(doc)]
    out.append("</ul>")
    for pg, blocks in page_items(doc):
        out.append(f"<section class='page'><h2>Page {pg['page']} <span class='muted'>({e(pg['text_source'])})</span></h2>")
        for b in blocks:
            warn = f"{WARN} " if b.get("flags") else ""
            if b["type"] == "table":
                out.append(table_html(b["table"], pf))
            elif b["type"] == "continued":
                out.append(f"<p class='muted'>({e(b['table_id'])} continues on this page; its rows are in the table above)</p>")
            elif b["type"] == "heading":
                lv = min(2 + b.get("level", 1), 6)
                out.append(f"<h{lv} class='blk'>{warn}{e(b['text'])}</h{lv}>")
            elif b["type"] == "text":
                out.append(f"<p class='blk'>{warn}{e(b['text'])}</p>")
            else:
                out.append(f"<p class='blk {b['type']}'><span class='tag'>{e(LABEL.get(b['type'], b['type']))}</span> {warn}{e(b['text'])}</p>")
        out.append("</section>")
    flags = all_flags(doc)
    out.append(f"<h2>Flagged cells ({len(flags)})</h2><ul>")
    out += [f"<li>{WARN} {e(tid)} row {r} col {c}: <code>{e(text)}</code> ({e(', '.join(reasons))})</li>" for tid, r, c, text, reasons in flags]
    out.append("</ul></main></body></html>")
    return "\n".join(out)


def table_html(t, pf):
    e = html.escape
    out = [f"<h3>{e(t['table_id'])}</h3><p class='muted'>{e(describe(t))}</p><div class='wrap'><table>"]
    rows = {}
    for c in t["cells"]:
        rows.setdefault(c["row"], []).append(c)
    for r in sorted(rows):
        out.append("<tr>")
        for c in sorted(rows[r], key=lambda c: c["col"]):
            reasons = cell_flags(c, t, pf)
            show = reasons and (c.get("value") is not None or c["flags"])
            tag = "th" if c["is_header"] else "td"
            cls = " ".join(x for x in ("num" if c.get("value") is not None else "", "flag" if show else "") if x)
            span = (f" rowspan='{c['row_span']}'" if c["row_span"] > 1 else "") + (f" colspan='{c['col_span']}'" if c["col_span"] > 1 else "")
            title = f" title='{e(', '.join(reasons))}'" if show else ""
            text = (f"{WARN} " if show else "") + e(c["raw_text"])
            out.append(f"<{tag}{span}{' class=' + repr(cls) if cls else ''}{title}>{text}</{tag}>")
        out.append("</tr>")
    out.append("</table></div>")
    return "\n".join(out)


# ---------------------------------------------------------------- run one file
class Progress:
    """Prints page progress; while an OCR page runs (about 2 min) a ticker shows the elapsed time."""

    def __init__(self, name):
        self.name, self.ticker, self.stop, self.t0 = name, None, threading.Event(), None

    def __call__(self, event, page, detail):
        if event == "ocr_start":
            print(f"  page {page}: OCR ({', '.join(detail)}), LightOnOCR + Tesseract, about 2 min ...", flush=True)
            self.t0, self.stop = time.perf_counter(), threading.Event()
            self.ticker = threading.Thread(target=self._tick, args=(page,), daemon=True)
            self.ticker.start()
        elif event == "ocr_done":
            self.stop.set()
            if self.ticker:
                self.ticker.join()
            state = "cannot be verified, page flagged" if detail.get("page_unverified") else "done"
            print(f"\r  page {page}: OCR {state} in {time.perf_counter() - self.t0:.0f}s" + " " * 20, flush=True)

    def _tick(self, page):
        while not self.stop.wait(5):
            print(f"\r  page {page}: OCR running ... {time.perf_counter() - self.t0:.0f}s", end="", flush=True)


def parse_one(path, fmt, show=True):
    """Parse and save one PDF. -> summary dict; raises on a bad file (the caller reports it)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"not a PDF: {path.name}")
    t0 = time.perf_counter()
    print(f"\nparsing {path.name} ...", flush=True)
    doc = P.parse(str(path), progress=Progress(path.name))
    secs = time.perf_counter() - t0
    if show:
        print_doc(doc)
    out = path.with_name(f"{path.stem}_parsed.{fmt}")
    text = {"md": to_markdown, "html": to_html, "json": lambda d: json.dumps(d, ensure_ascii=False, indent=1)}[fmt](doc)
    out.write_text(text, encoding="utf-8")
    return {"file": path.name, "pages": len(doc["pages"]), "tables": len(doc["tables"]), "flagged": len(all_flags(doc)),
            "ocr_pages": sum(p["text_source"] != "text_layer" for p in doc["pages"]), "seconds": secs, "saved": str(out)}


def summary_line(s):
    return (f"  {s['file']}: {s['pages']} pages, {s['tables']} tables, {s['flagged']} flagged cells, "
            f"{s['ocr_pages']} OCR pages, {s['seconds']:.1f}s -> {s['saved']}")


def expand(paths):
    """Folder -> its PDFs (sorted); files stay as given (non-PDFs are reported, not skipped silently)."""
    out = []
    for p in paths:
        p = Path(p).expanduser()
        out += sorted(x for x in p.iterdir() if x.is_file() and not x.name.startswith(".") and "_parsed." not in x.name) if p.is_dir() else [p]
    return out


def parse_many(paths, fmt):
    results, errors = [], []
    files = expand(paths)
    print(f"\n{len(files)} files")
    for f in files:
        try:
            results.append(parse_one(f, fmt, show=False))
            print(summary_line(results[-1]))
        except Exception as e:  # one bad file never stops the others
            errors.append((Path(f).name, e))
            print(f"  ERROR {Path(f).name}: {type(e).__name__}: {e}")
    print("\nsummary:")
    for s in results:
        print(summary_line(s))
    for name, e in errors:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
    return results, errors


# ---------------------------------------------------------------- menu
def ask_format():
    f = input("format, md, html or json [md]: ").strip().lower() or "md"
    if f not in ("md", "html", "json"):
        print(f"unknown format {f!r}, using md")
        f = "md"
    return f


def menu():
    while True:
        print("\n1  parse a single file\n2  parse several files (a folder or paths)\n3  exit")
        choice = input("> ").strip()
        if choice == "1":
            path = input("PDF path: ").strip().strip("'\"")
            fmt = ask_format()
            try:
                s = parse_one(path, fmt)
                print("\n" + summary_line(s))
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
        elif choice == "2":
            print("a folder, or PDF paths one per line; empty line to start")
            paths = []
            while True:
                line = input("path: ").strip().strip("'\"")
                if not line:
                    break
                paths.append(line)
            if not paths:
                print("no paths given")
                continue
            fmt = ask_format()
            try:
                parse_many(paths, fmt)
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
        elif choice == "3":
            print("bye")
            return
        else:
            print("choose 1, 2 or 3")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        try:
            menu()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
    else:
        fmt = args[args.index("--format") + 1] if "--format" in args else "md"
        paths = [a for i, a in enumerate(args) if a != "--format" and (i == 0 or args[i - 1] != "--format")]
        if len(paths) == 1 and Path(paths[0]).is_file():
            try:
                print(summary_line(parse_one(paths[0], fmt)))
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
                sys.exit(1)
        else:
            sys.exit(1 if parse_many(paths, fmt)[1] else 0)
