"""PDF table parser (design.md). Milestones done: m1 (text layer -> characters -> candidate cells
-> rows), m2 (columns by geometry, header rows + spans, indent levels, wrapped labels).

    python src/parser.py FILE.pdf [--out DIR]      -> DIR/<stem>.json  (design.md §2 schema)

Done: m1 rows, m2 columns/headers, m5 regions, ruling-line columns, m3 values, m4 validation + page
check, m6 text-layer audit + OCR fallback (src/ocr.py; setting PARSER_OCR). Still open: terminal app (m7).
"""
import json
import os
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

GAP_EM = 1.0  # a visual gap wider than this (in em) between glyphs ends a candidate cell
PAGE_NUMBER = re.compile(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$")
# a cell that is one number as printed: △2,979  (1,234)  -40.0  12.5%  ¥1,000  1,234※2
NUMBER = re.compile(r"^[△▲+\-−]?\(?[△▲+\-−]?[¥$€£]?\s?\d[\d,.]*%?\)?(?:[※*]\d+)?$")
YEAR = re.compile(r"^(?:19|20)\d\d$")
TOTAL = re.compile(r"(合計|計$|^total\b|\btotal$|^subtotal\b)", re.I)
WRAP_FILL = 0.75  # a label that reaches this share of its column's width may continue on the next line
GAP_RUNS = 2.2  # a vertical gap over this many row pitches separates two borderless regions
RULE_COVER = 0.6  # an interior column rule must be drawn over this share of the region's height
ABSORB_PITCH = 1.6  # header rows above a ruled box join it when this close (in row pitches)
CAPTION_GAP = 40.0  # pt: a title / unit line this close above a table belongs to it
UNIT = re.compile(r"(単位|千円|百万円|in (thousands|millions|billions)|amounts? in|\(in [a-z$]|\$ ?in )", re.I)
CONTINUED = re.compile(r"(続き|続く|continued|cont'd|\(cont\.?\))", re.I)


def page_chars(page):
    """Every visible character with its box. Spaces kept: they are part of raw_text."""
    # Text boxes come in unrotated page space; on a /Rotate 90|180|270 page (landscape tables)
    # map them into display space, where page.rect and the ruled boxes live.
    m = page.rotation_matrix
    turn = pymupdf.Matrix(m.a, m.b, m.c, m.d, 0, 0)
    box = (lambda b: list(pymupdf.Rect(b) * m)) if page.rotation else (lambda b: b)
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            d = pymupdf.Point(line["dir"]) * turn if page.rotation else pymupdf.Point(line["dir"])
            if abs(d.y) > 0.1:  # rotated line: keep it whole, one unit with its own text
                text = "".join(ch["c"] for sp in line["spans"] for ch in sp["chars"]).strip()
                if text:
                    out.append({"c": text, "bbox": box(line["bbox"]), "size": line["spans"][0]["size"], "rotated": True})
                continue
            for span in line["spans"]:
                for ch in span["chars"]:
                    out.append({"c": ch["c"], "bbox": box(ch["bbox"]), "size": span["size"]})
    return out


def rows_of(chars):
    """Group characters into rows (y-centers within half a glyph height), then split each row into
    candidate cells wherever two glyphs are more than GAP_EM apart. Built from characters, not
    spans, because producers split a cell into several spans (△ | 2,979)."""
    glyphs = sorted((c for c in chars if c["c"].strip() and not c.get("rotated")),
                    key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    rows, cur, yc = [], [], None
    for c in glyphs:
        y = (c["bbox"][1] + c["bbox"][3]) / 2
        if cur and abs(y - yc) > 0.5 * c["size"]:
            rows.append(cur)
            cur = []
        if not cur:
            yc = y
        cur.append(c)
    if cur:
        rows.append(cur)
    # rotated units (vertical column headers) that overlap vertically form one row of their own
    rot = sorted((c for c in chars if c.get("rotated")), key=lambda c: c["bbox"][1])
    for c in rot:
        last = rows[-1] if rows and rows[-1][0].get("rotated") else None
        grp = next((r for r in rows if r[0].get("rotated") and c["bbox"][1] < max(x["bbox"][3] for x in r)
                    and c["bbox"][3] > min(x["bbox"][1] for x in r)), None)
        (grp.append(c) if grp else rows.append([c]))
    rows.sort(key=lambda r: min(c["bbox"][1] for c in r))
    spaces = [c for c in chars if not c["c"].strip()]
    out = []
    for row in rows:
        row.sort(key=lambda c: c["bbox"][0])
        top, bottom = min(c["bbox"][1] for c in row), max(c["bbox"][3] for c in row)
        cells, cell = [], [row[0]]
        for prev, c in zip(row, row[1:]):
            if c.get("rotated") or prev.get("rotated") or c["bbox"][0] - prev["bbox"][2] > GAP_EM * c["size"]:
                cells.append(cell)
                cell = []
            cell.append(c)
        cells.append(cell)
        out.append([_cell(cl, spaces, top, bottom) for part in cells for cl in _split_values(part)])
    return out


VALUE_WORD = re.compile(r"^[△▲+\-−(]?[\d][\d,.:/%]*\)?$")
WORD_GAP_EM = 0.2


def _split_values(glyphs):
    """Columns set closer than GAP_EM read as one candidate cell (`7:04 7:09 7:14`). If every
    word of the cell is a value (number, time, signed number), each word is its own cell.
    ponytail: breaks space-grouped thousands (`1 234`, French); ja/en only for now."""
    if len(glyphs) == 1:
        return [glyphs]
    words, w = [], [glyphs[0]]
    for prev, c in zip(glyphs, glyphs[1:]):
        if c["bbox"][0] - prev["bbox"][2] > WORD_GAP_EM * c["size"]:
            words.append(w)
            w = []
        w.append(c)
    words.append(w)
    texts = [unicodedata.normalize("NFKC", "".join(c["c"] for c in w)) for w in words]
    if len(words) > 1 and all(VALUE_WORD.match(t) for t in texts):
        return words
    return [glyphs]


def _cell(glyphs, spaces, top, bottom):
    """raw_text = the cell's glyphs plus the space characters that sit between them, in x order."""
    x0, x1 = glyphs[0]["bbox"][0], glyphs[-1]["bbox"][2]
    inner = [s for s in spaces if x0 < (s["bbox"][0] + s["bbox"][2]) / 2 < x1
             and top <= (s["bbox"][1] + s["bbox"][3]) / 2 <= bottom]
    chars = sorted(glyphs + inner, key=lambda c: c["bbox"][0])
    text = ""
    for prev, c in zip([None] + chars, chars):
        # OCR words are units without space characters between them: put one back, except between CJK
        if prev is not None and prev.get("unit") and c.get("unit") and not (_cjk(prev["c"][-1]) and _cjk(c["c"][0])):
            text += " "
        text += c["c"]
    flags = sorted({"ocr_unconfirmed" for g in glyphs if g.get("ocr_reason")})
    return {"raw_text": text.strip(), "flags": flags,
            "bbox": [round(x0, 2), round(min(c["bbox"][1] for c in glyphs), 2),
                     round(x1, 2), round(max(c["bbox"][3] for c in glyphs), 2)]}


TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


# ---------------------------------------------------------------- m3: values (design.md §3)
BLANKS = {"": "empty", "-": "dash", "—": "dash", "–": "dash", "−": "dash", "‐": "dash", "―": "dash",
          "n/a": "na", "na": "na", "*": "dash"}
FOOTNOTE_REF = re.compile(r"(?:※\s?\d+|\*\d+)$")
NUM_BODY = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def normalize(raw):
    """raw_text -> text (NFKC), value (exact Decimal string, never float), value_type, blank_kind,
    footnote_refs, flags. raw_text itself is never changed. Only a cell that is one number as a
    whole gets a value; anything else keeps value=None (labels with digits stay text).
    ja/en separators only: `,` groups thousands (exactly 3 digits), `.` is the decimal point."""
    text = unicodedata.normalize("NFKC", raw).strip()
    out = {"text": text, "value": None, "value_type": "text", "blank_kind": None, "footnote_refs": [], "flags": []}
    t = text
    if t.lower() in BLANKS:
        out.update(value_type="blank", blank_kind=BLANKS[t.lower()])
        return out
    m = FOOTNOTE_REF.search(t)
    if m and m.start() > 0:
        out["footnote_refs"] = [m.group().replace(" ", "")]
        t = t[:m.start()].rstrip()
    neg = False
    if t[:1] in "△▲":
        neg, t = True, t[1:].lstrip()
    if len(t) > 2 and t[0] == "(" and t[-1] == ")":
        neg, t = True, t[1:-1].strip()
    if t[:1] in "-−" and len(t) > 1:
        neg, t = True, t[1:].lstrip()
    elif t[:1] == "+":
        t = t[1:]
    if t[:1] in "¥$€£":
        t = t[1:].lstrip()
    pct = t.endswith("%")
    if pct:
        t = t[:-1].rstrip()
    if not NUM_BODY.match(t):
        if any(ch.isdigit() for ch in t) and re.fullmatch(r"[\d,.\s]+", t):
            out["flags"].append("ambiguous_number")  # e.g. 1,23 or 1.234,5: never guess
        return out
    try:
        v = Decimal(t.replace(",", ""))
    except InvalidOperation:
        out["flags"].append("ambiguous_number")
        return out
    if neg and v == 0:
        out["notes"] = ["negative_zero"]  # △0: printed negative zero, value 0; a note, not a review flag
        neg = False
    out["value"] = str(-v if neg else v)
    out["value_type"] = "percent" if pct else "number"
    return out


SCALES = [(r"億円", 10**8), (r"百万円", 10**6), (r"千円", 1000), (r"円", 1), (r"\bin billions\b", 10**9),
          (r"\bin millions\b", 10**6), (r"\bin thousands\b", 1000), (r"\bamounts? in\b", 1)]
CURRENCIES = [(r"円|JPY|¥", "JPY"), (r"U\.?S\.? dollars|USD|US\$|\$", "USD"), (r"EUR|€|euros?", "EUR"), (r"GBP|£", "GBP")]


def parse_unit(raw):
    """Unit line -> {raw_text, scale, currency}. Recorded, never applied to values. scale None
    when the line names no scale the parser knows (m4 flags it)."""
    t = unicodedata.normalize("NFKC", raw)
    scale = next((n for pat, n in SCALES if re.search(pat, t, re.I)), None)
    cur = next((c for pat, c in CURRENCIES if re.search(pat, t, re.I)), None)
    return {"raw_text": raw, "scale": scale, "currency": cur}


def is_number(raw):
    """One printed value: a number, or a clock time (timetables, logs)."""
    t = unicodedata.normalize("NFKC", raw).strip()
    return bool(NUMBER.match(t) or TIME.match(t))


def furniture(r, page_height):
    """A page number in the bottom margin is not content."""
    return len(r) == 1 and r[0]["bbox"][1] > 0.9 * page_height and bool(PAGE_NUMBER.match(r[0]["raw_text"]))


def table_rows(rows, page_height, keep_top=False):
    """Table rows of one region. Borderless: first .. last row with >= 2 cells (title, unit line,
    footnotes, page number are not table content). Ruled box (keep_top): every row from the top of
    the box, since a full-width row inside the rules is part of the table."""
    rows = [r for r in rows if not furniture(r, page_height)]
    multi = [i for i, r in enumerate(rows) if len(r) >= 2]
    if not multi:
        return []
    m, end = multi[0], multi[-1]
    base = 0 if keep_top else m
    # A header row of cells centered across two header lines sits half a line below the line
    # above it (Parameter | Symbol | Unit between "Value" and "Min Typ Max"). Its cells span both
    # lines, so the line above belongs to the table and the half-line row merges into it.
    yc = [(min(c["bbox"][1] for c in r) + max(c["bbox"][3] for c in r)) / 2 for r in rows]
    pitch = sorted(b - a for a, b in zip(yc[m:end], yc[m + 1:end + 1]))
    pitch = pitch[len(pitch) // 2] if pitch else 0
    if 0 < m and m + 1 <= end and yc[m + 1] - yc[m - 1] <= 1.25 * pitch:
        merged = sorted(rows[m - 1] + rows[m], key=lambda c: c["bbox"][0])
        return rows[min(base, m - 1):m - 1] + [merged] + rows[m + 1:end + 1]
    return rows[base:end + 1]


def split_runs(rows):
    """Cut a page's free rows (outside ruled boxes) at vertical gaps > GAP_RUNS x the row pitch:
    two borderless tables, or a table and a paragraph, are separated by more than a line."""
    if not rows:
        return []
    yc = [(min(c["bbox"][1] for c in r) + max(c["bbox"][3] for c in r)) / 2 for r in rows]
    diffs = sorted(b - a for a, b in zip(yc, yc[1:]))
    pitch = diffs[len(diffs) // 2] if diffs else 0
    runs, cur = [], [rows[0]]
    for prev_y, y, r in zip(yc, yc[1:], rows[1:]):
        if pitch and y - prev_y > GAP_RUNS * pitch:
            runs.append(cur)
            cur = []
        cur.append(r)
    runs.append(cur)
    return runs


page_finder_count = {}  # id(page) -> tables PyMuPDF's line finder saw (for the page check)


def ruled_boxes(page):
    """Ruled table boxes from PyMuPDF's own table finder (already a dependency), with their
    column edges from the cell rules. A box splits where consecutive rows have incompatible rule
    sets (neither a subset of the other: two tables stacked in one grid); a merged cell only
    removes edges, so it never splits."""
    try:
        found = page.find_tables(strategy="lines").tables
    except Exception:  # malformed drawings: fall back to borderless detection
        return []
    page_finder_count[id(page)] = len(found)
    out = []
    for t in found:
        rows = []
        for r in t.rows:
            cells = [c for c in r.cells if c]
            if cells:
                rows.append((min(c[1] for c in cells), max(c[3] for c in cells),
                             frozenset(round(v) for c in cells for v in (c[0], c[2])),
                             [(round(c[0]), round(c[2])) for c in cells]))
        if not rows:
            out.append({"rect": pymupdf.Rect(t.bbox), "edges": []})
            continue
        def cuts(edges, cells):  # an edge strictly inside one of the other row's cells
            return any(x0 + 1 < e < x1 - 1 for e in edges for x0, x1 in cells)

        groups = [[rows[0]]]
        for prev, cur in zip(rows, rows[1:]):
            # merged cells cut one way only; two different grids stacked cut both ways
            if cuts(prev[2], cur[3]) and cuts(cur[2], prev[3]):
                groups.append([cur])
            else:
                groups[-1].append(cur)
        for g in groups:
            edges = sorted(set().union(*(e for _, _, e, _ in g)))
            out.append({"rect": pymupdf.Rect(t.bbox[0], g[0][0], t.bbox[2], g[-1][1]), "edges": edges})
    return out


def vertical_rules(page):
    """Drawn vertical segments (lines and rect edges) as (x, y0, y1), in display space."""
    m = page.rotation_matrix if page.rotation else None
    segs = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return segs
    for d in drawings:
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if m is not None:
                    a, b = a * m, b * m
                if abs(a.x - b.x) < 1.0:
                    segs.append((a.x, min(a.y, b.y), max(a.y, b.y)))
            elif it[0] == "re":
                r = it[1] * m if m is not None else it[1]
                segs += [(r.x0, r.y0, r.y1), (r.x1, r.y0, r.y1)]
    return segs


def real_edges(edges, segs, y0, y1, share=RULE_COVER):
    """Keep the outer edges, and the interior edges that a drawn vertical rule actually covers over
    >= share of [y0, y1]. A table finder can extend one table's grid into a borderless neighbour."""
    if len(edges) < 3 or y1 <= y0:
        return edges
    keep = [edges[0]]
    for e in edges[1:-1]:
        spans = sorted((max(a, y0), min(b, y1)) for x, a, b in segs if abs(x - e) <= 1.5 and b > y0 and a < y1)
        covered, end = 0.0, y0
        for a, b in spans:
            if b > end:
                covered += b - max(a, end)
                end = b
        if covered >= share * (y1 - y0):
            keep.append(e)
    return keep + [edges[-1]]


def page_regions(page, chars):
    """-> [(rows, bbox)] table regions of one page, plus the free rows left over (captions etc.).
    Ruled boxes first (chars inside a box belong to it, so side-by-side ruled tables stay apart);
    the remaining text is cut into runs and each run with >= 2 multi-cell rows is a borderless table.
    ponytail: side-by-side BORDERLESS tables are not split; needs a vertical-gutter rule."""
    H = page.rect.height
    taken, regions, lifted = set(), [], []
    segs = None
    for rb in sorted(ruled_boxes(page), key=lambda r: (r["rect"].y0, r["rect"].x0)):
        box = rb["rect"]
        ids = [i for i, c in enumerate(chars) if i not in taken and _center_in(c["bbox"], box, 1.0)]
        rows = table_rows(rows_of([chars[i] for i in ids]), H, keep_top=True)
        if sum(len(r) >= 2 for r in rows) >= 2:
            taken |= set(ids)
            # a unit line caught inside the box's top edge is a caption, not a row
            while rows and len(rows[0]) == 1 and UNIT.search(rows[0][0]["raw_text"]):
                lifted.append(rows.pop(0))
            if len(rb["edges"]) >= 3:
                segs = vertical_rules(page) if segs is None else segs
                y0, y1 = min(c["bbox"][1] for r in rows for c in r), max(c["bbox"][3] for r in rows for c in r)
                rb["edges"] = real_edges(rb["edges"], segs, y0, y1)
            regions.append((rows, [box.x0, min(c["bbox"][1] for r in rows for c in r), box.x1, box.y1], rb["edges"]))
    free = [r for r in rows_of([c for i, c in enumerate(chars) if i not in taken]) if not furniture(r, H)]
    # A header printed above the rules (the rule sits under the header): free multi-cell rows right
    # above a ruled box join it, nearest first, while each gap stays within ABSORB_PITCH row pitches.
    # A single-line row (caption, unit line) stops it.
    for k, (rows, box, edges) in enumerate(regions):
        yc = [_yc(r) for r in rows]
        diffs = sorted(b - a for a, b in zip(yc, yc[1:]))
        pitch = diffs[len(diffs) // 2] if diffs else 14.0
        top = yc[0]
        above = sorted((r for r in free if _yc(r) < top and any(overlap(c["bbox"][0], c["bbox"][2], box[0], box[2]) > 0 for c in r)),
                       key=_yc, reverse=True)
        got = []
        for r in above:
            if len(r) < 2 or top - _yc(r) > ABSORB_PITCH * pitch:
                break
            got.insert(0, r)
            top = _yc(r)
        if got:
            free = [r for r in free if all(r is not g for g in got)]
            box = [min(box[0], *(c["bbox"][0] for r in got for c in r)), min(c["bbox"][1] for r in got for c in r),
                   max(box[2], *(c["bbox"][2] for r in got for c in r)), box[3]]
            regions[k] = (table_rows(got + rows, H, keep_top=True), box, edges)
    used = set()
    for run in split_runs(free):
        rows = table_rows(run, H)
        if sum(len(r) >= 2 for r in rows) >= 2:
            regions.append((rows, None, []))
            used |= {id(r) for r in rows}
    leftover = [r for r in free if id(r) not in used] + lifted
    return regions, leftover


def _yc(r):
    return (min(c["bbox"][1] for c in r) + max(c["bbox"][3] for c in r)) / 2


def _center_in(bb, rect, pad=0.0):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return rect.x0 - pad <= cx <= rect.x1 + pad and rect.y0 - pad <= cy <= rect.y1 + pad


def attach_captions(table, leftover):
    """Single-line rows just above a table (within CAPTION_GAP) and overlapping it in x become its
    title and unit line; they are not table rows."""
    x0, y0, x1, _ = table["regions"][0]["bbox"]
    above = [r for r in leftover if len(r) == 1 and r[0]["bbox"][3] <= y0 + 1 and y0 - r[0]["bbox"][3] <= CAPTION_GAP
             and overlap(r[0]["bbox"][0], r[0]["bbox"][2], x0, x1) > 0]
    above.sort(key=lambda r: -r[0]["bbox"][3])  # nearest first
    for r in above:
        text = r[0]["raw_text"]
        if UNIT.search(text) and "unit" not in table:
            table["unit"] = parse_unit(text)
        elif "title" not in table:
            table["title"] = text
            table["title_bbox"] = r[0]["bbox"]
        leftover.remove(r)


def same_columns(a, b, tol=6.0):
    return len(a["_bands"]) == len(b["_bands"]) and all(abs(x - y) <= tol for x, y in zip(a["_bands"], b["_bands"]))


def header_texts(t):
    return [(c["row"], c["col"], c["raw_text"]) for c in t["cells"] if c["is_header"]]


def assign_hierarchy(t):
    """Indent levels (label x0 clusters 2 pt apart), parent_row (nearest row above with a smaller
    indent) and is_total for every body row. Run on the whole table, so a table joined across pages
    gets one hierarchy (page 2 alone would miss its top level and shift every indent)."""
    labels = {c["row"]: c for c in t["cells"] if c["col"] == 0 and not c["is_header"]}
    xs = sorted({round(c["bbox"][0]) for c in labels.values() if c["bbox"]})
    levels = []
    for x in xs:
        if not levels or x - levels[-1] > 2:
            levels.append(x)
    stack = []
    for r in sorted(t["rows"], key=lambda r: r["row"]):
        lab = labels.get(r["row"])
        indent = 0
        if lab and lab["bbox"]:
            indent = max((i for i, x in enumerate(levels) if x <= round(lab["bbox"][0]) + 2), default=0)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        r.update(indent_level=indent, parent_row=stack[-1][1] if stack else None,
                 is_total=bool(lab and TOTAL.search(lab["raw_text"].strip())))
        stack.append((indent, r["row"]))


def stitch(a, b):
    """Append table b (top of the next page) to table a (bottom of this page). b's header rows are
    a repeat and are dropped; every row keeps its own page."""
    nh, shift = b["header_rows"], a["n_rows"] - b["header_rows"]
    a["cells"] += [dict(c, row=c["row"] + shift) for c in b["cells"] if c["row"] >= nh]
    a["rows"] += [dict(r, row=r["row"] + shift, parent_row=None if r["parent_row"] is None else r["parent_row"] + shift)
                  for r in b["rows"]]
    a["regions"] += b["regions"]
    a["n_rows"] += b["n_rows"] - nh
    a["issues"] += b["issues"]
    assign_hierarchy(a)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_bands(rows):
    """Column bands from rows that hold a number. A cell joins the band it overlaps in x, but two
    cells of the same row never share a band (that would be two columns merged). Bands are left
    to right; numbers right-aligned in a column overlap at their shared right edge."""
    use = [ri for ri, r in enumerate(rows) if any(is_number(c["raw_text"]) for c in r)]
    if not use:  # a table with no plain numbers (text table, `EUR 20,000,000`): all multi-cell rows
        use = [ri for ri, r in enumerate(rows) if len(r) >= 2]
    bands = _bands(rows, use)
    # A row whose every cell sits in a band no other row uses is off-grid (a centered numeric
    # header `3 6 9 12` over right-aligned values): rebuild without it. place_header still
    # assigns its cells by nearest band.
    def solo(ri, c):  # the band holding this cell is used by no other row
        return all(b["rows"] == {ri} for b in bands if ri in b["rows"] and overlap(c["bbox"][0], c["bbox"][2], b["x0"], b["x1"]) > 0)

    off = {ri for ri in use if all(solo(ri, c) for c in rows[ri] if is_number(c["raw_text"]))}
    if off and len(use) - len(off) >= 2:
        bands = _bands(rows, [ri for ri in use if ri not in off])
    return bands


def _bands(rows, use):
    bands = []  # {x0, x1, rows:set, num}
    cells = sorted(((ri, c) for ri in use for c in rows[ri]), key=lambda rc: rc[1]["bbox"][0])
    for ri, c in cells:
        x0, x1 = c["bbox"][0], c["bbox"][2]
        hits = [b for b in bands if overlap(x0, x1, b["x0"], b["x1"]) > 0 and ri not in b["rows"]]
        num = is_number(c["raw_text"])
        if hits:
            b = max(hits, key=lambda b: overlap(x0, x1, b["x0"], b["x1"]))
            b["x0"], b["x1"] = min(b["x0"], x0), max(b["x1"], x1)
            b["rows"].add(ri)
            b["num"] = b["num"] or num
        else:
            bands.append({"x0": x0, "x1": x1, "rows": {ri}, "num": num})
    bands.sort(key=lambda b: (b["x0"] + b["x1"]) / 2)
    # Two neighbouring text-only bands that never share a row are one label column whose labels
    # start at different x (a `Total` row without the icon the other labels have).
    # ponytail: misses a genuinely sparse text column; flag-worthy once validation exists (m4).
    merged = []
    for b in bands:
        m = merged[-1] if merged else None
        if m and not m["num"] and not b["num"] and not (m["rows"] & b["rows"]):
            m["x0"], m["x1"], m["rows"] = min(m["x0"], b["x0"]), max(m["x1"], b["x1"]), m["rows"] | b["rows"]
        else:
            merged.append(b)
    return merged


def band_of(cell, bands):
    """Band with the most x-overlap, else the nearest by center."""
    x0, x1 = cell["bbox"][0], cell["bbox"][2]
    ov = [overlap(x0, x1, b["x0"], b["x1"]) for b in bands]
    if max(ov) > 0:
        return ov.index(max(ov))
    cx = (x0 + x1) / 2
    return min(range(len(bands)), key=lambda i: abs((bands[i]["x0"] + bands[i]["x1"]) / 2 - cx))


def n_header_rows(rows, bands):
    """Header = top rows with text in a value column (any band but the first) and no number in
    it; year rows (2025, 2026) count as header. design.md §1.4 changed: "rows before the first
    numeric row" would make section labels like 資産の部 headers."""
    n = 0
    for r in rows:
        vals = [c for c in r if band_of(c, bands) > 0]
        if not vals:
            break
        nums = [c for c in vals if is_number(c["raw_text"])]
        if nums and not all(YEAR.match(c["raw_text"].strip()) for c in nums):
            break
        n += 1
    return n


def place_header(hrows, bands):
    """Header cells -> grid cells with spans. Each header cell anchors to the nearest band (left to
    right, one cell per band). A band with no text of its own in that row but with sub-header text
    below joins the nearest header cell of the row (col_span). A cell whose bands are all empty in
    the next header rows extends down (row_span). Empty positions get empty cells (no span: nothing
    on the page shows one)."""
    centers = [(b["x0"] + b["x1"]) / 2 for b in bands]
    nb, nh = len(bands), len(hrows)
    own = []  # per header row: {band: cell}
    for r in hrows:
        got, last = {}, -1
        for c in sorted(r, key=lambda c: c["bbox"][0]):
            cx = (c["bbox"][0] + c["bbox"][2]) / 2
            choices = [i for i in range(last + 1, nb)] or [last]
            i = min(choices, key=lambda i: abs(centers[i] - cx))
            got[i] = c
            last = i
        own.append(got)
    span = []  # per header row: {anchor band: [bands]}
    for ri in range(nh):
        groups = {i: [i] for i in own[ri]}
        below = lambda b: any(b in own[k] for k in range(ri + 1, nh))
        # only a cell with sub-headers under it is a group header; one with nothing under it
        # (Symbol) spans rows, not columns
        anchors = [i for i in own[ri] if below(i)]
        for b in range(nb):
            if b in own[ri] or not anchors or not below(b):
                continue
            a = min(anchors, key=lambda i: abs(centers[i] - centers[b]))
            groups[a].append(b)
        span.append(groups)
    covered, out = set(), []
    for ri in range(nh):
        for a, bs in sorted(span[ri].items()):
            lo, hi = min(bs), max(bs)
            rs = 1
            while ri + rs < nh and not any(b in own[ri + rs] for b in range(lo, hi + 1)) \
                    and not any(lo <= b <= hi for g in span[ri + rs].values() for b in g):
                rs += 1
            c = own[ri][a]
            out.append(dict(c, row=ri, col=lo, row_span=rs, col_span=hi - lo + 1, is_header=True))
            covered |= {(ri + dr, b) for dr in range(rs) for b in range(lo, hi + 1)}
    for ri in range(nh):
        for b in range(nb):
            if (ri, b) not in covered:
                out.append({"raw_text": "", "bbox": None, "row": ri, "col": b, "row_span": 1, "col_span": 1,
                            "is_header": True})
    return out


def join_wrapped(rows, bands):
    """A label-only row whose label nearly fills the label column, followed by a row whose label
    starts at the same x, is one wrapped label (numbers sit on the last line). CJK text joins with
    no space, other text with one."""
    # fill is measured against the widest label actually printed in the label column: a ruled
    # column's edges are wider than its text
    labels = [c for r in rows for c in r if band_of(c, bands) == 0] if bands else []
    right = max((c["bbox"][2] for c in labels), default=0)
    width = right - bands[0]["x0"] if bands else 0
    out = []
    for r in rows:
        prev = out[-1] if out else None
        if (prev is not None and len(prev) == 1 and band_of(prev[0], bands) == 0 and r
                and band_of(r[0], bands) == 0 and abs(r[0]["bbox"][0] - prev[0]["bbox"][0]) < 1.5
                and prev[0]["bbox"][2] - bands[0]["x0"] >= WRAP_FILL * width and prev[0].get("wrap_ok", True)):
            a, b = prev[0], r[0]
            sep = "" if _cjk(a["raw_text"][-1]) and _cjk(b["raw_text"][0]) else " "
            merged = {"raw_text": a["raw_text"] + sep + b["raw_text"],
                      "bbox": [min(a["bbox"][0], b["bbox"][0]), a["bbox"][1], max(a["bbox"][2], b["bbox"][2]), b["bbox"][3]]}
            out[-1] = [merged] + r[1:]
        else:
            out.append(list(r))
    return out


def _cjk(ch):
    return unicodedata.east_asian_width(ch) in ("W", "F")


def ruled_bands(rows, edges):
    """Bands from the cell rules of a ruled box: exact column edges, so a column with no value in
    any row (`Notes`) still exists. Used only if every value row puts at most one cell per ruled
    column; otherwise the rules do not describe the cells (None -> geometry bands)."""
    if len(edges) < 3:  # fewer than 2 columns: rules are only top/bottom lines or row stripes
        return None
    bands = [{"x0": a, "x1": b, "rows": set(), "num": False} for a, b in zip(edges, edges[1:]) if b - a > 2]
    for ri, r in enumerate(rows):
        if not any(is_number(c["raw_text"]) for c in r):
            continue
        seen = set()
        for c in r:
            b = band_of(c, bands)
            if b in seen:
                return None
            seen.add(b)
            bands[b]["rows"].add(ri)
            bands[b]["num"] = bands[b]["num"] or is_number(c["raw_text"])
    # a ruled column holding no cell of this region at all (header or body) belongs to another
    # table sharing the grid; an empty-valued column with a header (`Notes`) is kept
    used = {band_of(c, bands) for r in rows for c in r}
    bands = [b for i, b in enumerate(bands) if i in used]
    return bands if len(bands) >= 2 else None


def normalize_cells(cells, pno):
    for c in cells:
        c["page"] = pno
        c.setdefault("flags", [])
        n = normalize(c["raw_text"])
        n.setdefault("notes", [])
        if c["is_header"]:  # headers are labels, even `2025`
            n.update(value=None, value_type="text", footnote_refs=[], flags=[], notes=[])
        c["flags"] = c["flags"] + n.pop("flags")
        c.update(n)


def build_table(rows, pno, table_id, box=None, edges=()):
    bands = ruled_bands(rows, list(edges)) or build_bands(rows)
    if not bands:
        return None
    nh = n_header_rows(rows, bands)
    cells = place_header(rows[:nh], bands)
    body = join_wrapped(rows[nh:], bands)
    flags = []
    x0s = sorted({round(c["bbox"][0]) for r in body for c in r if band_of(c, bands) == 0})
    levels = []
    for x in x0s:  # indent levels: label x0 clusters 2pt apart
        if not levels or x - levels[-1] > 2:
            levels.append(x)
    meta, stack = [], []
    for k, r in enumerate(body):
        ri = nh + k
        slots = {}
        for c in r:
            b = band_of(c, bands)
            if b in slots:  # two cells in one column: never merge silently
                flags.append({"check": "column_conflict", "row": ri, "col": b})
                slots[b]["raw_text"] += " " + c["raw_text"]
                slots[b]["flags"] = ["column_conflict"]
            else:
                slots[b] = dict(c, flags=list(c.get("flags", [])))
        for b in range(len(bands)):
            c = slots.get(b, {"raw_text": "", "bbox": None, "flags": []})
            cells.append(dict(c, row=ri, col=b, row_span=1, col_span=1, is_header=False))
        label = slots.get(0)
        indent = 0
        if label and label["bbox"]:
            indent = max((i for i, x in enumerate(levels) if x <= round(label["bbox"][0]) + 2), default=0)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        meta.append({"row": ri, "page": pno, "indent_level": indent, "parent_row": stack[-1][1] if stack else None,
                     "is_total": bool(label and TOTAL.search(label["raw_text"].strip()))})
        stack.append((indent, ri))
    normalize_cells(cells, pno)
    boxes = [c["bbox"] for c in cells if c["bbox"]]
    bb = box or [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    columns = [{"col": b, "header_path": [c["raw_text"] for c in sorted(cells, key=lambda c: c["row"])
                                          if c["is_header"] and c["raw_text"] and c["col"] <= b < c["col"] + c["col_span"]]}
               for b in range(len(bands))]
    return {"table_id": table_id, "regions": [{"page": pno, "bbox": [round(v, 2) for v in bb]}],
            "n_rows": nh + len(body), "n_cols": len(bands), "header_rows": nh, "columns": columns,
            "rows": meta, "cells": cells, "needs_review": False, "issues": flags,
            "_bands": [(b["x0"] + b["x1"]) / 2 for b in bands]}


# ---------------------------------------------------------------- m4: validation (design.md §4-5)
NET = re.compile(r"(純額|\bnet\b)", re.I)
TABLE_HARD = {"unassigned_text", "stitch_unconfirmed", "unit_unknown", "header_inside_body", "borderless_unverified",
              "cell_flags_escalated", "ocr_numbers_missing", "ocr_numbers_unmatched"}  # -> needs_review
# Page-level checks, a setting: "stray" = numbers in multi-cell rows outside every table;
# "count_lines" = PyMuPDF's line finder sees a different number of tables than we output;
# "mid_header" = header-like row inside a table body; "count_text" = PyMuPDF's text finder disagrees too.
# Default S2 = stray,count_lines,mid_header: lowest unflagged with review < 50% on ParseBench dev
# (dev: real unflagged 20.0% -> 15.6%, review 28.7% -> 48.1%). "borderless" / "escalate" are stricter
# options (escalate: review 60% on dev).
PAGE_CHECKS = set(filter(None, os.environ.get("PARSER_PAGE_CHECKS", "stray,count_lines,mid_header").split(",")))


def _flag(cell, reason):
    if reason not in cell["flags"]:
        cell["flags"].append(reason)


def _issue(t, check, **detail):
    t["issues"].append({"check": check, **detail})
    if check in TABLE_HARD:
        t["needs_review"] = True


def check_totals(t):
    """Total rows (合計 / Total) against the sum of their sibling rows, per numeric column, using
    the indent tree. A section row without a value takes its own total row's value; a gross row
    with a net child (（純額） / net) contributes the net value; a net row is itself checked as
    gross + the other children of the gross row. Tolerance = (#addends) x 1 unit of the last printed
    digit (each line is rounded on its own, design.md §0). A mismatch flags the total and every addend."""
    grid = {(c["row"], c["col"]): c for c in t["cells"] if not c["is_header"]}
    meta = {r["row"]: r for r in t["rows"]}
    kids = {}
    for r in t["rows"]:
        kids.setdefault(r["parent_row"], []).append(r["row"])
    label = lambda ri: grid.get((ri, 0), {}).get("raw_text", "")

    def val(ri, col, depth=0):
        c = grid.get((ri, col))
        if c and c.get("value") is not None and c["value_type"] == "number":
            return Decimal(c["value"]), [c]
        if c and c["raw_text"].strip() and c.get("blank_kind") is None:
            return "unknown", [c]  # printed but not a plain number: cannot check
        totals = [k for k in kids.get(ri, []) if meta[k]["is_total"]]
        if totals and depth < 4:
            return val(totals[-1], col, depth + 1)
        return None, []

    def addend(ri, col):
        net = [k for k in kids.get(ri, []) if NET.search(label(k))]
        return val(net[-1], col) if net else val(ri, col)

    checks = []  # (row of the checked cell, [addend rows], uses addend() for the addends)
    for r in t["rows"]:
        if r["is_total"]:
            sib = [k for k in kids.get(r["parent_row"], []) if k != r["row"] and not meta[k]["is_total"]]
            checks.append((r["row"], sib, True))
        if NET.search(label(r["row"])) and r["parent_row"] is not None:
            others = [k for k in kids.get(r["parent_row"], []) if k != r["row"]]
            checks.append((r["row"], [r["parent_row"]] + others, False))
    n_ok = n_bad = 0
    for col in range(1, t["n_cols"]):
        for ri, adds, use_net in checks:
            total, tcells = val(ri, col)
            if total is None or total == "unknown":
                continue
            parts = [addend(k, col) if use_net else val(k, col) for k in adds]
            if any(v == "unknown" for v, _ in parts):
                continue
            parts = [(v, cs) for v, cs in parts if v is not None]
            if not parts:
                continue
            dec = max(-min(v.as_tuple().exponent, 0) for v in [total] + [v for v, _ in parts])
            tol = len(parts) * Decimal(1).scaleb(-dec)
            if abs(sum(v for v, _ in parts) - total) <= tol:
                n_ok += 1
            else:
                n_bad += 1
                for c in tcells + [c for _, cs in parts for c in cs]:
                    _flag(c, "sum_mismatch")
                t["issues"].append({"check": "sum_mismatch", "row": ri, "col": col,
                                    "expected": str(sum(v for v, _ in parts)), "found": str(total)})
    return n_ok, n_bad


def check_column_types(t, min_cells=5, share=0.8):
    """A text cell in a mostly-numeric column, or a number in a mostly-text column, is a likely
    shift or merge: flag the cell."""
    for col in range(t["n_cols"]):
        body = [c for c in t["cells"] if c["col"] == col and not c["is_header"] and c["value_type"] != "blank"]
        if len(body) < min_cells:
            continue
        nums = [c for c in body if c["value_type"] in ("number", "percent")]
        if len(nums) >= share * len(body):
            for c in body:
                if c not in nums:
                    _flag(c, "type_outlier")
        elif len(nums) <= (1 - share) * len(body):
            for c in nums:
                _flag(c, "type_outlier")


def check_conservation(tables, chars):
    """Every character inside a table's area must sit in exactly one cell. Characters left over
    (a row dropped, an overlapping region) mark the table for review."""
    boxes = [c["bbox"] for t in tables for c in t["cells"] if c["bbox"]]
    for t in tables:
        own = [c["bbox"] for c in t["cells"] if c["bbox"]]
        if not own:
            continue
        x0, y0 = min(b[0] for b in own) + 0.5, min(b[1] for b in own) + 0.5
        x1, y1 = max(b[2] for b in own) - 0.5, max(b[3] for b in own) - 0.5
        lost = []
        for ch in chars:
            if not ch["c"].strip():
                continue
            cx, cy = (ch["bbox"][0] + ch["bbox"][2]) / 2, (ch["bbox"][1] + ch["bbox"][3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1 and not any(
                    b[0] - 0.6 <= cx <= b[2] + 0.6 and b[1] - 0.6 <= cy <= b[3] + 0.6 for b in boxes):
                lost.append(ch["c"])
        if lost:
            _issue(t, "unassigned_text", count=len(lost), sample="".join(lost)[:40])


def check_mid_header(t):
    """A row inside the body that looks like a header (text, no numbers, in >= 2 value columns) with
    numbers below it: likely two stacked tables merged into one."""
    body = sorted({c["row"] for c in t["cells"] if not c["is_header"]})
    by = {r: [c for c in t["cells"] if c["row"] == r and c["col"] > 0 and c["raw_text"].strip()] for r in body}
    has_num = {r: any(c["value"] is not None for c in by[r]) for r in body}
    seen_num = False
    for k, r in enumerate(body):
        if has_num[r]:
            seen_num = True
            continue
        texts = [c for c in by[r] if c["value_type"] == "text"]
        if seen_num and len(texts) >= 2 and any(has_num[q] for q in body[k + 1:]):
            _issue(t, "header_inside_body", row=r)
            return


def validate(t):
    """Cell- and table-level checks; then confidence: 1.0 verified, 0.0 flagged (no in-between)."""
    for c in t["cells"]:
        if any(unicodedata.category(ch) in ("Co", "Cn") or ch == "\ufffd" for ch in c["raw_text"]):
            _flag(c, "bad_glyph")
    if any("column_conflict" in c["flags"] for c in t["cells"]):
        t["issues"] = [i for i in t["issues"] if i["check"] != "column_conflict"] + [{"check": "column_conflict"}]
    check_column_types(t)
    if "mid_header" in PAGE_CHECKS:
        check_mid_header(t)
    ok, bad = check_totals(t)
    if "borderless" in PAGE_CHECKS and not t.get("_ruled"):
        _issue(t, "borderless_unverified")
    if "escalate" in PAGE_CHECKS and any(c["flags"] for c in t["cells"]):
        _issue(t, "cell_flags_escalated")
    t["issues"].append({"check": "totals", "passed": ok, "failed": bad})
    if t.get("unit") and t["unit"]["scale"] is None:
        _issue(t, "unit_unknown", raw_text=t["unit"]["raw_text"])
    for c in t["cells"]:
        c["confidence"] = 0.0 if (c["flags"] or t["needs_review"]) else 1.0
    t["confidence"] = min((c["confidence"] for c in t["cells"] if c.get("value") is not None), default=1.0)


# ---------------------------------------------------------------- m6: text-layer audit + OCR fallback
BAD_GLYPH_MAX = 0.01  # > 1% U+FFFD / private-use / control chars: look closer (icon font or broken map?)
GARBLED_CELLS = 0.2  # broken map: bad glyphs mixed into words in > 20% of cells
SPOT_MIN_CELLS, SPOT_FAIL = 10, 0.10  # Vision spot check: flags the page, never replaces the text layer
SPOT_CHECK = os.environ.get("PARSER_SPOT_CHECK", "1") == "1"


def audit(page, chars):
    """Hard reasons to distrust a page's text layer (each sends the page to OCR)."""
    glyphs = [c["c"] for c in chars if c["c"].strip()]
    if len(glyphs) < 20:
        return ["no_text_layer"]
    reasons = []
    is_bad = lambda ch: ch == "\ufffd" or unicodedata.category(ch) in ("Co", "Cn", "Cc")
    bad = sum(any(is_bad(ch) for ch in g) for g in glyphs)
    if bad / len(glyphs) > BAD_GLYPH_MAX:
        # Icon fonts (SF Symbols, Wingdings checkmarks) put private-use glyphs in cells of their own;
        # a broken ToUnicode map garbles the words themselves. Only the second is a bad text layer.
        cells = [c["raw_text"] for r in rows_of(chars) for c in r]
        mixed = sum(any(is_bad(ch) for ch in t) and any(ch.isalnum() for ch in t) for t in cells)
        if mixed > GARBLED_CELLS * max(len(cells), 1) or bad / len(glyphs) > 0.5:
            reasons.append("garbled_text_layer")
    try:
        trace = page.get_texttrace()
        n = sum(len(t["chars"]) for t in trace)
        if n and sum(len(t["chars"]) for t in trace if t["type"] == 3) / n > 0.5:
            reasons.append("invisible_text_layer")  # an old OCR layer: re-OCR it
    except Exception:
        pass
    return reasons


def image_regions(page, chars):
    """Images with no text over them (text that exists only as a picture): OCR region candidates."""
    out = []
    area = page.rect.width * page.rect.height
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            r = r & page.rect
            if r.is_empty or r.width * r.height < 0.03 * area:
                continue
            if sum(1 for c in chars if _center_in(c["bbox"], r)) < 5:
                out.append(r)
    return out


def page_text(pdf_path, page, progress=None):
    """-> (chars, info, flags) for one page: text layer if it passes the audit, else OCR (setting
    PARSER_OCR), plus OCR of image regions that hold numbers."""
    import ocr
    chars = page_chars(page)
    reasons = audit(page, chars)
    info, flags = {"reasons": reasons}, []
    if reasons:
        if ocr.ENGINE == "none":
            return [], dict(info, text_source="none"), ["no_text_layer"]
        if progress:
            progress("ocr_start", page.number + 1, reasons)
        units, rep = ocr.ocr_region(pdf_path, page)
        if progress:
            progress("ocr_done", page.number + 1, rep)
        info.update(text_source="ocr", ocr=rep)
        if rep["page_unverified"]:
            flags.append("ocr_unverified")  # LightOnOCR empty or cut: nothing can be confirmed
        return units, info, flags
    info["text_source"] = "text_layer"
    if ocr.ENGINE != "none":
        for r in image_regions(page, chars):
            words = ocr._cached(ocr._key(pdf_path, page.number + 1, "tesseract", r), lambda: ocr.tesseract_words(page, r))
            if sum(1 for w in words if is_number(w["text"])) >= 5:
                if progress:
                    progress("ocr_start", page.number + 1, ["image region"])
                units, rep = ocr.ocr_region(pdf_path, page, r)
                if progress:
                    progress("ocr_done", page.number + 1, rep)
                chars = chars + units
                info.setdefault("ocr_regions", []).append({"bbox": list(r), **rep})
                info["text_source"] = "text_layer+ocr"
                if rep["page_unverified"]:
                    flags.append("ocr_unverified")
    if SPOT_CHECK:
        cells = [c for row in rows_of([c for c in chars if not c.get("unit")]) for c in row if is_number(c["raw_text"])]
        if len(cells) >= SPOT_MIN_CELLS:
            rate, seen = ocr.spot_check(page, cells)
            info["spot_check"] = {"agreement": rate, "cells": seen}
            if rate is not None and seen >= SPOT_MIN_CELLS and rate < SPOT_FAIL:
                flags.append("spot_check_disagrees")
    return chars, info, flags


OCR_STRUCTURE = os.environ.get("PARSER_OCR_STRUCTURE", "lighton")  # lighton | tesseract (see design.md §1.2)


def _squash(t):
    return unicodedata.normalize("NFKC", t).replace(" ", "")


def _label_boxes(cells, words, nh):
    """Label-column boxes by geometry, not spelling (OCR typos break spelling): a body row's
    vertical range comes from its other boxed cells; its label is the Tesseract words in that range
    left of the value columns. Rows without boxed cells (section headings) take the left-zone words
    in the gap between their neighbours, one line per row."""
    vals = [c["bbox"] for c in cells if c["col"] > 0 and c["bbox"] and not c["is_header"]]
    if not vals:
        return
    left_of = min(b[0] for b in vals) - 2
    zone = sorted((w for w in words if (w["bbox"][0] + w["bbox"][2]) / 2 < left_of), key=lambda w: (w["bbox"][1], w["bbox"][0]))
    rows = sorted({c["row"] for c in cells if c["row"] >= nh})
    yr = {}
    for r in rows:
        bs = [c["bbox"] for c in cells if c["row"] == r and c["col"] > 0 and c["bbox"]]
        if bs:
            yr[r] = (min(b[1] for b in bs) - 2, max(b[3] for b in bs) + 2)
    lines = []  # left-zone words grouped into text lines
    for w in zone:
        yc = (w["bbox"][1] + w["bbox"][3]) / 2
        if lines and abs(yc - lines[-1][0]) < 0.5 * (w["bbox"][3] - w["bbox"][1]):
            lines[-1][1].append(w)
        else:
            lines.append([yc, [w]])
    taken = set()
    by_row = {}
    for r, (y0, y1) in yr.items():
        hit = [k for k, (yc, _) in enumerate(lines) if y0 <= yc <= y1 and k not in taken]
        if hit:
            by_row[r] = hit[-1]  # numbers sit on a wrapped label's last line
            taken |= set(hit)
    for k, r in enumerate(rows):  # rows with no boxed cell: next free line between the neighbours
        if r in by_row:
            continue
        lo = max((yr[q][1] for q in rows[:k] if q in yr), default=-1e9)
        hi = min((yr[q][0] for q in rows[k + 1:] if q in yr), default=1e9)
        free = [j for j, (yc, _) in enumerate(lines) if lo <= yc <= hi and j not in taken]
        if free:
            by_row[r] = free[0]
            taken.add(free[0])
    for c in cells:
        if c["col"] == 0 and c["row"] in by_row and c["raw_text"].strip():
            ws = lines[by_row[c["row"]]][1]
            c["bbox"] = [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws),
                         max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]


def lighton_tables(page_data, pno):
    """OCR page, structure from LightOnOCR's HTML tables, checked by Tesseract. A number is accepted
    only if Tesseract read the same number in the same block, in the same order; its cell box comes
    from that Tesseract word. Anything unconfirmed gets `ocr_unconfirmed`. Text cells get boxes from
    Tesseract words that spell them, which gives indent levels and the conservation check."""
    import ocr
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from parsebench_to_gold import TableHTML, to_table  # the HTML table reader already used for ParseBench GT
    words, lo = page_data
    tables, used = [], set()
    width = max((w["bbox"][2] for w in words), default=600.0)
    for b in lo["blocks"]:
        if not b["label"].startswith("table"):
            continue
        h = TableHTML()
        h.feed(b["text"])
        if not h.tables or not h.tables[0]:
            continue
        g = to_table(h.tables[0], "")
        pad = 0.01 * width
        mine = sorted((wi for wi, w in enumerate(words) if wi not in used and ocr._center_in(w["bbox"], b["bbox"], pad)),
                      key=lambda wi: (round(words[wi]["bbox"][1] / 4), words[wi]["bbox"][0]))
        used |= set(mine)
        cells = sorted(g["cells"], key=lambda c: (c["row"], c["col"]))
        # numbers: LightOn runs per cell vs Tesseract runs in the block, aligned in order
        theirs = [(ci, r) for ci, c in enumerate(cells) for _, _, r in ocr.nums(unicodedata.normalize("NFKC", c["raw_text"]))]
        ours = [(wi, r) for wi in mine for _, _, r in ocr.nums(unicodedata.normalize("NFKC", words[wi]["text"]))]
        ok_runs, boxes, extra = set(), {}, 0
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, [r for _, r in ours], [r for _, r in theirs], autojunk=False).get_opcodes():
            if tag in ("delete", "replace"):
                extra += i2 - i1  # Tesseract read numbers here that LightOn's table does not hold
            if tag == "equal":
                for (wi, r), (ci, _) in zip(ours[i1:i2], theirs[j1:j2]):
                    if not ocr.SIGN_UNCERTAIN.match(r):
                        ok_runs.add((ci, r))
                    boxes.setdefault(ci, []).append(words[wi]["bbox"])
        # text cells: Tesseract words (in block order) whose joined text spells the cell
        free = [wi for wi in mine]
        for ci, c in enumerate(cells):
            want = _squash(c["raw_text"])
            if not want or ci in boxes:
                continue
            for start in range(len(free)):
                acc, got = "", []
                for wi in free[start:start + 12]:
                    acc += _squash(words[wi]["text"])
                    got.append(wi)
                    if not want.startswith(acc):
                        break
                    if acc == want:
                        boxes[ci] = [words[x]["bbox"] for x in got]
                        free = [x for x in free if x not in got]
                        break
                if ci in boxes:
                    break
        out_cells = []
        for ci, c in enumerate(cells):
            bb = boxes.get(ci)
            runs = [r for _, _, r in ocr.nums(unicodedata.normalize("NFKC", c["raw_text"]))]
            flags = ["ocr_unconfirmed"] if any((ci, r) not in ok_runs for r in runs) else []
            out_cells.append(dict(c, bbox=[min(x[0] for x in bb), min(x[1] for x in bb), max(x[2] for x in bb), max(x[3] for x in bb)] if bb else None,
                                  flags=flags, source="ocr"))
        _label_boxes(out_cells, [words[wi] for wi in mine], g["header_rows"])
        normalize_cells(out_cells, pno)
        nh = g["header_rows"]
        body_rows = sorted({c["row"] for c in out_cells if c["row"] >= nh})
        t = {"table_id": "", "regions": [{"page": pno, "bbox": [round(v, 2) for v in b["bbox"]]}],
             "n_rows": g["n_rows"], "n_cols": g["n_cols"], "header_rows": nh,
             "columns": [{"col": k, "header_path": [c["raw_text"] for c in out_cells if c["is_header"] and c["raw_text"]
                                                    and c["col"] <= k < c["col"] + c["col_span"]]} for k in range(g["n_cols"])],
             "rows": [{"row": r, "page": pno} for r in body_rows], "cells": out_cells, "needs_review": False, "issues": [],
             "structure_source": "lighton"}
        if extra:  # conservation for OCR tables: LightOn may have dropped a row or value
            _issue(t, "ocr_numbers_unmatched", count=extra)
        assign_hierarchy(t)
        cols = {}
        for c in out_cells:
            if c["bbox"] and c["col_span"] == 1:
                cols.setdefault(c["col"], []).append((c["bbox"][0] + c["bbox"][2]) / 2)
        t["_bands"] = [sorted(cols[k])[len(cols[k]) // 2] if k in cols else -1.0 for k in range(g["n_cols"])]
        # captions: LightOn title / text blocks just above the table
        for cb in sorted((x for x in lo["blocks"] if not x["label"].startswith("table") and x["bbox"][3] <= b["bbox"][1] + 2
                          and b["bbox"][1] - x["bbox"][3] <= CAPTION_GAP), key=lambda x: -x["bbox"][3]):
            text = re.sub(r"^#+\s*", "", cb["text"]).strip()
            if UNIT.search(text) and "unit" not in t:
                t["unit"] = parse_unit(text)
            elif "title" not in t and text:
                t["title"] = text
        tables += split_repeated(t)
    return tables




def split_repeated(t):
    """A table whose last header row is k exact repeats of one column group is k side-by-side
    tables read as one (`Code | Rate | Min.` x 3, SERFF page275): split it by column group. Done
    after building, since the number check aligns Tesseract's row-by-row reading with the whole table."""
    nh, n = t["header_rows"], t["n_cols"]
    head = [c["raw_text"] for c in sorted((c for c in t["cells"] if c["row"] == nh - 1), key=lambda c: c["col"])] if nh else []
    for k in range(min(4, n // 2), 1, -1):
        w = n // k
        if n % k or len(head) != n or not any(head[:w]) or any(head[i * w:(i + 1) * w] != head[:w] for i in range(k)):
            continue
        if any(c["col"] // w != (c["col"] + c["col_span"] - 1) // w for c in t["cells"]):
            return [t]  # a span crosses the groups: not side-by-side copies
        parts = []
        for i in range(k):
            cells = [dict(c, col=c["col"] - i * w) for c in t["cells"] if i * w <= c["col"] < (i + 1) * w]
            rows = sorted({c["row"] for c in cells if c["row"] < nh or c["raw_text"].strip()})
            new = {r: j for j, r in enumerate(rows)}
            cells = [dict(c, row=new[c["row"]]) for c in cells if c["row"] in new]
            boxes = [c["bbox"] for c in cells if c["bbox"]]
            part = dict(t, cells=cells, n_cols=w, n_rows=len(rows), columns=t["columns"][:w],
                        rows=[{"row": new[r], "page": t["rows"][0]["page"] if t["rows"] else 1} for r in rows if r >= nh],
                        issues=list(t["issues"]) + [{"check": "split_side_by_side", "part": i + 1, "of": k}],
                        _bands=t["_bands"][i * w:(i + 1) * w])
            if boxes:
                part["regions"] = [{"page": t["regions"][0]["page"], "bbox": [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                                                              max(b[2] for b in boxes), max(b[3] for b in boxes)]}]
            assign_hierarchy(part)
            parts.append(part)
        return parts
    return [t]


def parse(pdf_path, doc_id=None, progress=None):
    """progress(event, page, detail): optional callback; events page_start, ocr_start, ocr_done."""
    doc = pymupdf.open(pdf_path)
    out = {"doc_id": doc_id or Path(pdf_path).stem, "source_file": str(pdf_path), "pages": [], "tables": []}
    per_page = []
    for pno, page in enumerate(doc, 1):
        if progress:
            progress("page_start", pno, len(doc))
        chars, info, pflags = page_text(pdf_path, page, progress)
        out["pages"].append({"page": pno, "width": page.rect.width, "height": page.rect.height,
                             "rotation": page.rotation, "text_source": info["text_source"],
                             "text_layer_audit": info, "flags": pflags})
        lo_struct = (info["text_source"] == "ocr" and OCR_STRUCTURE == "lighton" and info.get("ocr", {}).get("lighton")
                     and not info["ocr"]["page_unverified"])
        if lo_struct:
            import ocr
            data = ocr.page_data(pdf_path, page)
            # structure is LightOn's own, so a number Tesseract missed is already `ocr_unconfirmed` on its cell
            tables = lighton_tables(data, pno)
            tables.sort(key=lambda t: (t["regions"][0]["bbox"][1], t["regions"][0]["bbox"][0]))
            per_page.append(tables)  # conservation for these is ocr_numbers_unmatched (word boxes vs LightOn text)
            continue
        regions, leftover = page_regions(page, chars) if chars else ([], [])
        tables = []
        for rows, box, edges in regions:
            t = build_table(rows, pno, "", box, edges)
            if t:
                t["_ruled"] = box is not None
                attach_captions(t, leftover)
                tables.append(t)
        tables.sort(key=lambda t: (t["regions"][0]["bbox"][1], t["regions"][0]["bbox"][0]))
        check_conservation(tables, chars)
        flags = out["pages"][-1]["flags"]
        # numbers printed in a multi-cell row that no table took: a table may have been missed.
        # Single-line text (paragraphs, captions, footnotes) never counts.
        stray = [r for r in leftover if len(r) >= 2 and any(is_number(c["raw_text"]) for c in r)]
        if stray and "stray" in PAGE_CHECKS:
            flags.append("numbers_outside_tables")
        ocr_page = info["text_source"] == "ocr"
        # the line finder sees only drawn rules; an image page has none, so its count means nothing there
        if chars and not ocr_page and "count_lines" in PAGE_CHECKS and page_finder_count.get(id(page), 0) != len(tables):
            flags.append("table_count_disagrees")
        reps = ([info["ocr"]] if ocr_page else []) + info.get("ocr_regions", [])
        for t in tables:
            bb = t["regions"][0]["bbox"]
            for rep in reps:
                if any(b[0] < bb[2] and bb[0] < b[2] and b[1] < bb[3] and bb[1] < b[3] for b in rep["missing_blocks"]):
                    _issue(t, "ocr_numbers_missing")  # LightOnOCR read numbers here that Tesseract did not
                    break
            for c in t["cells"]:
                c["source"] = "ocr" if ocr_page else "text_layer"
        if chars and "count_text" in PAGE_CHECKS and "table_count_disagrees" in flags:
            try:
                n_text = len(page.find_tables(strategy="text").tables)
            except Exception:
                n_text = len(tables)
            if n_text == len(tables):  # the text finder sides with us: only the line finder disagrees
                flags.remove("table_count_disagrees")
        per_page.append(tables)
    # Cross-page: the last table of page p continues as the first table of page p+1 when the
    # columns line up and the header repeats or a continuation marker is printed (design.md §1.5).
    for p in range(len(per_page) - 1, 0, -1):
        prev, nxt = per_page[p - 1], per_page[p]
        if prev and nxt and same_columns(prev[-1], nxt[0]):
            a, b = prev[-1], nxt[0]
            repeated = b["header_rows"] and [x[1:] for x in header_texts(a)] == [x[1:] for x in header_texts(b)]
            marker = CONTINUED.search(b.get("title", ""))
            if repeated or marker:
                if not repeated:  # joined on the marker alone: no repeated header confirms the columns
                    _issue(a, "stitch_unconfirmed", pages=[p, p + 1])
                stitch(a, b)
                nxt.pop(0)
    for tables in per_page:
        for t in tables:
            t.pop("_bands")
            validate(t)
            t.pop("_ruled", None)
            t["table_id"] = f"t{len(out['tables']) + 1}"
            out["tables"].append(t)
    return out


def _selftest():
    sample = Path(__file__).resolve().parent.parent / "samples" / "balance_sheet_after_correction.pdf"
    t = parse(sample)["tables"][0]
    at = {(c["row"], c["col"]): c["raw_text"] for c in t["cells"]}
    assert (t["n_rows"], t["n_cols"], t["header_rows"]) == (45, 3, 2), (t["n_rows"], t["n_cols"], t["header_rows"])
    assert at[(0, 0)] == "" and at[(0, 1)] == "前連結会計年度" and at[(1, 2)] == "(2026 年３月31 日)"
    assert at[(13, 1)] == "△2,979" and at[(13, 2)] == "△0"  # split spans joined, own column
    assert at[(2, 0)] == "資産の部" and at[(2, 1)] == ""  # section row: label only, blanks filled
    rows = {r["row"]: r for r in t["rows"]}
    assert rows[4]["indent_level"] == 2 and rows[4]["parent_row"] == 3
    assert t["title"] == "（１）連結貸借対照表" and t["unit"]["raw_text"] == "（単位：千円）"
    assert t["unit"] == {"raw_text": "（単位：千円）", "scale": 1000, "currency": "JPY"}
    val = {(c["row"], c["col"]): (c["value"], c["blank_kind"], c["flags"]) for c in t["cells"]}
    assert val[(13, 1)] == ("-2979", None, []) and val[(13, 2)] == ("0", None, [])
    assert next(c for c in t["cells"] if (c["row"], c["col"]) == (13, 2))["notes"] == ["negative_zero"]
    assert val[(4, 1)] == ("2002429", None, []) and val[(0, 1)] == (None, None, [])
    cases = {"(1,234)": "-1234", "▲50": "-50", "－": None, "1,234※2": "1234", "１２.５％": "12.5", "-40.0": "-40.0",
             "¥1,000": "1000", "1,23": None, "2,002,429": "2002429", "(500 sheets x 5 reams)": None}
    for raw, want in cases.items():
        assert normalize(raw)["value"] == want, (raw, normalize(raw))
    assert normalize("1,23")["flags"] == ["ambiguous_number"] and normalize("－")["blank_kind"] == "dash"
    assert normalize("1,234※2")["footnote_refs"] == ["※2"] and normalize("１２.５％")["value_type"] == "percent"
    assert parse_unit("(In thousands of U.S. dollars)") == {"raw_text": "(In thousands of U.S. dollars)", "scale": 1000, "currency": "USD"}
    totals = next(i for i in t["issues"] if i["check"] == "totals")
    assert totals["failed"] == 0 and totals["passed"] >= 10, totals  # 流動資産合計 .. 資産合計, 純額 rows
    assert not t["needs_review"] and not any(c["flags"] for c in t["cells"] if c["value"] is not None)
    bad = json.loads(json.dumps(t))
    for c in bad["cells"]:
        if c["raw_text"] == "2,002,429":
            c["value"] = "2012429"  # a wrong digit in the 10,000s must break 流動資産合計. A last-digit
            # error (2002428) passes: it is inside the rounding tolerance, which is why OCR digits need
            # two agreeing reads (m6) rather than sums
    for c in bad["cells"]:
        c["flags"] = [f for f in c["flags"] if f != "sum_mismatch"]
    bad["issues"] = []
    assert check_totals(bad)[1] >= 1 and any("sum_mismatch" in c["flags"] for c in bad["cells"] if c["value"] == "2012429")
    gold = Path(__file__).resolve().parent.parent / "gold" / "pdf"
    two = parse(gold / "syn_en_two_tables_one_page.pdf")["tables"]
    assert [(x["n_rows"], x["n_cols"], x.get("title")) for x in two] == [(7, 4, "Invoice details"), (8, 6, "Electrical characteristics")], \
        [(x["n_rows"], x["n_cols"], x.get("title")) for x in two]
    cross = parse(gold / "syn_ja_bs_cross_page.pdf")["tables"]
    assert len(cross) == 1 and [r["page"] for r in cross[0]["regions"]] == [1, 2], [(x["n_rows"], x["regions"]) for x in cross]
    print("self-check ok")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _selftest()
        sys.exit(0)
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in [a for a in sys.argv[1:] if a.endswith(".pdf")]:
        res = parse(f)
        dst = out_dir / f"{res['doc_id']}.json"
        dst.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(dst)
