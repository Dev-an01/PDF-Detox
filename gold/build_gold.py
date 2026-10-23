"""Build the golden set: synthetic PDFs with exact ground truth, plus gold for the real sample.

    python gold/build_gold.py        -> gold/pdf/*.pdf, gold/json/*.json  (deterministic, seed 7)

Synthetic ground truth comes from the generator itself (the true numbers and the exact
positions it drew), never from parsing the PDF, so it is independent of the parser.
The real sample's gold is read from its text layer, which was checked against the
rendered page (design.md §0); the subtotal sums are re-checked below.
Gold JSON follows design.md §2. Cell bbox = tight text box (cell box when the cell is empty).
"""
import json
import random
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
PDF_DIR, JSON_DIR = ROOT / "pdf", ROOT / "json"
SAMPLE = ROOT.parent / "samples" / "balance_sheet_after_correction.pdf"

PAGE_W, PAGE_H, LEFT, RIGHT, BOTTOM = 595.0, 842.0, 50.0, 545.0, 780.0
FS, LINE, INDENT = 8.5, 14.0, 12.0
STRIPE, RULE = (0.82, 0.92, 0.98), (0, 0, 0)

CJK = pymupdf.Font("cjk")
HELV = pymupdf.Font("helv")

# ---------------------------------------------------------------- number formats
def fmt_ja(n):
    return ("△" if n < 0 else "") + f"{abs(n):,}"


def fmt_en(n):
    return f"({abs(n):,})" if n < 0 else f"{n:,}"


def num_cell(n, lang):
    return {"raw": (fmt_ja if lang == "ja" else fmt_en)(n), "value": str(n), "value_type": "number"}


def blank_cell(lang):
    return {"raw": "－" if lang == "ja" else "—", "value": None, "value_type": "blank", "blank_kind": "dash"}


def text_cell(s):
    return {"raw": s, "value": None, "value_type": "text"}


# ---------------------------------------------------------------- vocab
V = {
    "ja": {
        "sections": [("流動資産", "流動資産合計", ["現金及び預金", "受取手形", "売掛金", "電子記録債権", "商品及び製品", "仕掛品",
                                       "原材料及び貯蔵品", "前払費用", "未収入金", "短期貸付金", "その他"], "貸倒引当金"),
                     ("固定資産", "固定資産合計", ["建物及び構築物", "機械装置及び運搬具", "土地", "リース資産", "建設仮勘定",
                                       "ソフトウエア", "のれん", "投資有価証券", "関係会社株式", "長期貸付金", "繰延税金資産"], "減価償却累計額"),
                     ("流動負債", "流動負債合計", ["支払手形及び買掛金", "短期借入金", "１年内返済予定の長期借入金", "未払金",
                                       "未払法人税等", "賞与引当金", "契約負債", "その他"], None),
                     ("固定負債", "固定負債合計", ["長期借入金", "リース債務", "退職給付に係る負債", "資産除去債務",
                                       "繰延税金負債", "その他"], None)],
        "periods": ["前連結会計年度", "当連結会計年度"], "dates": ["(2025年３月31日)", "(2026年３月31日)"],
        "unit": ("（単位：千円）", 1000, "JPY"), "title": "連結貸借対照表", "cont": "（続き）",
        "bill": {"title": "請求明細書", "head": ["品名", "数量", "単価", "金額"], "unit": ("（単位：円）", 1, "JPY"),
                 "items": ["事務用品一式", "コピー用紙Ａ４（５００枚×５冊）", "トナーカートリッジ", "配送料",
                           "ノートパソコン保守契約（年間・オンサイト対応含む）", "会議室利用料", "値引き"],
                 "tax": "消費税（10％）", "total": "合計金額"},
        "ds": {"title": "電気的特性", "head": ["項目", "記号", "最小", "標準", "最大", "単位"], "group": "規格値",
               "rows": [("電源電圧", "VDD", "V"), ("消費電流", "IDD", "mA"), ("動作温度", "TA", "℃"),
                        ("入力リーク電流", "ILK", "μA"), ("出力電圧（ハイ）", "VOH", "V"), ("クロック周波数", "fCLK", "MHz")]},
        "fn_text": "は担保に供している資産であります。", "fn_mark": "※",
        "joiner": "",
    },
    "en": {
        "sections": [("Current assets", "Total current assets", ["Cash and cash equivalents", "Trade receivables",
                      "Contract assets", "Inventories", "Prepaid expenses", "Short-term investments", "Other current assets"],
                      "Allowance for doubtful accounts"),
                     ("Non-current assets", "Total non-current assets", ["Buildings and structures", "Machinery and equipment",
                      "Land", "Right-of-use assets", "Construction in progress", "Goodwill", "Intangible assets",
                      "Investment securities", "Deferred tax assets"], "Accumulated depreciation"),
                     ("Current liabilities", "Total current liabilities", ["Trade payables", "Short-term borrowings",
                      "Current portion of long-term debt", "Accrued expenses", "Income taxes payable", "Contract liabilities"], None),
                     ("Non-current liabilities", "Total non-current liabilities", ["Long-term debt", "Lease liabilities",
                      "Retirement benefit obligations", "Asset retirement obligations", "Deferred tax liabilities"], None)],
        "periods": ["Prior year", "Current year"], "dates": ["(March 31, 2025)", "(March 31, 2026)"],
        "unit": ("(In thousands of U.S. dollars)", 1000, "USD"), "title": "Consolidated Balance Sheet",
        "cont": " (continued)",
        "bill": {"title": "Invoice details", "head": ["Description", "Qty", "Unit price", "Amount"],
                 "unit": ("(Amounts in USD)", 1, "USD"),
                 "items": ["Office supplies", "Copy paper A4 (500 sheets x 5 reams)", "Toner cartridge", "Shipping",
                           "Laptop maintenance contract (annual, including on-site support)", "Meeting room rental", "Discount"],
                 "tax": "Sales tax (8%)", "total": "Total due"},
        "ds": {"title": "Electrical characteristics", "head": ["Parameter", "Symbol", "Min", "Typ", "Max", "Unit"],
               "group": "Value",
               "rows": [("Supply voltage", "VDD", "V"), ("Supply current", "IDD", "mA"), ("Operating temperature", "TA", "°C"),
                        ("Input leakage current", "ILK", "µA"), ("Output high voltage", "VOH", "V"), ("Clock frequency", "fCLK", "MHz")]},
        "fn_text": " Pledged as collateral.", "fn_mark": "*",
        "joiner": " ",
    },
}

# ---------------------------------------------------------------- table specs
# A spec: header = list of header rows, each a list of (text, row_span, col_span) placed left to right
# into free grid slots; body = list of rows {cells:[cell...], indent, is_total, label_lines}.

def bs_body(lang, rng, n_vals, sections, footnotes=False, neg_zero=False, grand=False):
    """grand=True wraps the sections in 資産の部 ... 資産合計 / Assets ... Total assets (3 indent levels)."""
    v, body, fns = V[lang], [], []
    base = 1 if grand else 0
    empty_vals = [{"raw": "", "value": None, "value_type": "blank", "blank_kind": "empty"}] * n_vals
    if grand:
        body.append({"cells": [text_cell("資産の部" if lang == "ja" else "Assets")] + empty_vals, "indent": 0, "is_total": False})
    grand_sums = [0] * n_vals
    for si, (head, total, items, contra) in enumerate(sections):
        body.append({"cells": [text_cell(head)] + empty_vals, "indent": base, "is_total": False})
        sums = [0] * n_vals
        for ii, item in enumerate(items):
            vals = []
            for c in range(n_vals):
                if rng.random() < 0.08:
                    vals.append(blank_cell(lang))
                    continue
                n = rng.choice([rng.randint(1, 999), rng.randint(1_000, 99_999), rng.randint(100_000, 9_999_999)])
                sums[c] += n
                vals.append(num_cell(n, lang))
            label = item
            if footnotes and ii in (1, 4):
                mark = f"{v['fn_mark']}{len(fns) + 1}"
                fns.append((mark, f"{mark} {item}{v['fn_text']}"))
                label = item + mark
            row = {"cells": [text_cell(label)] + vals, "indent": base + 1, "is_total": False}
            if label != item:
                row["cells"][0]["footnote_refs"] = [label[len(item):]]
            body.append(row)
        if contra:
            vals = []
            for c in range(n_vals):
                n = 0 if (neg_zero and c == n_vals - 1) else -rng.randint(1, 99_999)
                sums[c] += n
                cell = num_cell(n, lang)
                if n == 0:  # filings print a negative zero as △0; value stays "0"
                    cell = {"raw": "△0" if lang == "ja" else "(0)", "value": "0", "value_type": "number"}
                vals.append(cell)
            body.append({"cells": [text_cell(contra)] + vals, "indent": base + 1, "is_total": False})
        body.append({"cells": [text_cell(total)] + [num_cell(s, lang) for s in sums], "indent": base + 1, "is_total": True})
        grand_sums = [g + x for g, x in zip(grand_sums, sums)]
    if grand:
        body.append({"cells": [text_cell("資産合計" if lang == "ja" else "Total assets")] + [num_cell(x, lang) for x in grand_sums],
                     "indent": 1, "is_total": True})
    return body, fns


def period_header(lang):
    v = V[lang]
    # empty stub = one empty cell per header row, no span (ParseBench GT majority: 11 of 16 tables)
    return [[("", 1, 1)] + [(p, 1, 1) for p in v["periods"]], [("", 1, 1)] + [(d, 1, 1) for d in v["dates"]]]


def spec_bs(lang, rng, style, sections, **kw):
    body, fns = bs_body(lang, rng, 2, sections, **kw)
    v = V[lang]
    return {"title": v["title"], "unit": v["unit"], "header": period_header(lang), "body": body, "footnotes": fns,
            "widths": [255, 120, 120], "align": ["L", "R", "R"], "style": style}


def spec_bs_quarters(lang, rng, style):
    """Two-level header: fiscal year spans two columns (col_span=2)."""
    v = V[lang]
    yrs = ["2025年３月期", "2026年３月期"] if lang == "ja" else ["FY2025", "FY2026"]
    sub = ["第２四半期", "通期"] if lang == "ja" else ["Q2", "Full year"]
    header = [[("", 1, 1), (yrs[0], 1, 2), (yrs[1], 1, 2)], [("", 1, 1)] + [(sub[0], 1, 1), (sub[1], 1, 1)] * 2]
    body, _ = bs_body(lang, rng, 4, v["sections"][:2])
    return {"title": v["title"], "unit": v["unit"], "header": header, "body": body, "footnotes": [],
            "widths": [175, 80, 80, 80, 80], "align": ["L", "R", "R", "R", "R"], "style": style}


def spec_bill(lang, rng, style, n_items=None):
    b = V[lang]["bill"]
    items = b["items"][: n_items or len(b["items"])]
    body, subtotal = [], 0
    for it in items:
        qty = rng.randint(1, 12)
        price = rng.randint(100, 98_000)
        amt = qty * price * (-1 if it in ("値引き", "Discount") else 1)
        subtotal += amt
        body.append({"cells": [text_cell(it), num_cell(qty, lang), num_cell(price, lang), num_cell(amt, lang)],
                     "indent": 0, "is_total": False})
    tax = subtotal * (10 if lang == "ja" else 8) // 100
    empty = {"raw": "", "value": None, "value_type": "blank", "blank_kind": "empty"}
    body.append({"cells": [text_cell(b["tax"]), empty, empty, num_cell(tax, lang)], "indent": 0, "is_total": False})
    body.append({"cells": [text_cell(b["total"]), empty, empty, num_cell(subtotal + tax, lang)], "indent": 0, "is_total": True})
    return {"title": b["title"], "unit": b["unit"], "header": [[(h, 1, 1) for h in b["head"]]], "body": body,
            "footnotes": [], "widths": [235, 60, 100, 100], "align": ["L", "R", "R", "R"], "style": style, "wrap": 34 if lang == "en" else 18}


def spec_datasheet(lang, rng, style):
    """Row spans in the header: Parameter/Symbol/Unit span 2 rows, 'Value' spans Min/Typ/Max."""
    d = V[lang]["ds"]
    h = d["head"]
    header = [[(h[0], 2, 1), (h[1], 2, 1), (d["group"], 1, 3), (h[5], 2, 1)], [(h[2], 1, 1), (h[3], 1, 1), (h[4], 1, 1)]]
    body = []
    for name, sym, unit in d["rows"]:
        lo = round(rng.uniform(-40, 5), 1)
        typ = round(lo + rng.uniform(0.5, 30), 1)
        hi = round(typ + rng.uniform(0.5, 60), 1)
        vals = [{"raw": f"{x:.1f}", "value": f"{x:.1f}", "value_type": "number"} for x in (lo, typ, hi)]
        if rng.random() < 0.4:  # datasheets often leave Min or Typ unspecified
            vals[rng.choice([0, 1])] = blank_cell(lang)
        body.append({"cells": [text_cell(name), text_cell(sym)] + vals + [text_cell(unit)], "indent": 0, "is_total": False})
    return {"title": d["title"], "unit": None, "header": header, "body": body, "footnotes": [],
            "widths": [175, 60, 65, 65, 65, 65], "align": ["L", "C", "R", "R", "R", "C"], "style": style}


# ---------------------------------------------------------------- renderer
class Doc:
    def __init__(self, lang):
        self.lang = lang
        self.pdf = pymupdf.open()
        self.font = CJK if lang == "ja" else HELV
        self.fontname = "cj" if lang == "ja" else "hv"
        self.pages = []
        self.new_page()

    def new_page(self):
        p = self.pdf.new_page(width=PAGE_W, height=PAGE_H)
        # Embedded (Identity-H) fonts so every char maps back exactly; base-14 Helvetica is Latin-1 only.
        p.insert_font(fontname=self.fontname, fontbuffer=self.font.buffer)
        self.pages.append(p)
        self.page, self.y = p, 60.0
        self.text(f"- {len(self.pages)} -", PAGE_W / 2, PAGE_H - 30, "C")  # page furniture, not table content

    def text(self, s, x, baseline, align="L", fs=FS):
        w = self.font.text_length(s, fs)
        x0 = {"L": x, "R": x - w, "C": x - w / 2}[align]
        if s:
            self.page.insert_text((x0, baseline), s, fontname=self.fontname, fontsize=fs)
        return [round(x0, 2), round(baseline - self.font.ascender * fs, 2), round(x0 + w, 2), round(baseline - self.font.descender * fs, 2)]

    def wrap(self, s, width):
        if len(s) <= width:
            return [s]
        if self.lang == "ja":
            return [s[i:i + width] for i in range(0, len(s), width)]
        lines, cur = [], ""
        for w in s.split(" "):
            if cur and len(cur) + 1 + len(w) > width:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        return lines + [cur]


def place_header(header, n_cols):
    """Grid-place header cells honoring row/col spans -> [(r, c, text, rs, cs)]."""
    taken, out = set(), []
    for r, row in enumerate(header):
        c = 0
        for text, rs, cs in row:
            while (r, c) in taken:
                c += 1
            out.append((r, c, text, rs, cs))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            c += cs
    assert all((r, c) in taken for r in range(len(header)) for c in range(n_cols)), "header does not cover grid"
    return out


def draw_table(doc, spec, table_id):
    lang, v = doc.lang, V[doc.lang]
    widths, align = spec["widths"], spec["align"]
    n_cols = len(widths)
    xs = [LEFT]
    for w in widths:
        xs.append(xs[-1] + w)
    n_hdr = len(spec["header"])
    hdr_cells = place_header(spec["header"], n_cols)
    cells, rows, regions = [], [], {}
    title_bbox = doc.text(spec["title"], LEFT, doc.y + 11, fs=11)
    doc.y += 18
    unit = None
    if spec["unit"]:
        raw, scale, cur = spec["unit"]
        doc.text(raw, xs[-1], doc.y + 9, "R")
        unit = {"raw_text": raw, "scale": scale, "currency": cur}
        doc.y += 14

    def region_add(bb):
        r = regions.setdefault(len(doc.pages), [bb[0], bb[1], bb[2], bb[3]])
        r[:] = [min(r[0], bb[0]), min(r[1], bb[1]), max(r[2], bb[2]), max(r[3], bb[3])]

    def draw_header(record):
        top = doc.y
        doc.page.draw_line((xs[0], top), (xs[-1], top), color=RULE, width=1.2)
        for r, c, text, rs, cs in hdr_cells:
            x0, x1 = xs[c], xs[c + cs]
            y0, y1 = top + r * LINE, top + (r + rs) * LINE
            base = (y0 + y1) / 2 + FS * 0.35
            bb = doc.text(text, (x0 + x1) / 2, base, "C") if text else [x0, y0, x1, y1]
            if spec["style"] == "ruled":
                doc.page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=RULE, width=0.5)
            region_add([x0, y0, x1, y1])
            if record:
                cells.append({"row": r, "col": c, "row_span": rs, "col_span": cs, "is_header": True,
                              "page": len(doc.pages), "bbox": bb, "raw_text": text, "text": text,
                              "value": None, "value_type": "text", "blank_kind": None if text else "empty",
                              "footnote_refs": []})
        doc.y = top + n_hdr * LINE
        doc.page.draw_line((xs[0], doc.y), (xs[-1], doc.y), color=RULE, width=0.6)

    draw_header(record=True)
    stack = []  # (indent, row index) for parent_row
    for bi, brow in enumerate(spec["body"]):
        r = n_hdr + bi
        label_lines = doc.wrap(brow["cells"][0]["raw"], spec["wrap"]) if spec.get("wrap") else [brow["cells"][0]["raw"]]
        h = LINE * len(label_lines)
        if doc.y + h > BOTTOM:  # table continues on the next page with a repeated header
            doc.page.draw_line((xs[0], doc.y), (xs[-1], doc.y), color=RULE, width=1.2)
            doc.new_page()
            doc.text(spec["title"] + v["cont"], LEFT, doc.y + 11, fs=11)
            doc.y += 18
            draw_header(record=False)
        top = doc.y
        if spec["style"] == "striped" and bi % 2 == 0:
            doc.page.draw_rect(pymupdf.Rect(xs[0], top, xs[-1], top + h), color=None, fill=STRIPE)
        if brow["is_total"] and spec["style"] != "ruled":
            doc.page.draw_line((xs[1], top), (xs[-1], top), color=RULE, width=0.5)
        last_base = top + h - LINE / 2 + FS * 0.35
        for c, cell in enumerate(brow["cells"]):
            x0, x1 = xs[c], xs[c + 1]
            if c == 0:
                indent = brow["indent"] * INDENT
                boxes = [doc.text(line, x0 + 4 + indent, top + i * LINE + LINE / 2 + FS * 0.35)
                         for i, line in enumerate(label_lines)]
                bb = [min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3]]
            elif cell["raw"]:
                a = align[c]
                bb = doc.text(cell["raw"], {"L": x0 + 4, "R": x1 - 4, "C": (x0 + x1) / 2}[a], last_base, a)
            else:
                bb = [x0, top, x1, top + h]
            if spec["style"] == "ruled":
                doc.page.draw_rect(pymupdf.Rect(x0, top, x1, top + h), color=RULE, width=0.5)
            region_add([x0, top, x1, top + h])
            raw = v["joiner"].join(label_lines) if c == 0 else cell["raw"]
            cells.append({"row": r, "col": c, "row_span": 1, "col_span": 1, "is_header": False,
                          "page": len(doc.pages), "bbox": [round(x, 2) for x in bb], "raw_text": raw,
                          "text": _nfkc(raw),
                          "value": cell["value"], "value_type": cell["value_type"],
                          "blank_kind": cell.get("blank_kind"), "footnote_refs": cell.get("footnote_refs", [])})
        while stack and stack[-1][0] >= brow["indent"]:
            stack.pop()
        rows.append({"row": r, "page": len(doc.pages), "indent_level": brow["indent"],
                     "parent_row": stack[-1][1] if stack else None, "is_total": brow["is_total"]})
        stack.append((brow["indent"], r))
        doc.y = top + h
    doc.page.draw_line((xs[0], doc.y), (xs[-1], doc.y), color=RULE, width=1.2)
    footnotes = []
    doc.y += 6
    for mark, text in spec["footnotes"]:
        footnotes.append({"marker": mark, "raw_text": text, "page": len(doc.pages),
                          "bbox": doc.text(text, LEFT, doc.y + 9)})
        doc.y += 12
    doc.y += 24
    columns = []
    for c in range(n_cols):
        path = [t for r, cc, t, rs, cs in sorted(hdr_cells) if cc <= c < cc + cs and t]
        columns.append({"col": c, "header_path": path, "role": "label" if c == 0 else ("value" if align[c] == "R" else "text")})
    return {"table_id": table_id, "title": spec["title"], "title_bbox": title_bbox, "unit": unit,
            "regions": [{"page": p, "bbox": [round(x, 2) for x in bb]} for p, bb in sorted(regions.items())],
            "n_rows": n_hdr + len(spec["body"]), "n_cols": n_cols, "header_rows": n_hdr,
            "columns": columns, "rows": rows, "cells": cells, "footnotes": footnotes,
            "confidence": 1.0, "needs_review": False, "issues": []}


def _nfkc(s):
    return unicodedata.normalize("NFKC", s)


def save(doc_id, doc, tables, tags, image_only=False):
    pdf = doc.pdf
    pdf.subset_fonts()
    text_source = "text_layer"
    if image_only:  # same page, no text layer: forces the OCR path
        img = pymupdf.open()
        for p in pdf:
            np_ = img.new_page(width=p.rect.width, height=p.rect.height)
            np_.insert_image(np_.rect, pixmap=p.get_pixmap(dpi=200))
        pdf, text_source = img, "ocr"
    pdf.save(PDF_DIR / f"{doc_id}.pdf", garbage=4, deflate=True)
    gold = {"doc_id": doc_id, "source_file": f"gold/pdf/{doc_id}.pdf", "tags": tags,
            "pages": [{"page": i + 1, "width": PAGE_W, "height": PAGE_H, "rotation": 0, "text_source": text_source}
                      for i in range(len(doc.pages))],
            "tables": tables}
    (JSON_DIR / f"{doc_id}.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- real sample
def sample_gold():
    """Gold for the real sample, read from its text layer (verified against the render, design.md §0)."""
    page = pymupdf.open(SAMPLE)[0]
    lines = {}
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    lines.setdefault(round(s["bbox"][1]), []).append(s)
    indent_of = {62: 0, 75: 1, 88: 2, 101: 3, 114: 4}
    hdr_y = [y for y in lines if 118 <= y <= 135]
    cells, rows, stack = [], [], []
    for r, y in enumerate(sorted(hdr_y)):  # two header rows: period name, then date
        cells.append({"row": r, "col": 0, "row_span": 1, "col_span": 1, "is_header": True, "page": 1, "bbox": None,
                      "raw_text": "", "text": "", "value": None, "value_type": "text", "blank_kind": "empty", "footnote_refs": []})
        for c, side in ((1, lambda x: x < 400), (2, lambda x: x >= 400)):
            sp = sorted((s for s in lines[y] if side(s["bbox"][0])), key=lambda s: s["bbox"][0])
            raw = "".join(s["text"] for s in sp).strip()
            cells.append({"row": r, "col": c, "row_span": 1, "col_span": 1, "is_header": True, "page": 1,
                          "bbox": [round(min(s["bbox"][0] for s in sp), 2), round(sp[0]["bbox"][1], 2),
                                   round(max(s["bbox"][2] for s in sp), 2), round(sp[0]["bbox"][3], 2)],
                          "raw_text": raw, "text": _nfkc(raw), "value": None, "value_type": "text", "blank_kind": None,
                          "footnote_refs": []})
    r = 2
    for y in sorted(y for y in lines if 145 <= y <= 670):
        sp = sorted(lines[y], key=lambda s: s["bbox"][0])
        label = sp[0]
        indent = indent_of[min(indent_of, key=lambda k: abs(k - label["bbox"][0]))]
        # A sign and its digits can be separate spans (△ | 2,979): merge every span per column.
        by_col = {0: [label]}
        for s in sp[1:]:
            by_col.setdefault(1 if s["bbox"][2] < 430 else 2, []).append(s)
        for c in range(3):
            group = by_col.get(c, [])
            raw = "".join(x["text"] for x in group).strip()
            s = [min(x["bbox"][0] for x in group), min(x["bbox"][1] for x in group),
                 max(x["bbox"][2] for x in group), max(x["bbox"][3] for x in group)] if group else None
            value, vtype, blank = None, "text" if c == 0 else "blank", None if c == 0 else "empty"
            if c and raw == "－":
                blank = "dash"
            elif c and raw:
                digits = raw.lstrip("△").replace(",", "")
                value, vtype, blank = (str(-int(digits)) if raw.startswith("△") else digits), "number", None
            cells.append({"row": r, "col": c, "row_span": 1, "col_span": 1, "is_header": False, "page": 1,
                          "bbox": [round(x, 2) for x in s] if s else None, "raw_text": raw, "text": _nfkc(raw),
                          "value": value, "value_type": vtype, "blank_kind": blank, "footnote_refs": []})
        while stack and stack[-1][0] >= indent:
            stack.pop()
        rows.append({"row": r, "page": 1, "indent_level": indent, "parent_row": stack[-1][1] if stack else None,
                     "is_total": label["text"].strip().endswith("合計")})
        stack.append((indent, r))
        r += 1
    # sanity: identities from the statement itself, within 千円 rounding (1 unit per addend, design.md §5)
    lab = {}
    for c in cells:
        if c["col"] == 0 and not c["is_header"]:
            lab.setdefault(c["raw_text"], []).append(c["row"])
    val = {(c["row"], c["col"]): int(c["value"]) for c in cells if c["value"] is not None}
    first = lambda name: lab[name][0]
    for col in (1, 2):
        cur = [k for k in range(first("流動資産") + 1, first("流動資産合計"))]
        checks = [(first("流動資産合計"), cur),
                  (first("資産合計"), [first("流動資産合計"), first("固定資産合計")]),
                  (first("固定資産合計"), [first("有形固定資産合計"), first("無形固定資産合計"), first("投資その他の資産合計")]),
                  (first("建物及び構築物（純額）"), [first("建物及び構築物"), first("建物及び構築物") + 1])]
        for total, kids in checks:
            got = sum(val.get((k, col), 0) for k in kids)
            assert abs(got - val[(total, col)]) <= len(kids), (total, col, got, val[(total, col)])
    regions = [{"page": 1, "bbox": [55.0, 115.0, 528.0, 672.0]}]
    table = {"table_id": "t1", "title": "（１）連結貸借対照表", "unit": {"raw_text": "（単位：千円）", "scale": 1000, "currency": "JPY"},
             "regions": regions, "n_rows": r, "n_cols": 3, "header_rows": 2,
             "columns": [{"col": 0, "header_path": [], "role": "label"},
                         {"col": 1, "header_path": [c["raw_text"] for c in cells if c["is_header"] and c["col"] == 1], "role": "value"},
                         {"col": 2, "header_path": [c["raw_text"] for c in cells if c["is_header"] and c["col"] == 2], "role": "value"}],
             "rows": rows, "cells": cells, "footnotes": [], "confidence": 1.0, "needs_review": False, "issues": []}
    gold = {"doc_id": "real_ja_balance_sheet", "source_file": "samples/" + SAMPLE.name,
            "tags": ["real", "ja", "financial", "striped", "hierarchy", "multi_header_lines", "text_layer"],
            "pages": [{"page": 1, "width": 595.2, "height": 841.8, "rotation": 0, "text_source": "text_layer"}],
            "tables": [table]}
    (JSON_DIR / "real_ja_balance_sheet.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1))
    return r


# ---------------------------------------------------------------- golden set
def build():
    PDF_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)
    rng = random.Random(7)
    made = []
    for lang in ("ja", "en"):
        secs = V[lang]["sections"]
        cases = [
            # (name, tags, builder, ocr_variant)
            ("bs_striped_hierarchy", ["financial", "striped", "hierarchy"],
             lambda: [spec_bs(lang, rng, "striped", secs[:2], neg_zero=True, grand=True)], True),
            ("bs_borderless_footnotes", ["financial", "borderless", "footnotes", "blanks"],
             lambda: [spec_bs(lang, rng, "borderless", secs[2:], footnotes=True)], False),
            ("bs_ruled_colspan_header", ["financial", "ruled", "multi_header", "colspan"],
             lambda: [spec_bs_quarters(lang, rng, "ruled")], False),
            ("bill_ruled_wrapped", ["bill", "ruled", "wrapped_label"],
             lambda: [spec_bill(lang, rng, "ruled")], True),
            ("datasheet_rowspan_header", ["datasheet", "ruled", "multi_header", "rowspan", "decimals", "blanks"],
             lambda: [spec_datasheet(lang, rng, "ruled")], True),
            ("two_tables_one_page", ["multi_table", "borderless", "ruled"],
             lambda: [spec_bill(lang, rng, "borderless", n_items=4), spec_datasheet(lang, rng, "ruled")], False),
            ("bs_cross_page", ["financial", "striped", "hierarchy", "cross_page"],
             lambda: [spec_bs(lang, rng, "striped", secs + secs, grand=True)], True),
        ]
        for name, tags, builder, ocr in cases:
            specs = builder()
            for image_only in ((False, True) if ocr else (False,)):
                doc_id = f"syn_{lang}_{name}" + ("_ocr" if image_only else "")
                doc = Doc(lang)
                tables = [draw_table(doc, s, f"t{i + 1}") for i, s in enumerate(specs)]
                save(doc_id, doc, tables, ["synthetic", lang] + tags + (["ocr"] if image_only else ["text_layer"]), image_only)
                made.append(doc_id)
    n = sample_gold()
    made.append("real_ja_balance_sheet")
    print(f"built {len(made)} gold docs ({n} rows in real sample)")
    return made


if __name__ == "__main__":
    build()
