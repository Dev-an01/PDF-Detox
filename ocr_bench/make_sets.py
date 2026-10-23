"""Build the three OCR test sets and their answers -> ocr_bench/sets/manifest.json

    python ocr_bench/make_sets.py

  gold_ocr     8 image-only gold docs (10 pages). Answer: the text layer of each doc's text twin
               (same drawing, checked by gold/check_gold.py).
  pb_notext    the 6 ParseBench table pages with no usable text layer. Answer: ParseBench's HTML
               tables, so order only, no positions.
  real_render  the real sample + 39 ParseBench pages whose text layer passes the audit, rendered to
               images. Answer: their own text layer.

The unit scored is a "number": a run like `2,002,429`, `△2,979`, `(1,234)`, `12.5%`, `-40.0`,
found by NUM_RE after NFKC (so ３ == 3 and full-width brackets fold).
"""
import json
import random
import sys
import re
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ocr_bench" / "sets"
sys.path.insert(0, str(ROOT / "src"))
from engines import NUM_RE, nums  # noqa: E402,F401  (one number definition for parser OCR and bench)
PB_NOTEXT = ["2023-conocophillips-aim-presentation (1)_page7", "BSC-Hospital-List-by-County (2)_page1",
             "LTCprodkitLincolnStateAvailabilty0216_page1", "SERFF_CA_random_pages 1_page2600",
             "SERFF_CA_random_pages 1_page275", "SERFF_TX_random_pages 1_page298"]


def text_layer_numbers(page):
    """Numbers with tight boxes from a page's text layer. A gap > 0.25 em between characters
    counts as a space (producers often draw adjacent cells with no space character)."""
    found = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            pieces, prev = [], None
            for span in line["spans"]:
                for ch in span["chars"]:
                    if prev is not None and ch["bbox"][0] - prev[2] > 0.25 * span["size"]:
                        pieces.append((" ", None))
                    for c in unicodedata.normalize("NFKC", ch["c"]):
                        pieces.append((c, ch["bbox"]))
                    prev = ch["bbox"]
            text = "".join(c for c, _ in pieces)
            for s, e, run in nums(text):
                boxes = [bb for _, bb in pieces[s:e] if bb]
                found.append({"t": run, "bbox": [round(min(x[0] for x in boxes), 2), round(min(x[1] for x in boxes), 2),
                                                 round(max(x[2] for x in boxes), 2), round(max(x[3] for x in boxes), 2)]})
    return found


def audit_ok(page):
    """Text layer good enough to serve as the answer (design.md §1.1, the cheap checks)."""
    t = page.get_text()
    if len(t.strip()) < 200 or any(c == "�" or unicodedata.category(c) in ("Co", "Cc") and c not in "\n\t" for c in t):
        return False
    return all(tr["type"] != 3 for tr in page.get_texttrace())  # render mode 3 = invisible (old OCR)


def item(id_, set_, pdf, page_index, gt, ordered_only=False):
    page = pymupdf.open(ROOT / pdf)[page_index]
    return {"id": id_, "set": set_, "pdf": str(pdf), "page": page_index, "width": page.rect.width,
            "height": page.rect.height, "ordered_only": ordered_only, "gt": gt}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted((ROOT / "gold" / "pdf").glob("*_ocr.pdf")):
        twin = pymupdf.open(f.with_name(f.name.replace("_ocr.pdf", ".pdf")))
        for i, p in enumerate(twin):
            items.append(item(f"{f.stem}_p{i + 1}", "gold_ocr", f.relative_to(ROOT), i, text_layer_numbers(p)))
    for stem in PB_NOTEXT:
        gold = json.loads((ROOT / "eval" / "parsebench_gold" / f"{stem}.json").read_text())
        gt = []
        for t in gold["tables"]:
            for c in sorted(t["cells"], key=lambda c: (c["row"], c["col"])):
                gt += [{"t": run, "bbox": None} for _, _, run in nums(unicodedata.normalize("NFKC", c["raw_text"]))]
        items.append(item(stem, "pb_notext", Path("parsebench/docs/table") / f"{stem}.pdf", 0, gt, ordered_only=True))
    sample = Path("samples/balance_sheet_after_correction.pdf")
    items.append(item("real_ja_balance_sheet", "real_render", sample, 0, text_layer_numbers(pymupdf.open(ROOT / sample)[0])))
    recs = [json.loads(l) for l in (ROOT / "parsebench" / "table.jsonl").read_text().splitlines()]
    rng = random.Random(7)
    rng.shuffle(recs)
    picked = {"easy": 0, "hard": 0}
    quota = {"easy": 33, "hard": 6}  # 39 pages, proportional to ParseBench's 427 easy / 76 hard
    for r in recs:
        tag = r["tags"][0]
        if picked[tag] >= quota[tag]:
            continue
        page = pymupdf.open(ROOT / "parsebench" / r["pdf"])[0]
        gt = text_layer_numbers(page) if audit_ok(page) else []
        if len(gt) < 20:
            continue
        items.append(item(Path(r["pdf"]).stem, "real_render", Path("parsebench") / r["pdf"], 0, gt))
        picked[tag] += 1
    (OUT / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    for s in ("gold_ocr", "pb_notext", "real_render"):
        sel = [i for i in items if i["set"] == s]
        print(f"{s:12s} pages={len(sel):3d} numbers={sum(len(i['gt']) for i in sel)}")


if __name__ == "__main__":
    assert [r for _, _, r in nums("△2,979 (1,234) 12.5% -40.0 2025年3月31日. 1,000,")] == \
        ["△2,979", "(1,234)", "12.5%", "-40.0", "2025", "3", "31", "1,000"]
    build()
