"""Check the golden set against its own PDFs. Run after build_gold.py; exits 1 on any problem.

    python gold/check_gold.py

1. text-layer docs: the PDF text inside each gold cell bbox equals the cell's raw_text
2. ocr docs: the PDF really has no text layer
3. every table: spans tile the n_rows x n_cols grid exactly once; header rows come first
"""
import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent


def squash(s):
    return "".join(s.split())


def check(gold):
    errs = []
    pdf = pymupdf.open(ROOT / gold["source_file"])
    has_text = any(p.get_text().strip() for p in pdf)
    if gold["pages"][0]["text_source"] == "ocr" and has_text:
        errs.append("ocr doc has a text layer")
    for t in gold["tables"]:
        seen = {}
        for c in t["cells"]:
            for r in range(c["row"], c["row"] + c["row_span"]):
                for col in range(c["col"], c["col"] + c["col_span"]):
                    seen[(r, col)] = seen.get((r, col), 0) + 1
            if c["is_header"] != (c["row"] < t["header_rows"]):
                errs.append(f"{t['table_id']} r{c['row']}c{c['col']}: header flag vs header_rows")
            if gold["pages"][0]["text_source"] == "text_layer" and c["raw_text"] and c["bbox"]:
                got = pdf[c["page"] - 1].get_textbox(pymupdf.Rect(c["bbox"]) + (-0.3, -0.3, 0.3, 0.3))
                if squash(got) != squash(c["raw_text"]):
                    errs.append(f"{t['table_id']} r{c['row']}c{c['col']}: gold {c['raw_text']!r} pdf {got!r}")
        grid = {(r, c) for r in range(t["n_rows"]) for c in range(t["n_cols"])}
        if set(seen) != grid or any(v != 1 for v in seen.values()):
            errs.append(f"{t['table_id']}: spans do not tile the grid exactly once")
    return errs


if __name__ == "__main__":
    bad = 0
    for f in sorted((ROOT / "gold" / "json").glob("*.json")):
        errs = check(json.loads(f.read_text()))
        bad += len(errs)
        for e in errs[:5]:
            print(f"{f.stem}: {e}")
    print("gold ok" if not bad else f"{bad} problems")
    sys.exit(1 if bad else 0)
