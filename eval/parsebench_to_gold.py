"""Convert ParseBench table ground truth (HTML in table.jsonl) into the design.md §2 gold schema.

    python eval/parsebench_to_gold.py parsebench/table.jsonl eval/parsebench_gold

ParseBench gives no cell bboxes and no normalized values, so converted cells carry raw_text only
(no `value`, no `regions`); evaluate.py then matches tables by content and scores numeric cells
on their exact printed text. The official GTRM score still comes from ParseBench's own scorer.
"""
import json
import sys
from html.parser import HTMLParser
from pathlib import Path


class TableHTML(HTMLParser):
    """<table>/<tr>/<td|th rowspan colspan> -> list of tables, each a list of rows of cells."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.cell, self.depth = [], None, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.tables.append([])
        elif self.depth != 1:
            return  # nested tables are kept as text of the outer cell
        elif tag == "tr":
            self.tables[-1].append([])
        elif tag in ("td", "th"):
            if not self.tables[-1]:
                self.tables[-1].append([])
            self.cell = {"th": tag == "th", "rs": int(a.get("rowspan") or 1), "cs": int(a.get("colspan") or 1), "text": []}
        elif tag == "br" and self.cell is not None:
            self.cell["text"].append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        elif tag in ("td", "th") and self.cell is not None and self.depth == 1:
            self.tables[-1][-1].append(self.cell)
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"].append(data)


def to_table(rows, table_id):
    taken, cells = set(), []
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in taken:
                c += 1
            raw = " ".join("".join(cell["text"]).split())
            cells.append({"row": r, "col": c, "row_span": cell["rs"], "col_span": cell["cs"], "is_header": cell["th"],
                          "raw_text": raw})
            taken |= {(r + dr, c + dc) for dr in range(cell["rs"]) for dc in range(cell["cs"])}
            c += cell["cs"]
    n_rows = max((r for r, _ in taken), default=-1) + 1
    n_cols = max((c for _, c in taken), default=-1) + 1
    header_rows = 0
    while header_rows < n_rows and all(c["is_header"] for c in cells if c["row"] == header_rows):
        header_rows += 1
    return {"table_id": table_id, "n_rows": n_rows, "n_cols": n_cols, "header_rows": header_rows, "cells": cells}


def convert(jsonl, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in Path(jsonl).read_text().splitlines():
        rec = json.loads(line)
        p = TableHTML()
        p.feed(rec["expected_markdown"])
        doc_id = Path(rec["pdf"]).stem
        gold = {"doc_id": doc_id, "source_file": f"parsebench/{rec['pdf']}",
                "tags": ["parsebench", "en"] + rec["tags"] + [k for k, v in json.loads(rec["rule"]).items() if v],
                "tables": [to_table(rows, f"t{i + 1}") for i, rows in enumerate(p.tables) if rows]}
        (out / f"{doc_id}.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1))
        n += 1
    return n


def _selftest():
    p = TableHTML()
    p.feed('<table><tr><th rowspan="2">A</th><th colspan="2">B</th></tr><tr><th>b1</th><th>b2</th></tr>'
           '<tr><td><strong>x</strong><br/>y</td><td>1,234</td><td>(5)</td></tr></table>')
    t = to_table(p.tables[0], "t1")
    assert (t["n_rows"], t["n_cols"], t["header_rows"]) == (3, 3, 2), t
    assert [(c["row"], c["col"]) for c in t["cells"]] == [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert t["cells"][4]["raw_text"] == "x y"
    print("self-check ok")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _selftest()
        sys.exit(0)
    print(f"converted {convert(sys.argv[1], sys.argv[2])} docs -> {sys.argv[2]}")
