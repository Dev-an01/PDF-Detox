"""OCR fallback (design.md §1.2, m6): only for a page or region whose text layer is missing,
garbled or fails the audit. Engine choice is one setting, PARSER_OCR:

    lighton+tesseract  (default) Tesseract gives words + boxes; a number is accepted only if
                       LightOnOCR-3-1B read the same number in the same block, in the same order.
                       The only system with 0 unflagged wrong numbers on the OCR bench (ocr_bench/).
    tesseract          Tesseract alone, flags word conf < 90. Fast, but NOT zero-unflagged (bench: 26 on gold OCR).
    none               no OCR; the page is flagged `no_text_layer` and has no tables.

Results are cached in .ocr_cache/ (file hash + page + engine + region), since LightOnOCR takes
~70-145 s per page on an M5 (MPS).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

import engines  # LightOn loader, tool paths, number definition (shared with ocr_bench/)
from engines import nums

ROOT = engines.ROOT

ENGINE = os.environ.get("PARSER_OCR", "lighton+tesseract")
CACHE = ROOT / ".ocr_cache"
DPI = 300
TESS_CONF = 90.0
SIGN_UNCERTAIN = re.compile(r"^\(?[-−–]")  # OCR cannot tell hyphen / minus / en dash apart
_lighton = None


class Words(HTMLParser):
    """Tesseract hOCR -> words with box (px) and confidence."""

    def __init__(self):
        super().__init__()
        self.words, self.cur = [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("class") == "ocrx_word":
            m = re.search(r"bbox (\d+) (\d+) (\d+) (\d+); x_wconf (\d+)", a.get("title", ""))
            self.cur = {"text": "", "bbox": [float(v) for v in m.groups()[:4]], "conf": float(m.group(5))}

    def handle_endtag(self, tag):
        if tag == "span" and self.cur is not None:
            if self.cur["text"].strip():
                self.words.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cur is not None:
            self.cur["text"] += data


def _key(pdf_path, pno, engine, clip):
    h = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()[:16]
    c = "page" if clip is None else "_".join(str(round(v)) for v in clip)
    return CACHE / f"{h}_p{pno}_{engine}_{c}.json"


def _cached(path, fn):
    if path.exists():
        return json.loads(path.read_text())
    res = fn()
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False))
    return res


def tesseract_words(page, clip=None):
    """-> words in page points: {text, bbox, conf}."""
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "p.png"
        pix = page.get_pixmap(dpi=DPI, clip=clip)
        pix.save(img)
        out = subprocess.run(["tesseract", str(img), "stdout", "-l", "jpn+eng", "--dpi", str(DPI), "hocr"],
                             capture_output=True, text=True, check=True,
                             env={**os.environ, "TESSDATA_PREFIX": str(engines.TESSDATA)})
    p = Words()
    p.feed(out.stdout)
    s = 72.0 / DPI
    x0, y0 = (clip.x0, clip.y0) if clip is not None else (0.0, 0.0)
    return [{"text": unicodedata.normalize("NFC", w["text"].strip()), "conf": w["conf"],
             "bbox": [x0 + w["bbox"][0] * s, y0 + w["bbox"][1] * s, x0 + w["bbox"][2] * s, y0 + w["bbox"][3] * s]}
            for w in p.words]


def lighton_blocks(page, clip=None):
    """-> {blocks:[{label, bbox (page pt), text}], truncated, empty}. Rendered per the 1B card."""
    global _lighton
    if _lighton is None:
        _lighton = engines.LightOn("lighton-1b")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdf = Path(tmp) / "one.pdf"
        one = pymupdf.open()
        one.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
        if clip is not None:
            one[0].set_cropbox(clip)
        one.save(tmp_pdf)
        res = _lighton({"pdf": str(tmp_pdf), "page": 0}, tmp)
    dx, dy = (clip.x0, clip.y0) if clip is not None else (0.0, 0.0)
    blocks = [dict(b, bbox=[b["bbox"][0] + dx, b["bbox"][1] + dy, b["bbox"][2] + dx, b["bbox"][3] + dy])
              for b in res["blocks"]]
    return {"blocks": blocks, "truncated": res["truncated"], "empty": not blocks, "seconds": res["seconds"]}


def block_runs(text):
    t = re.sub(r"</t[dh]>", " │ ", text)
    t = re.sub(r"</tr>|<br\s*/?>", "\n", t)
    t = unicodedata.normalize("NFKC", re.sub(r"<[^>]+>", " ", t).replace("&amp;", "&"))
    return [r for _, _, r in nums(t)]


def _center_in(b, box, pad):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return box[0] - pad <= cx <= box[2] + pad and box[1] - pad <= cy <= box[3] + pad


def page_data(pdf_path, page, clip=None):
    """(Tesseract words, LightOn blocks) for a page, from the cache."""
    pno = page.number + 1
    words = _cached(_key(pdf_path, pno, "tesseract", clip), lambda: tesseract_words(page, clip))
    lo = _cached(_key(pdf_path, pno, "lighton-1b", clip), lambda: lighton_blocks(page, clip))
    return words, lo


def ocr_region(pdf_path, page, clip=None):
    """OCR one page (or one clip of it). -> (units, report).
    units: parser glyph units {c, bbox, size, unit, ocr_ok, ocr_reason}; one per Tesseract word.
    A unit is ocr_ok only when every number in it was confirmed (see module doc)."""
    pno = page.number + 1
    words = _cached(_key(pdf_path, pno, "tesseract", clip), lambda: tesseract_words(page, clip))
    report = {"engine": ENGINE, "words": len(words), "lighton": None, "page_unverified": False, "missing_blocks": []}
    for w in words:
        w["runs"] = [r for _, _, r in nums(unicodedata.normalize("NFKC", w["text"]))]
        w["ok"] = {k: False for k in range(len(w["runs"]))}
    if ENGINE == "tesseract":
        for w in words:
            for k, r in enumerate(w["runs"]):
                w["ok"][k] = w["conf"] >= TESS_CONF and not SIGN_UNCERTAIN.match(r)
    elif ENGINE == "lighton+tesseract":
        lo = _cached(_key(pdf_path, pno, "lighton-1b", clip), lambda: lighton_blocks(page, clip))
        report["lighton"] = {"blocks": len(lo["blocks"]), "truncated": lo["truncated"], "empty": lo["empty"]}
        if lo["empty"] or lo["truncated"]:
            report["page_unverified"] = True  # nothing on the page can be confirmed
        else:
            pad = 0.01 * page.rect.width
            used = set()
            for bi, b in enumerate(lo["blocks"]):
                mine = [(wi, k) for wi, w in enumerate(words) if wi not in used and _center_in(w["bbox"], b["bbox"], pad)
                        for k in range(len(w["runs"]))]
                used |= {wi for wi, _ in mine}
                mine.sort(key=lambda wk: (round(words[wk[0]]["bbox"][1] / 4), words[wk[0]]["bbox"][0]))
                theirs = block_runs(b["text"])
                sm = SequenceMatcher(None, [words[wi]["runs"][k] for wi, k in mine], theirs, autojunk=False)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        for wi, k in mine[i1:i2]:
                            words[wi]["ok"][k] = not SIGN_UNCERTAIN.match(words[wi]["runs"][k])
                    elif j2 > j1:  # LightOn saw numbers here that Tesseract did not report
                        report["missing_blocks"].append(b["bbox"])
    units = []
    for w in words:
        bad = [r for k, r in enumerate(w["runs"]) if not w["ok"][k]]
        units.append({"c": w["text"], "bbox": w["bbox"], "size": max(w["bbox"][3] - w["bbox"][1], 1.0), "unit": True,
                      "ocr_ok": not bad, "ocr_reason": None if not bad else ("ocr_unconfirmed:" + ",".join(bad))})
    return units, report


def _selftest():
    assert block_runs("<table><tr><td>1,234</td><td>(5)</td></tr></table>") == ["1,234", "(5)"]
    p = Words()
    p.feed("<span class='ocrx_word' id='w1' title='bbox 10 20 30 40; x_wconf 96'>2,979</span>")
    assert p.words == [{"text": "2,979", "bbox": [10.0, 20.0, 30.0, 40.0], "conf": 96.0}]
    assert SIGN_UNCERTAIN.match("-200") and SIGN_UNCERTAIN.match("(–5)") and not SIGN_UNCERTAIN.match("△200")
    print("ocr self-check ok")


if __name__ == "__main__":
    _selftest()


def vision_chars(page, dpi=200):
    """Apple Vision (OS framework, no install) on the rendered page -> [(char, bbox pt)] in display
    space. Used only for the text-layer spot check."""
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "p.png"
        page.get_pixmap(dpi=dpi).save(img)
        out = subprocess.run([str(engines.vision_bin()), str(img)], capture_output=True, text=True, check=True)
    s = 72.0 / dpi
    return [(unicodedata.normalize("NFKC", c["c"]), [v * s for v in c["bbox"]])
            for line in json.loads(out.stdout)["lines"] for c in line["chars"] if c["bbox"]]


def spot_check(page, number_cells, sample=40):
    """Share of text-layer number cells whose characters Vision reads the same at the same place.
    A broken ToUnicode map (text layer says one thing, the page shows another) scores near 0."""
    step = max(1, len(number_cells) // sample)
    picked = number_cells[::step][:sample]
    vc = vision_chars(page)
    ok = seen = 0
    for c in picked:
        got = "".join(ch for ch, b in sorted(vc, key=lambda x: x[1][0]) if _center_in(b, c["bbox"], 1.0)).replace(" ", "")
        if not got:  # Vision read nothing here (tiny fonts, some whole pages): no evidence either way
            continue
        seen += 1
        ok += got == unicodedata.normalize("NFKC", c["raw_text"]).replace(" ", "")
    return (ok / seen if seen else None), seen
