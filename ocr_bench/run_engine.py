"""Run one OCR engine over every page in the manifest; cache raw output per page.

    python ocr_bench/run_engine.py tesseract|vision|vision-auto|lighton-0.8b|lighton-1b [--limit N]

vision = languages [ja-JP, en-US] (first listed dominates); vision-auto = Vision's language auto-detect.

Output: ocr_bench/out/<engine>/<id>.json  (boxes converted to PDF points, origin top-left)
        ocr_bench/out/<engine>/_run.json  (load time, peak memory)
Boxed engines (tesseract, vision) give lines of characters with boxes + confidence.
LightOnOCR gives labeled blocks (box per block, 0-1000 normalized on the image) + text.
Pages already cached are skipped, so an interrupted run resumes.
"""
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "ocr_bench"
sys.path.insert(0, str(ROOT / "src"))
from engines import LIGHTON, OCR_DPI, TESSDATA, LightOn, render, vision_bin  # noqa: E402  (shared with the parser)

# ---------------------------------------------------------------- tesseract
class HOCR(HTMLParser):
    """hOCR with hocr_char_boxes=1 -> lines of chars; each char carries its word's x_wconf."""

    def __init__(self):
        super().__init__()
        self.lines, self.stack, self.word_conf, self.char_box = [], [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls, title = a.get("class", ""), a.get("title", "")
        self.stack.append(cls)
        if cls in ("ocr_line", "ocr_textfloat", "ocr_header", "ocr_caption"):
            self.lines.append({"chars": [], "conf": None})
        elif cls == "ocrx_word":
            if self.lines and self.lines[-1]["chars"]:
                self.lines[-1]["chars"].append({"c": " ", "bbox": None, "conf": None})
            self.word_conf = float(re.search(r"x_wconf (\d+)", title).group(1))
        elif cls == "ocrx_cinfo":
            self.char_box = [float(x) for x in re.search(r"x_bboxes (\d+) (\d+) (\d+) (\d+)", title).groups()]

    def handle_endtag(self, tag):
        cls = self.stack.pop() if self.stack else ""
        if cls == "ocrx_cinfo":
            self.char_box = None

    def handle_data(self, data):
        if self.char_box is not None and data and self.lines:
            self.lines[-1]["chars"].append({"c": data, "bbox": self.char_box, "conf": self.word_conf})


def run_tesseract(item, tmp):
    img = Path(tmp) / "p.png"
    scale = render(item, OCR_DPI, img)
    t0 = time.perf_counter()
    out = subprocess.run(["tesseract", str(img), "stdout", "-l", "jpn+eng", "--dpi", str(OCR_DPI),
                          "-c", "hocr_char_boxes=1", "hocr"], capture_output=True, text=True, check=True,
                         env={**os.environ, "TESSDATA_PREFIX": str(TESSDATA)})
    secs = time.perf_counter() - t0
    p = HOCR()
    p.feed(out.stdout)
    lines = []
    for l in p.lines:
        for c in l["chars"]:
            c["bbox"] = [round(v / scale, 2) for v in c["bbox"]] if c["bbox"] else None
        if l["chars"]:
            lines.append({"text": "".join(c["c"] for c in l["chars"]), "chars": l["chars"]})
    return {"seconds": secs, "lines": lines}


# ---------------------------------------------------------------- apple vision
def run_vision(item, tmp, langs="ja-JP,en-US"):
    img = Path(tmp) / "p.png"
    scale = render(item, OCR_DPI, img)
    t0 = time.perf_counter()
    out = subprocess.run([str(vision_bin()), str(img), langs], capture_output=True, text=True, check=True)
    secs = time.perf_counter() - t0
    lines = []
    for l in json.loads(out.stdout)["lines"]:
        chars = [{"c": c["c"], "bbox": [round(v / scale, 2) for v in c["bbox"]] if c["bbox"] else None,
                  "conf": l["conf"]} for c in l["chars"]]
        lines.append({"text": l["text"], "chars": chars})
    return {"seconds": secs, "lines": lines}


# ---------------------------------------------------------------- lighton
def main():
    engine = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    items = json.loads((BENCH / "sets" / "manifest.json").read_text())[:limit]
    out_dir = BENCH / "out" / engine
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if engine == "tesseract":
        fn = run_tesseract
    elif engine == "vision":
        fn = run_vision
    elif engine == "vision-auto":
        fn = lambda it, tmp: run_vision(it, tmp, "auto")
    else:
        fn = LightOn(engine)
    load_s = time.perf_counter() - t0
    mps_peak = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for n, it in enumerate(items, 1):
            dst = out_dir / f"{it['id']}.json"
            if dst.exists():
                continue
            res = fn(it, tmp)
            mps_peak = max(mps_peak, res.get("mps_driver_mb") or 0)
            dst.write_text(json.dumps(res, ensure_ascii=False))
            print(f"[{engine}] {n}/{len(items)} {it['id']} {res['seconds']:.1f}s", flush=True)
    self_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS reports bytes
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    run = {"engine": engine, "load_seconds": load_s, "peak_rss_self_mb": self_mb, "peak_rss_child_mb": child_mb,
           "peak_mps_driver_mb": mps_peak or None}
    prev = out_dir / "_run.json"
    if prev.exists():  # resumed runs: keep the worst case seen
        old = json.loads(prev.read_text())
        for k in ("peak_rss_self_mb", "peak_rss_child_mb", "peak_mps_driver_mb"):
            run[k] = max(run[k] or 0, old.get(k) or 0) or None
    prev.write_text(json.dumps(run, indent=1))
    print(json.dumps(run))


if __name__ == "__main__":
    main()
