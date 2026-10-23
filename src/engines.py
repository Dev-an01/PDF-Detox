"""Shared OCR engine pieces, used by the parser's OCR fallback (src/ocr.py) and the OCR bench
(ocr_bench/). Local tool paths:

    models/tessdata/   Tesseract jpn, jpn_vert, eng, osd (tessdata_best) + hocr configs, not in git
    bin/vision_ocr     Apple Vision helper, compiled from src/vision_ocr.swift on first use, not in git
    LightOnOCR-3-1B    downloaded by transformers into ~/.cache/huggingface on first use
"""
import re
import subprocess
import time
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
TESSDATA = ROOT / "models" / "tessdata"
VISION_SRC = ROOT / "src" / "vision_ocr.swift"
VISION_BIN = ROOT / "bin" / "vision_ocr"
OCR_DPI = 300  # tesseract / vision

# One number as the benches count it: 2,002,429  △2,979  (1,234)  12.5%  -40.0 (after NFKC)
NUM_RE = re.compile(r"[(\[]?[△▲+\-−]?\d[\d,.]*%?[)\]]?")


def nums(text):
    """NFKC text -> [(start, end, run)]; trailing , or . is punctuation, not part of the number."""
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group().rstrip(",.")
        out.append((m.start(), m.start() + len(s), s))
    return out


def vision_bin():
    """Path to the Apple Vision helper; compiled with the system swiftc if missing (no install)."""
    if not VISION_BIN.exists():
        VISION_BIN.parent.mkdir(exist_ok=True)
        subprocess.run(["swiftc", "-O", str(VISION_SRC), "-o", str(VISION_BIN)], check=True)
    return VISION_BIN


# Rendering per each LightOnOCR-3 model card (they differ): (dpi, longest side px)
LIGHTON = {
    "lighton-0.8b": {"repo": "lightonai/LightOnOCR-3-0.8B", "dpi": 400, "max_side": 2048},
    "lighton-1b": {"repo": "lightonai/LightOnOCR-3-1B", "dpi": 200, "max_side": 1540},
}
MAX_NEW_TOKENS = 4096  # the card snippet uses 1024, which truncates dense table pages
BLOCK_RE = re.compile(r"!\[([a-zA-Z_+]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*")


def render(item, dpi, path, max_side=None):
    page = pymupdf.open(ROOT / item["pdf"])[item["page"]]
    pix = page.get_pixmap(dpi=dpi)
    if max_side and max(pix.width, pix.height) > max_side:
        from PIL import Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.thumbnail((max_side, max_side))
        img.save(path)
        return img.width / page.rect.width  # px per point
    pix.save(path)
    return pix.width / page.rect.width



class LightOn:
    """Loads exactly as the model cards show: 0.8B via Qwen3_5 classes in bf16 (the snippet you
    gave), 1B via LightOnOcr classes with float32 on MPS (the 1B card's own MPS code path)."""

    def __init__(self, name):
        import torch
        from transformers import AutoProcessor
        self.torch, self.cfg = torch, LIGHTON[name]
        repo = self.cfg["repo"]
        if name == "lighton-0.8b":
            from transformers import Qwen3_5ForConditionalGeneration as M
            # snippet used device_map="auto", which needs `accelerate` (not installed); .to("mps") is equivalent here
            self.model = M.from_pretrained(repo, dtype=torch.bfloat16).to("mps")
            self.template_kw = {"enable_thinking": False}
        else:
            from transformers import LightOnOcrForConditionalGeneration as M
            self.model = M.from_pretrained(repo, dtype=torch.float32).to("mps")
            self.template_kw = {}
        self.processor = AutoProcessor.from_pretrained(repo)

    def __call__(self, item, tmp):
        from PIL import Image
        img_path = Path(tmp) / "p.png"
        scale = render(item, self.cfg["dpi"], img_path, self.cfg["max_side"])
        img = Image.open(img_path).convert("RGB")
        conv = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "grounding"}]}]
        inputs = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt", **self.template_kw).to(self.model.device)
        t0 = time.perf_counter()
        # greedy: the card snippet samples (temperature 0.2), which does not repeat run to run
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if self.model.device.type == "mps":
            self.torch.mps.synchronize()
        secs = time.perf_counter() - t0
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen, skip_special_tokens=True)
        W, H = img.width / scale, img.height / scale  # page size in points
        marks = list(BLOCK_RE.finditer(text))
        blocks = []
        for k, m in enumerate(marks):
            end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
            x0, y0, x1, y1 = (int(m[i]) for i in range(2, 6))
            blocks.append({"label": m[1], "bbox": [x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H],
                           "text": text[m.end():end].strip()})
        mps_mb = self.torch.mps.driver_allocated_memory() / 2**20 if self.model.device.type == "mps" else None
        return {"seconds": secs, "blocks": blocks, "raw": text, "new_tokens": int(gen.shape[0]),
                "truncated": int(gen.shape[0]) >= MAX_NEW_TOKENS, "mps_driver_mb": mps_mb}
