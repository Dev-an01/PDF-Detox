"""Rerun every check and evaluation from scratch, in order. Stops at the first failure.

    python eval/run_all.py                 self-checks, gold, ParseBench (if present), scoreboard, OCR bench scores
    python eval/run_all.py --no-parsebench gold only (no dataset needed)
    python eval/run_all.py --rebuild-gold  also regenerate the synthetic gold PDFs (changes their bytes, so
                                           OCR on the 8 image-only docs is recomputed: ~12 min LightOnOCR)
    python eval/run_all.py --ocr-rerun     also rerun every OCR engine on the 56 bench pages (hours)

Results: docs/scores.md, docs/scores.html (row "final"), preds/final*, ocr_bench/results*.json.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def step(title, *cmd, cwd=ROOT):
    t0 = time.perf_counter()
    print(f"\n== {title}\n   $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)
    print(f"   ok ({time.perf_counter() - t0:.0f}s)", flush=True)


def main():
    args = set(sys.argv[1:])
    pb = (ROOT / "parsebench" / "table.jsonl").exists() and "--no-parsebench" not in args
    step("self-check: parser", PY, "src/parser.py")
    step("self-check: ocr", PY, "src/ocr.py")
    step("self-check: evaluate", PY, "-I", "eval/evaluate.py")
    step("self-check: ParseBench converter", PY, "-I", "eval/parsebench_to_gold.py")
    if "--rebuild-gold" in args:
        step("gold: rebuild", PY, "gold/build_gold.py")
    step("gold: check against its PDFs", PY, "gold/check_gold.py")
    step("gold: predict", PY, "eval/predict.py", "gold/json", "preds/final")
    if pb:
        step("ParseBench: convert ground truth", PY, "-I", "eval/parsebench_to_gold.py", "parsebench/table.jsonl", "eval/parsebench_gold")
        step("ParseBench: dev/test split", PY, "-I", "eval/split_parsebench.py")
        step("ParseBench: predict (OCR pages ~2 min each when not cached)", PY, "eval/predict.py", "eval/parsebench_gold", "preds/final_pb")
        step("score: gold + ParseBench dev/test -> docs/scores.*", PY, "-I", "eval/scoreboard.py", "final", "preds/final", "preds/final_pb",
             "--note", "eval/run_all.py")
    else:
        print("\n(ParseBench not found or --no-parsebench: scoring gold only)")
        step("score: gold", PY, "-I", "eval/evaluate.py", "gold/json", "preds/final", "--out", "preds/final_report.json")
    if (ROOT / "ocr_bench" / "out").exists() or "--ocr-rerun" in args:
        if "--ocr-rerun" in args:
            for e in ("tesseract", "vision", "vision-auto", "lighton-1b"):
                step(f"OCR bench: run {e}", PY, "-I", "ocr_bench/run_engine.py", e)
        step("OCR bench: score all engines and pairs", PY, "score.py", cwd=ROOT / "ocr_bench")
        step("OCR bench: dev/test split, old vs new scorer", PY, "compare_scorers.py", cwd=ROOT / "ocr_bench")
    else:
        print("\n(ocr_bench/out not found: OCR bench scores skipped; run with --ocr-rerun to produce them)")
    print("\nall done. See docs/scores.html")


if __name__ == "__main__":
    main()
