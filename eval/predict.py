"""Run the parser on every gold doc's PDF; save predictions under the gold doc's name.

    python eval/predict.py GOLD_DIR PRED_DIR

evaluate.py pairs gold and pred by filename, so each prediction is written as <gold doc_id>.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import parser  # noqa: E402  (src/parser.py)

if __name__ == "__main__":
    gold_dir, pred_dir = Path(sys.argv[1]), Path(sys.argv[2])
    pred_dir.mkdir(parents=True, exist_ok=True)
    for g in sorted(gold_dir.glob("*.json")):
        gold = json.loads(g.read_text())
        pred = parser.parse(ROOT / gold["source_file"], doc_id=gold["doc_id"])
        (pred_dir / g.name).write_text(json.dumps(pred, ensure_ascii=False, indent=1))
    print(f"predicted {len(list(gold_dir.glob('*.json')))} docs -> {pred_dir}")
