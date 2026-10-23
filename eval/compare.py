"""Per-page comparison of two prediction dirs on ParseBench dev and test.

    python eval/compare.py BEFORE_PRED_DIR AFTER_PRED_DIR

For each split: pages better / worse / same on exact numbers, on shifts and on unflagged; how
much of the net change comes from the top 1 and top 2 pages; and the keep rule (test not worse
on unflagged and shifts). Dev page names are printed; test is counts only (never inspect test).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import evaluate as ev  # noqa: E402

METRICS = [("nums_exact", +1, "exact numbers"), ("col_shifts", -1, "shifts"), ("num_errors_unflagged", -1, "unflagged")]


def per_page(pred_dir, stems):
    out = {}
    for s in stems:
        gold = json.loads((ROOT / "eval" / "parsebench_gold" / f"{s}.json").read_text())
        pf = Path(pred_dir) / f"{s}.json"
        pred = json.loads(pf.read_text()) if pf.exists() else {"tables": []}
        out[s] = ev.eval_doc(gold, pred, with_teds=False)
    return out


def compare(before, after, stems, show_names):
    a, b = per_page(before, stems), per_page(after, stems)
    rep = {}
    for key, sign, label in METRICS:
        d = {s: sign * (b[s].get(key, 0) - a[s].get(key, 0)) for s in stems}  # > 0 means better
        net = sum(d.values())
        top = sorted(d.items(), key=lambda kv: -abs(kv[1]))
        share = lambda n: round(sum(v for _, v in top[:n]) / net, 2) if net else None
        rep[label] = {"better": sum(v > 0 for v in d.values()), "worse": sum(v < 0 for v in d.values()),
                      "same": sum(v == 0 for v in d.values()), "net_gain": net,
                      "total_before": sum(a[s].get(key, 0) for s in stems), "total_after": sum(b[s].get(key, 0) for s in stems),
                      "top1_share_of_net": share(1), "top2_share_of_net": share(2)}
        if show_names:
            rep[label]["top_pages"] = [(s, v) for s, v in top[:3]]
    return rep


if __name__ == "__main__":
    split = json.loads((ROOT / "eval" / "parsebench_split.json").read_text())
    res = {part: compare(sys.argv[1], sys.argv[2], split[part], show_names=(part == "dev")) for part in ("dev", "test")}
    for part, rep in res.items():
        print(f"== {part} ({len(split[part])} pages)")
        for label, r in rep.items():
            line = (f"  {label:14s} better {r['better']:3d}  worse {r['worse']:3d}  same {r['same']:3d}  "
                    f"total {r['total_before']} -> {r['total_after']}  top1 {r['top1_share_of_net']}  top2 {r['top2_share_of_net']}")
            print(line)
            if "top_pages" in r:
                print("      top dev pages:", r["top_pages"])
    t = res["test"]
    ok = t["unflagged"]["total_after"] <= t["unflagged"]["total_before"] and t["shifts"]["total_after"] <= t["shifts"]["total_before"]
    print(f"KEEP RULE (test unflagged and shifts not worse): {'PASS' if ok else 'FAIL'}")
