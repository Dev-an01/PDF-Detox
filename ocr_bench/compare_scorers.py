"""OCR bench with a dev/test split, old scorer vs new scorer, per engine.

    python ocr_bench/compare_scorers.py      -> table + ocr_bench/sets/split.json

Split: ParseBench pages keep the ParseBench dev/test split (eval/parsebench_split.json, by source
document); gold OCR docs and the real sample are split by source document, seed 7, ~60/40, with the
gold docs inspected while building the scorer forced into dev. Scored on the two sets that have
positions (gold_ocr, real_render); pb_notext has no positions (order-only) and is left out.
Variants: v1 = char-level reading, no new rules (the original scorer)
          v2 = run-level reading, no new rules (measurement fix only)
          v3 = v2 + new flag rules (LightOn empty/cut guard, sign-uncertain dash, collapsed boxes)
"""
import json
import random
import re
from pathlib import Path

import score

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
INSPECTED_GOLD = {"syn_ja_bill_ruled_wrapped_ocr", "syn_en_bill_ruled_wrapped_ocr", "syn_ja_bs_cross_page_ocr",
                  "syn_en_bs_cross_page_ocr", "syn_en_bs_striped_hierarchy_ocr"}


def split(items):
    pb = json.loads((ROOT / "eval" / "parsebench_split.json").read_text())
    part = {i: "dev" for i in pb["dev"]} | {i: "test" for i in pb["test"]}
    out, rest = {}, set()
    for it in items:
        src = re.sub(r"_p\d+$", "", it["id"])
        if it["id"] in part:
            out[it["id"]] = part[it["id"]]
        elif src in INSPECTED_GOLD:
            out[it["id"]] = "dev"
        else:
            rest.add(src)
    rest = sorted(rest)
    random.Random(7).shuffle(rest)
    dev_src = set(rest[: round(0.6 * len(rest))])
    for it in items:
        if it["id"] not in out:
            out[it["id"]] = "dev" if re.sub(r"_p\d+$", "", it["id"]) in dev_src else "test"
    return out


def run(items, systems):
    res = {}
    for name in systems:
        parts = name.split("+")
        for part in ("dev", "test"):
            recs = []
            for it in items[part]:
                outs = [score.load(p, it) for p in parts]
                if any(o is None for o in outs):
                    continue
                if parts[0] in score.BOXED:
                    chars, runs = score.boxed_chars(outs[0], parts[0])
                    tau = score.TAU[parts[0]]
                    sk = {k for r in runs if r["sign_uncertain"] for k in r["chars"]}
                    r = score.score_boxed(it, chars, runs, lambda k, c=chars, t=tau, sk=sk: k in sk or (c[k]["conf"] is not None and c[k]["conf"] < t))
                elif len(parts) == 1:
                    r = score.guarded(score.score_lighton(it, score.lighton_runs(outs[0]), it["width"]), score.lighton_guard(outs[0]))
                else:
                    chars, runs = score.boxed_chars(outs[1], parts[1])
                    r = score.guarded(score.score_pair(it, chars, runs, score.lighton_runs(outs[0]), it["width"]), score.lighton_guard(outs[0]))
                recs.append(r)
            flat = [x for r in recs for x in r]
            n = len(flat) or 1
            res[(name, part)] = {"pages": len(recs), "numbers": len(flat), "exact": 100 * sum(x["ok"] for x in flat) / n,
                                 "unflagged": sum(not x["ok"] and not x["flagged"] for x in flat),
                                 "review": 100 * sum(x["flagged"] for x in flat) / n}
    return res


def main():
    items = [i for i in json.loads((BENCH / "sets" / "manifest.json").read_text()) if not i["ordered_only"]]
    sp = split(items)
    (BENCH / "sets" / "split.json").write_text(json.dumps(sp, indent=1))
    by = {"dev": [i for i in items if sp[i["id"]] == "dev"], "test": [i for i in items if sp[i["id"]] == "test"]}
    print(f"split: dev {len(by['dev'])} pages / {sum(len(i['gt']) for i in by['dev'])} numbers, "
          f"test {len(by['test'])} pages / {sum(len(i['gt']) for i in by['test'])} numbers")
    systems = list(score.BOXED) + list(score.LIGHTON) + [f"{l}+{b}" for l in score.LIGHTON for b in score.BOXED]
    variants = {}
    for v, (run_level, rules) in {"v1": (False, False), "v2": (True, False), "v3": (True, True)}.items():
        score.RUN_LEVEL, score.RULES = run_level, rules
        variants[v] = run(by, systems)
    print(f"\n{'system':24s} | {'UNFLAGGED dev v1/v2/v3':>24s} | {'UNFLAGGED test v1/v2/v3':>25s} | {'exact% test v1/v2':>17s} | {'review% test v2/v3':>18s}")
    for name in systems:
        u = lambda part: "/".join(str(variants[v][(name, part)]["unflagged"]) for v in ("v1", "v2", "v3"))
        print(f"{name:24s} | {u('dev'):>24s} | {u('test'):>25s} | "
              f"{variants['v1'][(name, 'test')]['exact']:7.1f}/{variants['v2'][(name, 'test')]['exact']:<9.1f} | "
              f"{variants['v2'][(name, 'test')]['review']:7.1f}/{variants['v3'][(name, 'test')]['review']:<9.1f}")
    json.dump({f"{v}|{k[0]}|{k[1]}": x for v, r in variants.items() for k, x in r.items()},
              open(BENCH / "results_split.json", "w"), indent=1)


if __name__ == "__main__":
    main()
