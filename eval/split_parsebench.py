"""Split ParseBench into dev (~60%) and test (~40%) by SOURCE DOCUMENT, fixed seed.

    python eval/split_parsebench.py   -> eval/parsebench_split.json, eval/parsebench_dev/, eval/parsebench_test/

Pages of one filing are near-duplicates (BRWS-134565917 pages 853 and 1210 have identical numbers),
so a page-level split would leak. Tune only on dev; report both; never inspect test pages.
Source documents already inspected while tuning m2 are forced into dev so test stays unseen.
"""
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED, DEV_SHARE = 7, 0.6
# pages looked at while building m2 (fixes or printed shapes) -> their whole source doc is dev
SEEN = ["SERFF_CA_random_pages 1_page1377", "1 timetable (1)_page7", "203924_fb04e77929bf4bbc93dbd659653a4f01_page10",
        "SERFF_CA_random_pages 1_page2443", "BRWS-134565917_page1210", "BRWS-134565917_page853",
        "KEY.2014.page_103.pdf_98246_page1", "FBLB-134215544_page98", "PrintableRulesSection_page17",
        "20250401_Axos_Bank_SFR_Correspondent_Rate_Sheet_page1", "BRWS-134565917_page851",
        "20230125-weather-climate-catastrophe-insight_page9", "222876fb_page21"]


def source(stem):
    return re.sub(r"_page\d+$", "", stem)


def main():
    stems = sorted(p.stem for p in (ROOT / "parsebench_gold").glob("*.json"))
    groups = {}
    for s in stems:
        groups.setdefault(source(s), []).append(s)
    seen_src = {source(s) for s in SEEN}
    rest = sorted(set(groups) - seen_src)
    random.Random(SEED).shuffle(rest)
    dev, n_dev = set(seen_src & set(groups)), sum(len(groups[g]) for g in seen_src & set(groups))
    for g in rest:  # fill dev up to ~60% of pages, the rest is test
        if n_dev < DEV_SHARE * len(stems):
            dev.add(g)
            n_dev += len(groups[g])
    split = {"seed": SEED, "by": "source document",
             "dev": sorted(s for g in dev for s in groups[g]),
             "test": sorted(s for g in set(groups) - dev for s in groups[g])}
    assert not set(split["dev"]) & set(split["test"]) and len(split["dev"]) + len(split["test"]) == len(stems)
    assert not {source(s) for s in split["dev"]} & {source(s) for s in split["test"]}
    (ROOT / "parsebench_split.json").write_text(json.dumps(split, indent=1))
    for part in ("dev", "test"):
        d = ROOT / f"parsebench_{part}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for s in split[part]:
            shutil.copy(ROOT / "parsebench_gold" / f"{s}.json", d / f"{s}.json")
    print(f"dev {len(split['dev'])} pages / {len(dev)} docs, test {len(split['test'])} pages / "
          f"{len(groups) - len(dev)} docs")


if __name__ == "__main__":
    main()
