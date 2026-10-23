# PDF-Detox

Turns PDF pages (financial statements, filings, bills, datasheets; Japanese and English) into
structured output where **nothing on the page is lost**. Every table becomes cells that keep their
exact printed text, numbers get an exact value, and every heading, caption, unit line, text line,
footnote and page number is kept in reading order. Every number that the parser cannot verify is
**flagged for review instead of guessed**.

The goal is not the highest average score. It is: **no wrong number reaches the output unflagged.**

## What "correct" means here

Every number in a ground-truth table ends up in exactly one of three tiers:

| tier | meaning |
|---|---|
| **correct** | the parsed value equals the printed number exactly (no tolerance) |
| **wrong, but flagged** | the parser got it wrong, but marked the cell, table or page for review: a person will see it |
| **wrong and NOT flagged** | the dangerous case: a wrong number that looks verified. **Target: 0** |

"Sent to review" is the share of all numbers that are flagged, right or wrong. It is the cost
paid for safety, and it is reported next to every score.

ParseBench's ground truth sometimes splits cells differently from us (`EUR 20,000,000` vs
`EUR | 20,000,000`). So ParseBench results are reported two ways, always side by side:
**cell by cell** (strict) and **real** (not counting numbers whose whole row text matches an output
row on the page). The real figure is lenient, so the truth lies between the two.

## Results

Apple M5, 16 GB, macOS 26.6.1, Python 3.12.13. All numbers are shares of the numbers in the ground-truth tables.

### Gold set (23 docs, 1,241 numbers)

| set | numbers correct | wrong, flagged | **wrong, NOT flagged** | sent to review | table F1 | cell exact |
|---|---|---|---|---|---|---|
| all 23 docs | **100%** | 0% | **0** | 22.9% | 1.0 | 97.7% |
| the real sample page (ja) | 100% | 0% | **0** | 0% | 1.0 | 100% |
| 15 text-layer docs | 100% | 0% | **0** | 0% | 1.0 | 100% |
| 8 image-only docs (OCR) | 100% | 0% | **0** | 64.4% | 1.0 | 93.6% |
| Japanese docs | 100% | 0% | **0** | 34.0% | 1.0 | 96.7% |

The gold set is 1 real Japanese page plus 22 synthetic docs, whose ground truth comes from the
generator rather than from parsing (`gold/`): balance sheets with indent hierarchy, multi-row
headers, colspan/rowspan, bills with wrapped labels, datasheets, two tables on a page, one table
across two pages, and image-only copies.

### ParseBench (503 English pages), dev 303 / test 200 pages

The split is by source document, fixed seed. Tuning was done on dev only.

| | numbers correct | wrong (cell by cell / **real**) | wrong & NOT flagged (cell by cell / **real**) | sent to review | column shifts |
|---|---|---|---|---|---|
| dev | 53.3% | 46.7% / **27.6%** | 20.7% / **14.5%** | 48.9% | 2,095 |
| test | 48.5% | 51.5% / **21.6%** | 16.4% / **7.1%** | 64.5% | 1,891 |

**Zero unflagged is not reached on ParseBench.** Most remaining unflagged errors are tables that
are segmented differently from the ground truth (merged, split, or side-by-side), and rows that
pair differently. The default page check (`PARSER_PAGE_CHECKS`, below) trades review for safety:

| page check setting | dev unflagged (real) | dev review | test unflagged (real) | test review |
|---|---|---|---|---|
| S0 `stray` | 31.5% (20.0%) | 28.7% | 27.6% (13.3%) | 32.7% |
| S1 `stray,count_lines` | 22.5% (15.6%) | 46.0% | 18.5% (9.0%) | 59.1% |
| **S2 `stray,count_lines,mid_header` (default)** | 21.7% (15.6%) | 48.1% | 16.4% (7.1%) | 64.5% |
| S5 `+ escalate` | 20.2% (14.5%) | 60.3% | 12.0% (6.0%) | 80.6% |

S2 was picked as the lowest unflagged with review under 50% on dev. On test, no setting except
S0 stays under 50% review. Switch with `PARSER_PAGE_CHECKS=stray` if review volume matters more
than safety. Scores per milestone: `docs/scores.md`, and the dashboard `docs/scores.html`.

**Comparison with published baselines.** ParseBench publishes a Tables score (GTRM) for other
parsers, for example PyMuPDF4LLM 72.0, Docling 66.4 and LightOnOCR-3-1B 84.8. **This project
has not been run through the official ParseBench scorer** (that needs a `parse-bench` provider,
not written yet), so those numbers are **not directly comparable** with the exact-number metrics
above. Running the official scorer is the first item under next steps.

### OCR engines (fallback only), `ocr_bench/`

These are scored on pages that have no text layer, or whose text layer is set aside: 10 gold OCR
pages, plus 40 real pages rendered to images where the text layer is the answer (dev 31 / test 19
pages, split by source document). The OCR engine runs only when a page's text layer is missing,
garbled or fails the audit.

| system | dev: wrong & NOT flagged | dev: review | test: wrong & NOT flagged | test: review | s/page |
|---|---|---|---|---|---|
| Tesseract 5 (jpn+eng) | 382 | 9.0% | 377 | 27.8% | 1–9 |
| Apple Vision (ja-first) | 1,304 | 30.6% | 2,395 | 30.4% | 0.4–0.8 |
| Apple Vision (auto language) | 1,609 | 2.3% | 2,990 | 3.6% | 0.4 |
| LightOnOCR-3-1B alone | 14 | 56.0% | 12 | 75.8% | 70–136 |
| **LightOnOCR-3-1B + Tesseract (default)** | **0** | 61.6% | **0** | 80.4% | 72–145 |
| LightOnOCR-3-1B + Vision | 3 | 67.6% | 0 | 79.8% | 71–136 |

In the pair, a number is accepted only when both engines read the same number in the same block,
in the same order. On real rendered pages **the pair sends ~74–80% of numbers to review and takes
about 130–145 s per page**, which is why it is a fallback, never a second pass on good pages.
LightOnOCR alone going from 2,807 to 12 unflagged on test comes from whole-page flags (empty or
cut-off output), which costs review: 0% → 75.8%.

**Honesty notes on the OCR numbers:**
- The OCR scorer was changed once, from reading characters to reading whole numbers. This fixed
  boxes that were slightly off, and both versions are shown in `ocr_bench/compare_scorers.py`.
- The flag rules are what bring test to 0:
  - **The "LightOnOCR output empty or cut" guard was first seen on a test page** (a dense VIN
    page), and on test it is the guard that gives 0.
  - The dash and thin-box rules alone leave test at 250.
  - So the test 0 is not a fully clean held-out result.

**Time per page (M5):**
- text-layer page: 0.1 s, or 0.7 s with the Vision spot check;
- OCR page: LightOnOCR 70–263 s (median 78 s) plus Tesseract 1–9 s.

The full ParseBench run (503 pages, 19 of them OCR) took 37 min uncached. OCR results are cached
in `.ocr_cache/`.

## Setup

Requirements:
- macOS on Apple silicon (tested on an M5 with 16 GB RAM). LightOnOCR runs on the GPU through
  MPS: peak about 6 GB process + 8 GB MPS on an OCR page.
- Disk: about 1 GB for the venv, 3.8 GB for the LightOnOCR-3-1B weights, 55 MB for Tesseract
  data, and 565 MB for ParseBench if you run the benchmarks.
- Python 3.12, [uv](https://docs.astral.sh/uv/), Homebrew, and the Xcode Command Line Tools
  (`swiftc`, for the Apple Vision helper).

```bash
brew install tesseract
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt   # pins transformers==5.5.4 (5.20 mis-loads LightOnOCR-3-1B)

# Tesseract models (tessdata_best) + hOCR configs, into models/ (not in git)
mkdir -p models/tessdata
for l in jpn jpn_vert eng osd; do
  curl -L -o models/tessdata/$l.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata
done
cp -R "$(brew --prefix)/share/tessdata/configs" "$(brew --prefix)/share/tessdata/tessconfigs" models/tessdata/

# LightOnOCR-3-1B (~3.8 GB) downloads itself on the first OCR page; to fetch it now:
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('lightonai/LightOnOCR-3-1B')"
```

The Apple Vision helper (`bin/vision_ocr`) is compiled from `src/vision_ocr.swift` on first use.

**ParseBench** (only for the benchmarks) is Apache-2.0, from Kaggle ("ParseBench: Document OCR
for AI Agents") or Hugging Face. Put it in `./parsebench/`, so that `parsebench/table.jsonl` and
`parsebench/docs/table/*.pdf` exist:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('llamaindex/ParseBench', repo_type='dataset', local_dir='parsebench')"
```

**Never commit datasets, model weights or caches.** `parsebench/`, `models/`, `bin/`,
`.ocr_cache/`, `ocr_bench/out/` and `preds/` are in `.gitignore`.

## Usage

**Terminal app**

```bash
.venv/bin/python app.py
```
```
1  parse a single file
2  parse several files (a folder or paths)
3  exit
```
- **Option 1:** shows each page in the terminal in reading order (headings, captions, unit
  lines, text, tables, notes, page number) and saves `<name>_parsed.md`, `.html` or `.json` next
  to the PDF. It asks for the format; md is the default.
- **Option 2:** takes a folder or several paths, asks for the format once, saves each file, and
  prints a summary per file: pages, tables, flagged cells, OCR pages, time, saved path.
- **Flagged cells** are marked ⚠ in the terminal, md and html, with a flag list at the end, and
  each page says whether it used the text layer or OCR.
- **OCR pages** show their progress, since they take about 2 minutes each.
- **A bad path, a non-PDF or a broken PDF** shows the error and never stops the other files.

**Command line**

```bash
.venv/bin/python app.py samples/balance_sheet_after_correction.pdf               # -> ..._parsed.md
.venv/bin/python app.py samples/balance_sheet_after_correction.pdf --format html
.venv/bin/python app.py some/folder other.pdf --format json                     # full schema, see docs/design.md §2
.venv/bin/python src/parser.py file.pdf --out out_dir/                          # parser only, JSON
```

**What is saved.** Nothing on the page is dropped:

- **Pages:** each page has `blocks` in reading order. A block's `type` is one of `heading` (with
  `level`), `caption`, `unit`, `text`, `footnote`, `furniture` (page numbers), `figure` (OCR
  only), or `table` (with `table_id`; `continued: true` on later pages of a table that spans
  pages).
- **Block details:** every block has its `text`, `bbox` and `source` (`text_layer` or `ocr`). Text
  read by OCR is marked `ocr_text_unverified`, since only table numbers are double-checked.
- **Tables:**
  - `captions` holds all the lines stacked directly above a table (for the sample:
    `（訂正後）`, `３．連結財務諸表及び主な注記`, `（１）連結貸借対照表`, `（単位：千円）`).
  - `title` (the nearest caption line) and `unit` are kept as well.
- **Nothing-lost check:** every character on a text-layer page must end up in a table cell or a
  text line. Otherwise the page is flagged `text_not_captured`; this happens on 0 of 530 pages.
- **Output formats:** md and html show the whole page in that order; json has the full schema.

**Settings (environment variables)**

| variable | default | meaning |
|---|---|---|
| `PARSER_OCR` | `lighton+tesseract` | OCR fallback engine: `lighton+tesseract`, `tesseract` (fast, **not** zero-unflagged), `none` |
| `PARSER_OCR_STRUCTURE` | `lighton` | on OCR pages, who gives the table structure: LightOnOCR's HTML (`lighton`) or Tesseract word geometry (`tesseract`) |
| `PARSER_PAGE_CHECKS` | `stray,count_lines,mid_header` | page checks (S2). `stray` alone = S0; add `escalate` for S5 |
| `PARSER_SPOT_CHECK` | `1` | Apple Vision spot check of text layers (flags only, never replaces a text layer) |

## How it works

```
PDF page
  │
  ├─ text layer ──► audit: text present? broken glyphs mixed into words? invisible (old OCR) layer?
  │                   │ok                                   │failed
  │                   ▼                                     ▼
  │              characters + boxes                  OCR fallback (page or image region):
  │              (Vision spot check: flags only)     LightOnOCR-3-1B table + Tesseract words,
  │                                                  a number is kept only if both agree
  ▼
regions ── ruled boxes (PyMuPDF line finder, edges only) + borderless runs; captions & unit line; rotation
  ▼
structure ── candidate cells (1 em gaps) → rows → columns (ruled edges, else geometry bands)
             → header rows + spans → wrapped labels → indent hierarchy → cross-page join
  ▼
values ── exact Decimal strings: △/▲/(1,234) negatives, －/— blanks, full width, footnote marks;
          unit + scale recorded, never applied
  ▼
validation ── totals with rounding tolerance (net 純額 rows), every character in one cell,
              column type, column conflicts, cross-page confirmation, unit, OCR agreement,
              page checks → flags (verified or flagged, nothing in between)
  ▼
page blocks ── every character not in a table cell → text lines → heading (numbering / size) · caption
               (stacked above a table) · unit · text · footnote · page furniture; all in reading order
  ▼
output ── JSON (pages[].blocks, tables[].cells: raw_text, value, bbox, flags) · Markdown · HTML
```

```
app.py                  terminal app (menu + CLI)
src/
  parser.py             the parser: text layer → regions → structure → values → validation
  ocr.py                OCR fallback (LightOnOCR + Tesseract check), cache in .ocr_cache/
  engines.py            shared engine code: LightOnOCR loader, tool paths, number definition
  vision_ocr.swift      Apple Vision helper (spot check), compiled to bin/ on first use
eval/
  evaluate.py           metrics: tables, TEDS, cell exact, numeric exact, shifts, unflagged, review, split-only
  predict.py            run the parser on every gold doc
  scoreboard.py         score a milestone → docs/scores.md, docs/scores.jsonl, docs/scores.html
  dashboard.py          docs/scores.html (plain HTML + SVG)
  compare.py            per-page better/worse/same on ParseBench dev and test + keep rule
  run_all.py            rerun everything from scratch
  review.py             visual check: page vs parsed table; 50 random numbers
  parsebench_to_gold.py, split_parsebench.py, parsebench_split.json   ParseBench ground truth + dev/test split
gold/                   golden set: build_gold.py, check_gold.py, json/ (ground truth), pdf/
samples/                the real sample page
ocr_bench/              OCR engine benchmark: make_sets, run_engine, score, compare_scorers, results*.json
docs/                   design.md (full design and every decision), scores.*
```

`docs/design.md` has the full design, every change made against the first draft, and why.

## Run the evaluations

```bash
.venv/bin/python eval/run_all.py                 # self-checks, gold, ParseBench (if present), scoreboard, OCR bench scores
.venv/bin/python eval/run_all.py --no-parsebench # gold only, no dataset needed
```

Single steps:
- `eval/evaluate.py GOLD_DIR PRED_DIR`: running it with no arguments runs its self-check.
- `eval/compare.py BEFORE AFTER`: page counts and the keep rule.
- `gold/check_gold.py`: checks the gold against its own PDFs.

## Verify it yourself

```bash
# one page: image left, parsed tables right; flagged cells amber, wrong vs gold red; click a cell to see its crop
.venv/bin/python eval/review.py samples/balance_sheet_after_correction.pdf --gold gold/json/real_ja_balance_sheet.json
.venv/bin/python eval/review.py gold/pdf/syn_en_bill_ruled_wrapped_ocr.pdf --gold gold/json/syn_en_bill_ruled_wrapped_ocr.json

# 50 random numbers from ParseBench test (seed 7): crop of each number next to its parsed value, tick right/wrong
.venv/bin/python eval/review.py --sample50 --split test --seed 7 --pred-dir preds/final_pb
```

The 50-number check is the quickest way to judge the parser without trusting any metric. Each
crop must match its parsed text character for character, and the value must be the printed
number (sign kept, no scale applied).

## Known limits

- **Only 1 real Japanese page.** The rest of the Japanese gold set is synthetic. Real filings
  from EDINET (XBRL as the answer key) are the main next step. They need an EDINET API key
  (`EDINET_API_KEY`).
- **Zero unflagged is reached on gold, not on ParseBench:** 14.5% (dev) / 7.1% (test) real, and
  20.7% / 16.4% cell by cell.
- **High review rates:** ParseBench dev 48.9% and test 64.5%; OCR pages 64–80%.
- **Not run through the official ParseBench scorer (GTRM),** so the published baselines are not
  directly comparable.
- **Invisible text is read as numbers.**
  - A test page (found with `eval/review.py`) has 148 white `0.00` placeholders that the parser
    outputs as values.
  - Proposed fix: drop text whose colour matches the fill behind it, checked on dev. Not done,
    because it was found on a test page.
- **Side-by-side tables:**
  - Borderless ones are not split.
  - OCR tables are split only when the header repeats exactly. That rule came from one dev page.
- **Fixes with a cost:** the icon-font audit fix (keeping SF Symbols / Wingdings pages on their
  text layer) lost 16 exact numbers on one test page.
- **Totals can't catch a wrong last digit:** each printed line is rounded, so the tolerance is 1
  unit per addend. Text-layer digits are exact copies; OCR digits rely on the two-engine agreement.
- **Not handled:**
  - leading-dot decimals (`.125`) are not read as numbers;
  - space-grouped thousands (`1 234`) would be split;
  - ja/en number formats only.
- **OCR is slow** (~2 min per page) and needs the 3.8 GB model. LightOnOCR-3-0.8B is
  impractical on MPS (24 min/page).
- **Not built:** CSV output (JSON is the machine format), and the region-level ink coverage check
  in the text-layer audit.
- **Page text blocks:**
  - Reading order is top to bottom, then left to right, so a page with two text columns
    interleaves their lines.
  - Headings are found by numbering (`３．`, `（１）`, `第1章`), short bracketed labels
    (`（訂正後）`) and font size: a rule, not a model.
  - Text read by OCR is kept but marked unverified.
  - Text blocks are not scored by any metric, because the benchmarks have table ground truth only.

## Next steps

1. Run the official ParseBench scorer (write a `parse-bench` provider) to compare with published baselines.
2. Ten real EDINET filings with XBRL as the answer key (BS/PL/CF, 報告式/勘定式, 千円/百万円), reported separately.
3. Drop invisible text (fill-colour check, tuned on dev).
4. Reduce ParseBench's unflagged share from table segmentation: split side-by-side borderless tables and stacked tables with the same rules.
5. Lower review on OCR pages: a second Japanese-capable reader, so correct numbers are confirmed more often.
