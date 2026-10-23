# Table Parser — Design

Goal: turn PDF pages (filings, datasheets, bills; Japanese and English) into structured tables
that match the page exactly, with no changed digits, no column shifts and no dropped rows.
Anything the parser cannot verify is flagged for review, never guessed.

**Constraints (decided, §9):** offline, runs on one Apple-silicon Mac (M5, 16 GB); no CUDA, but
MPS/MLX are allowed. Input is born-digital PDFs (no photos or scans). OCR is a **fallback only**,
for a page or region whose text layer is missing, garbled or fails the audit. The engine is a
setting, chosen by the OCR benchmark (`ocr_bench/`). Numbers are kept as printed and never
rescaled. The target is **zero wrong numbers reaching output unflagged**.

## 0. What the sample tells us (`balance_sheet_after_correction.pdf`)

| Check | Result |
|---|---|
| Pages | 1 page, A4 (595×842 pt), Japanese 連結貸借対照表 (consolidated balance sheet), labeled （訂正後） (corrected version) |
| Text layer | Present, 1002 chars. No U+FFFD, private-use or control chars; all text rendered visibly (no render mode 3); nothing outside the page |
| Layer vs. render | Matches the rendered page row by row (checked visually at 110 dpi and with word bboxes) |
| Ruling | **No vertical lines.** 263 filled rects (alternating row stripes) and 78 horizontal rules (subtotal underlines). A lattice-only (line-based) table detector would fail here. |
| Columns | Label column plus 2 numeric columns, right-aligned at x≈398 and x≈523. Header has 2 lines per column (`前連結会計年度` / `(2025年３月31日)`). |
| Row hierarchy | Encoded **only by indentation**: label x0 ∈ {62, 75, 88, 101, 114}, which is 5 levels at a ~13 pt step |
| Number formats | `2,002,429`; negative `△2,979`; zero-negative `△0`; blank `－` (U+FF0D); mixed full-width digits in the header (`３月31日`) |
| Scale | `（単位：千円）` means values are in thousands of JPY |
| Arithmetic | Totals are **off by up to 3** against the sum of their children (e.g. 流動資産合計 5,863,684 vs. sum 5,863,681) because each line is rounded to 千円. Checks need a rounding tolerance. |
| Page furniture | Page number `- 4 -`, section titles above the table |
| Split spans | `△` and its digits are **separate text spans** (`'△'` \| `'2,979'`). A parser that takes one span per cell silently turns −2,979 into +2,979. The golden-set sum check caught this, so cells must be built from **all** spans that fall in the cell geometry. |
| Net rows | `建物及び構築物（純額）` = gross + `減価償却累計額`. Totals roll up the *net* rows, not the gross ones, so arithmetic checks need the net-row pattern (`（純額）`, `net`), not just "parent = Σ children". |

The sample is a "clean" digital PDF, so the hard parts are structural (borderless table,
indent hierarchy, multi-line headers), not OCR.

## 1. Pipeline

```
PDF ─► per page: text-layer audit ──reliable──► words+bboxes (exact chars)
                         └─unreliable─► render 300 dpi ─► OCR ─► words+bboxes+conf
      ─► layout: table regions (+ title / unit / footnote blocks near each region)
      ─► structure: rows ─► columns ─► header rows ─► cells (spans) ─► row hierarchy
      ─► cross-page stitching
      ─► normalization (additive; raw text kept)
      ─► validation + confidence ─► JSON / CSV (+ review queue)
```

**1.1 Text-layer audit (per page; fall back per region when only part of a page fails).** The
layer is used only if all of these pass:
- Glyph mapping is sane: no U+FFFD, `(cid:N)`, private-use or control chars, and the ratio of
  printable characters is high.
- Coverage: render the page and find ink connected components. Ink with no text bbox over it means
  text that exists only as an image (scanned stamps, pasted images).
- Invisible layer (render mode 3) means a previous OCR pass. Treat it as OCR-quality and re-verify it.
- Spot check: OCR a sample of word crops and compare against layer text. CER above a threshold
  marks the layer unreliable (catches broken ToUnicode maps that pass the first check).

**1.2 OCR path (as built in m6; `src/ocr.py`, setting `PARSER_OCR`).** OCR is a **fallback only**.
- **When it runs:**
  - A **page** goes to OCR when its text layer fails the hard audit: no text (< 20 glyphs), > 1%
    U+FFFD / private-use / control chars (a broken ToUnicode map), or > 50% invisible text
    (render mode 3, an old OCR layer).
  - A **region** goes to OCR when it is an image covering ≥ 3% of the page, with < 5 text chars on
    top of it, where Tesseract finds ≥ 5 numbers.
- **Engine, chosen by the OCR bench (`ocr_bench/`; gold OCR 491 numbers, 40 real pages rendered to
  images 8,983, 6 ParseBench no-text pages 474):** **LightOnOCR-3-1B + Tesseract (jpn+eng).** It
  is the only system with **0 unflagged wrong numbers** on both position-scored sets, and 0
  accepted-but-wrong numbers on the no-text pages.

  | system | gold OCR unflagged | real-render unflagged | review (gold / real) | s/page |
  |---|---|---|---|---|
  | Tesseract | 26 | 733 | 11% / 19% | 1.3 / 8.6 |
  | Apple Vision (ja-first) | 11 | 3,688 | 47% / 30% | 0.4 / 0.8 |
  | Apple Vision (auto language) | 21 | 4,578 | 9% / 3% | 0.4 / 0.4 |
  | LightOnOCR-3-1B alone | 0 | 26 | 0% / 69% | 70 / 136 |
  | **LightOnOCR-3-1B + Tesseract** | **0** | **0** | 12% / 74% | 72 / 144 |
  | LightOnOCR-3-1B + Vision | 0 | 3 | 14% / 77% | 71 / 136 |

  LightOnOCR-3-0.8B was not run: 24 min/page on MPS, because transformers falls back to a slow
  pure-PyTorch path for its linear attention.
- **Cost of the pair: fallback pages only.** On real pages rendered to images, **the OCR pair sends
  74% of numbers to review and takes about 130–145 s per page**: LightOnOCR-1B 70–263 s,
  median 78 s per uncached page, plus Tesseract 1–9 s. A text-layer page takes 0.1 s, or 0.7 s
  with the Vision spot check. So OCR runs only where the text layer fails, never as a second
  pass on good pages.
- **OCR bench dev/test split** (`ocr_bench/compare_scorers.py`, `ocr_bench/sets/split.json`):
  ParseBench pages keep the ParseBench split; gold OCR docs are split by source document with
  seed 7, and the docs inspected while building the scorer are forced into dev. Dev is 31 pages /
  4,831 numbers, test 19 pages / 4,643 numbers. Unflagged wrong numbers, v1 → v2 → v3:

  | system | dev | test |
  |---|---|---|
  | Tesseract | 424 → 383 → 382 | 289 → 377 → 377 |
  | Vision (ja-first) | 1371 → 1305 → 1304 | 2234 → 2395 → 2395 |
  | Vision (auto) | 2255 → 1655 → 1609 | 3135 → 3008 → 2990 |
  | LightOnOCR-1B | 1111 → 1111 → 14 | 2807 → 2807 → 12 |
  | **LightOnOCR-1B + Tesseract** | 45 → 5 → **0** | 228 → 250 → **0** |
  | LightOnOCR-1B + Vision | 212 → 42 → 3 | 1782 → 1899 → 0 |

  v1 = old char-level scorer; v2 = run-level reading (a measurement fix only); v3 = v2 + new flag
  rules (LightOnOCR empty/cut guard, sign-uncertain dash, collapsed boxes).
  - **The scorer fix mostly helped dev.** On test it removed no errors (the pair went 228 → 250).
  - **The flag rules are what reach 0.** For the pair, the guard alone gives test 0 / dev 5, and
    sign + thin boxes alone give dev 0 / test 250. Both are needed, and both are kept because test is 0.
  - **Honest note:** the guard's "empty output" case was first seen on a test page (the VIN page,
    `llong_page1`). Its "cut at the token limit" case is also visible on dev (3 SERFF pages).
- **Design change: structure on OCR pages comes from LightOnOCR's HTML table**
  (`PARSER_OCR_STRUCTURE=lighton`). The first draft had Tesseract's word geometry own the
  structure, but Tesseract drops single digits (the `Qty` column) and headers, which lost cells.
  - Each LightOnOCR number is **accepted only if Tesseract read the same number in the same block,
    in the same order**. The cell's box comes from that Tesseract word.
  - Label boxes (for indent levels) come from the Tesseract words in each row's y-range left of
    the value columns, placed by geometry, not spelling.
  - On the 8 gold OCR docs: structure from Tesseract gave numeric exact 50%, **2 unflagged**,
    79 shifts; structure from LightOnOCR gives **numeric exact 100%, 0 unflagged**, review 64%.
- **Flags on OCR pages:**
  - `ocr_unconfirmed`: a number the two engines do not both read the same.
  - `ocr_numbers_unmatched`: Tesseract read numbers in the block that LightOnOCR's table lacks,
    so a row may have been dropped. This flags the table.
  - `ocr_unverified`: LightOnOCR returned nothing (the dense VIN page: 1 token, 0 blocks) or hit
    the 4,096-token limit (8 of 56 bench pages). This flags the page.
  - **Sign rule:** a leading `-` / `−` / `–` on an OCR'd number is always flagged. Both engines
    read a printed en dash `–200` as `-200`, and agreeing on a shared misread is no protection.
  - **Collapsed boxes:** Tesseract jpn can give zero-width char boxes, so the number's position is
    unreliable and it is flagged.
- **Design change: the Vision spot check only flags (`spot_check_disagrees`); it never replaces a
  text layer.** On dev it gave false alarms on known-good layers (SERFF page1377: 3% agreement,
  while the text-layer parse got 650/670 right), and Vision returns nothing at all on some pages
  (tiny fonts, CINF). It counts only cells Vision did read, needs ≥ 10 of them, and flags under
  10% agreement (≈ 2% of dev pages).
- **Line-finder count check:** the S2 check is skipped on OCR pages. PyMuPDF's finder sees only
  drawn rules, and an image page has none.
- **Side-by-side OCR tables:** a LightOnOCR table whose last header row is k exact repeats of one
  column group (`Code | Rate | Min.` × 3) is split into k tables after building. The number check
  aligns Tesseract's row-by-row reading with the whole table. This fixed SERFF page275 (dev:
  shifts 2,217 → 2,095, +260 exact numbers); test is unchanged.
- **Garbled vs icon fonts:**
  - A page is `garbled_text_layer` only if bad glyphs (U+FFFD, private-use, control) are mixed
    into words with letters or digits in > 20% of cells, or make up > 50% of glyphs.
  - Icon fonts keep their text layer: SF Symbols, or Wingdings checkmarks as on LTCprodkit, sit
    in cells of their own.
  - A truly broken map is still caught: Apple Def 14a has words like `\x1b8?.;7*7,.`.
  - That changed 1 dev and 1 test page from OCR back to text layer. The test page lost 16 exact
    numbers (counts only; unflagged and shifts not worse).

**1.3 Table regions.** Use two signals and take their union:
- Ruling lines and filled rects from vector drawings (text-layer PDFs) or line detection (images).
- Whitespace/alignment: blocks of consecutive lines that share ≥2 stable right- or left-aligned
  x-anchors. This is what finds borderless tables like the sample.
- On OCR pages there are no vector drawings, so ruling lines come from horizontal and vertical
  run-length and projection profiles of the rendered raster (numpy only). Layout models (Table
  Transformer, DocLayout-YOLO) are **out of scope** under the no-heavy-vision constraint. When the
  two rules disagree, the region is flagged for review rather than resolved by a model.

Multiple tables per page are separate regions, split on vertical gaps or a change in column anchors.
Title, unit line and footnotes are attached to the nearest region above or below it.

**As built in m5 (text-layer pages):**
- **Ruled regions:** box edges come from PyMuPDF's own `page.find_tables(strategy="lines")`,
  which is already a dependency. Only the boxes are used, and cells come from our geometry
  (§1.4). Characters are assigned to the box that contains their center, so **side-by-side ruled
  tables stay apart**. A box needs ≥ 2 multi-cell rows. On dev it finds ruled tables on 272/301
  pages, and its count matches gold on 69%. It also catches "borderless" tables that have only
  top and bottom rules or row stripes: every gold region was found, IoU 0.88–1.0.
- **Header above the rules:** when the rule sits under the header (the sample), the free
  multi-cell rows directly above a box join it, nearest first, while each gap is ≤ 1.6 row
  pitches. A single-line row (caption, unit line) stops the absorption.
- **Unit line inside a box:** a leading single-line row inside a box that matches the unit
  pattern is lifted out as a caption.
- **Borderless regions:** text outside every box is cut into runs at vertical gaps > 2.2 row
  pitches. Each run is trimmed to its first through last multi-cell row (§1.4) and needs ≥ 2 such
  rows. Known limit: **side-by-side borderless tables are not split**; that needs a vertical-gutter rule.
- **Captions:** single-line rows within 40 pt above a table that overlap it in x become `unit`
  (`単位`, `千円`, `in thousands`, `Amounts in`, `$ in`) or `title`. They are no longer table rows.
- **Columns from cell rules (after m5):**
  - **Edges:** a ruled box takes its column edges from the finder's cell grid (the union of
    cell x-edges).
  - **Verified edges:** an interior edge is kept only if a vertical rule (line or rect edge) is
    actually drawn there over ≥ 60% of the region's height, because the finder can extend one
    table's grid into a borderless neighbour.
  - **Used only when the rules describe the cells:** there must be ≥ 2 columns, no two values of
    one row in one column, and every kept column holding at least one cell of the region.
    Otherwise the region keeps its geometry bands (§1.4).
  - **Grid split:** a grid splits where two consecutive rows each have an edge strictly inside
    the other's cells (two different grids stacked); merged cells cut only one way.
  - **Effect:** this recovers empty columns that have a header (`Notes`: 0 → 430/439 numbers on
    that page). Dev: shifts 3,056 → 2,093, numeric exact 0.485 → 0.525. Test: shifts
    1,930 → 1,891, but table F1 0.649 → 0.633. Gold unchanged at 1.0.
  - **Not fixed:** stacked tables with the same rules and no gap between them (SERFF page41).
- **Rotated pages** (`/Rotate 90|180|270`, landscape tables: 19 dev / 8 test pages): character
  boxes and line directions are mapped through `page.rotation_matrix` into display space, where
  `page.rect` and the ruled boxes already are. Without this, every line of a landscape page reads
  as vertical text.

**1.4 Structure** (as built in m1–m2; changes from the first draft are marked *changed*).
- **Candidate cells** (*changed*: there are no 0.25 em tokens). Start from characters with boxes,
  not from PDF spans or words: producers split spans arbitrarily (see "Split spans" in §0).
  Within a row, a visual gap wider than **1 em** starts a new candidate cell. Space characters are
  kept, so `(2025 年３月31 日)` stays one cell with its spaces.
  - **Value split:** a candidate cell made only of space-separated values (numbers, times, signed
    numbers, as in `7:04 7:09 7:14`) is split into one cell per value. Tightly set columns would
    otherwise merge. Known limit: this breaks space-grouped thousands (`1 234`), which is fine for ja/en.
  - **Rotated text:** a rotated text line (vertical column headers) is kept whole as one cell.
    Rotated lines that overlap vertically form one row.
- **Rows:** cluster characters by y-center (within 0.5 × glyph size).
  - **Wrapped labels** are merged. A label-only row whose label fills ≥ 75% of the label column,
    followed by a row whose label starts at the same x, is one label; its numbers sit on the last
    line. CJK joins with no space, other text with one.
- **Table rows (single-table m2 rule):** the first through the last row with ≥ 2 candidate cells.
  A page number in the bottom margin is excluded. Region detection proper is m5 (§1.3).
- **Columns ("bands")**, *changed* from right-edge anchor clustering:
  - Bands are built only from rows that hold a value (a number or a clock time). A cell joins the
    band it overlaps in x, **but two cells of the same row never share a band**, because that
    would be two columns merged. Right-aligned numbers overlap at their shared right edge, and
    centered or left-aligned text overlaps the same way. Every cell is then placed by geometry
    (max x-overlap, else nearest band), never by its order in the line.
  - **Fallback:** a table with no value rows (a text table, `EUR 20,000,000`) builds its bands from
    all its rows with ≥ 2 cells.
  - **Off-grid rows:** a value row whose numbers all sit in bands no other row uses (a centered
    numeric header `3 6 9 12` over right-aligned values) is left out and the bands are rebuilt.
  - **Split label column:** two neighbouring text-only bands that never share a row are merged
    into one label column (a `Total` row lacks the icon the other labels have).
  - Two cells still landing in one band are never merged silently: the table is flagged
    `column_conflict`.
- **Header** (*changed*). The header is the top rows that have text in a value column (any band
  but the first) and **no number** there. Year-only rows (`2025 2026`) count as header. The first
  draft's rule ("top rows before the first numeric row") would make section labels like 資産の部
  headers.
  - **Header cells spanning two lines:** a row sitting halfway between the line above and the
    line below (`Parameter | Symbol | Unit` next to `Value` over `Min Typ Max`) holds cells that
    span both lines. The line above joins the table, and the half-line row merges into it.
  - **Column spans:** each header cell anchors to its nearest band (left to right, one cell per
    band). A band with no header text of its own joins the nearest header cell **that has
    sub-headers under it** (`col_span`). A cell with nothing under it (`Symbol`) never takes
    columns.
  - **Row spans:** a cell whose bands are empty in the next header rows extends down (`row_span`).
  - **Empty stub cells:** empty header positions are separate empty cells, with no span, because
    nothing on the page shows one. This is also ParseBench's majority convention (11 of 16 tables
    with an empty stub over a multi-row header). The gold follows it.
- **Row hierarchy:** cluster label x0 into indent levels (2 pt apart). `parent_row` is the nearest
  preceding row with a smaller indent. Rows marked `合計`, `計`, `Total` or `Subtotal` become
  `is_total`. An underline rule above as a total signal is still to do.

**1.5 Cross-page tables.** Pages *p* and *p+1* are merged when the table on *p* touches the bottom
margin, the table on *p+1* starts at the top, the column anchors match within tolerance, and
either the header repeats or a continuation marker appears (`continued`, `(続き)`, `続く`).
A repeated header is dropped, not duplicated. Every row keeps its source page.
*As built (m5):* the last table of page *p* joins the first table of page *p+1* when they have the
same column count, band centers within 6 pt, and either an identical header or a `続き` / `続く` /
`continued` / `cont'd` marker in the next table's title. Not yet checked: that nothing else
follows the table on *p*. That's fine on the gold set, but it belongs in m4's checks.

**1.6 Principle: exactness first.** The parser never edits `raw_text`. Normalization only adds
fields. Each word inside a table region must end up in exactly one cell (conservation check in §5).

## 2. Output schema

### JSON (one file per document)

```json
{
  "doc_id": "balance_sheet_after_correction",
  "source_file": "balance_sheet_after_correction.pdf",
  "pages": [
    {"page": 1, "width": 595.2, "height": 841.8, "rotation": 0,
     "text_source": "text_layer",
     "text_layer_audit": {"bad_glyph_ratio": 0.0, "uncovered_ink_ratio": 0.0, "spot_cer": 0.0},
     "flags": []}
  ],
  "tables": [
    {
      "table_id": "t1",
      "title": "（１）連結貸借対照表",
      "unit": {"raw_text": "（単位：千円）", "scale": 1000, "currency": "JPY"},
      "regions": [{"page": 1, "bbox": [55, 115, 528, 672]}],
      "n_rows": 45, "n_cols": 3, "header_rows": 2,
      "columns": [
        {"col": 0, "header_path": [], "role": "label"},
        {"col": 1, "header_path": ["前連結会計年度", "(2025年３月31日)"], "role": "value"},
        {"col": 2, "header_path": ["当連結会計年度", "(2026年３月31日)"], "role": "value"}
      ],
      "rows": [
        {"row": 4, "page": 1, "indent_level": 2, "parent_row": 3, "is_total": false}
      ],
      "cells": [
        {"row": 4, "col": 1, "row_span": 1, "col_span": 1, "is_header": false,
         "page": 1, "bbox": [357.0, 172.3, 398.0, 181.0],
         "raw_text": "2,002,429",
         "text": "2,002,429",
         "value": "2002429", "value_type": "number",
         "blank_kind": null, "footnote_refs": [],
         "source": "text_layer", "confidence": 0.99, "flags": []}
      ],
      "footnotes": [{"marker": "※1", "raw_text": "...", "bbox": [0, 0, 0, 0]}],
      "confidence": 0.97,
      "needs_review": false,
      "issues": [{"check": "sum", "row": 14, "col": 1, "expected": "5863681",
                  "found": "5863684", "within_tolerance": true}]
    }
  ]
}
```

- Coordinates are PDF points with the origin at top-left. OCR pages convert from pixels.
- `value` is a **decimal string** (never a float) of the number as printed. The scale and unit
  are recorded in `unit` and **never applied**. There is no rescaled-value field, so the printed
  figure is never altered. If the unit or scale is missing, ambiguous, or differs between columns
  in a way the parser cannot pin down, the table is flagged for review.
- Spanned positions hold no cells. Only the anchor (top-left) cell is stored.

### CSV (two flavours; both derived from the JSON)
1. **`{doc_id}_{table_id}.csv`**: the table as it looks on the page. Header rows are kept as
   rows and every cell is `raw_text`. Spanned header cells repeat their text into each covered
   column. For eyeballing and spreadsheet use.
2. **`{doc_id}_cells.csv`** (long format, one row per cell): `doc_id, table_id, page, row, col,
   row_span, col_span, is_header, header_path (joined with " > "), indent_level, raw_text, value,
   value_type, blank_kind, scale, footnote_refs, x0, y0, x1, y1, confidence, flags`.
   For loading into databases and pandas.

UTF-8 with BOM (so Excel opens Japanese correctly) and RFC 4180 quoting.

*Status: **CSV is not built.** The app (`app.py`) saves Markdown, HTML, or JSON (the full §2 schema:
every cell with raw_text, value, bbox, flags); CSV was dropped at review in favour of JSON.*

## 3. Normalization (adds `text` and `value`; `raw_text` untouched)

| Case | Rule | Example → value |
|---|---|---|
| Width/compat forms | NFKC for `text` only | `３月` → `3月` |
| Thousands/decimal separators | Decided **per column** by majority evidence: `1,234.5` (en/ja), `1.234,5` (de/es), `1 234,5` (fr), `1'234.5` (ch), `12,34,567` (Indian lakh). If the column is ambiguous, `value=null` and the cell is flagged | `2,002,429` → `2002429` |
| Negatives | `-`, `−` (U+2212), `△`, `▲`, `(123)`, trailing `-`, `CR`. The marker is kept in `raw_text` | `△2,979` → `-2979`; `△0` → `-0` stored as `"0"` with flag `negative_zero` |
| Blanks | empty, `－`, `—`, `–`, `-` alone, `n/a`, `N/A`, `*` | `－` → `value=null, blank_kind="dash"` |
| Percent / currency | stripped into `value_type="percent"` or `unit`, never mixed into `value` | `12.5%` → `12.5` |
| Scale | Parsed from unit text (`単位：千円`, `百万円`, `in thousands`, `$ in millions`, `₹ in lakhs/crores`) and stored on the table (or column, if headers differ). Never multiplied into `value` | `千円` → `scale=1000` |
| Footnotes | Trailing `*1`, `※1`, `(a)`, superscript digits (detected by smaller font size and raised baseline) are moved into `footnote_refs` and removed from `value` | `1,234※2` → `1234`, refs `["※2"]` |
| Unparseable | `value=null`, `value_type="text"`. Digits are never dropped silently | |

Numbers are parsed with `decimal.Decimal`, never `float`, so `0.1`, `1e3` and large integers stay exact.

*As built (m3):*
- **What gets a value:** only a cell that is one number as a whole. Labels with digits
  (`(500 sheets x 5 reams)`), times, codes (`B9`) and ratios (`99 / 99`) keep `value=None`.
- **Headers** never get a value, even `2025`.
- **Separators:** ja/en only. `,` groups exactly 3 digits and `.` is the decimal point. Anything
  digit-like that doesn't parse cleanly (`1,23`) gets `value=None` and the flag `ambiguous_number`.
- **Units:** the unit line is parsed into `scale` (円 / 千円 / 百万円 / 億円 / in thousands,
  millions, billions / amounts in) and `currency` (JPY / USD / EUR / GBP). Recorded, never applied.
- **Results:** gold numeric exact is 1.0 on every text-layer doc, and all 17 gold units match. On
  ParseBench, 85% of digit cells get a value, with ~1.2k per split flagged `ambiguous_number`.
- **Not yet read:** leading-dot decimals (`.125`).

## 4. Correctness rules

1. **Text-layer characters are copied byte-for-byte.** No spell-check, no "fixing" OCR'd digits.
2. **Token conservation:** every word inside a table bbox is in exactly one cell. If any are
   left over or duplicated, the table is flagged.
3. **Column assignment by geometric anchor** (right edge for numbers). A number is never placed
   in a column whose anchor it does not overlap.
4. **Column type consistency:** a value column with one text-like outlier is flagged as a
   possible shift.
5. **Row count stability:** for a cross-page table, rows(p) + rows(p+1) − repeated header = merged rows.
6. **Ambiguity means flagging, not guessing** (separator ambiguity, OCR double-read disagreement,
   overlapping column anchors).

## 5. Validation and confidence (runs on every page and table)

| Check | Applies to | Signal |
|---|---|---|
| Text-layer audit (§1.1) | page | chooses source; bad → OCR |
| Token conservation | table | hard fail if violated |
| Grid sanity | table | rectangular after spans; no empty column; ≥1 header row or explicit `header_rows=0` |
| Column type purity | column | share of cells matching column's dominant type |
| Arithmetic (financial) | rows with `is_total`, net rows | `total ≈ Σ children` using the `parent_row` tree, with net rows (`（純額）`/`net`) standing in for their gross row plus contra rows; tolerance = (#addends) × 1 unit of last printed digit (the sample needs this: off by 3 over 10 children) |
| Cross-table identities | doc | e.g. 資産合計 = 負債純資産合計 / Assets = Liabilities + Equity when both pages present |
| OCR confidence | cell | min char conf; digit double-read agreement |
| Cross-page stitch | table | anchor match score |

**Confidence.** A cell's confidence is the minimum of its signals: source confidence (1.0 for a
text layer that passes the audit, otherwise OCR confidence) and any check penalties. A table's
confidence is the minimum over its hard checks and the 5th percentile of its cell confidences.
Minimum is used instead of a mean so that one bad digit cannot be averaged away.

*As built (m4):*
- **The blanket `needs_review` is gone.** A table is flagged only by its table-level checks:
  `unassigned_text` (a character inside the table area that is in no cell), `stitch_unconfirmed`
  (a cross-page join on a continuation marker alone, without a repeated header), and
  `unit_unknown` (a unit line printed whose scale is unknown).
- **Cell-level flags:**
  - `sum_mismatch`: totals use the indent tree. A section row without a value takes its total
    row's value, a gross row with a 純額/net child contributes the net value, and a net row is
    checked as gross + the other children. Tolerance is the number of addends × 1 unit of the last
    digit. A mismatch flags the total and all of its addends.
  - `type_outlier`: text in a ≥ 80% numeric column, or a number in a ≥ 80% text column.
  - `column_conflict`, `ambiguous_number` (m3) and `bad_glyph`.
- **Page flag** `numbers_outside_tables`: a multi-cell row with numbers that no table took.
- **Hierarchy after stitching:** it is recomputed over the whole joined table. Page 2 alone lacks
  its top level, which made every total on the cross-page gold fail.
- **Notes are not flags:** `negative_zero` is a note, not a review flag.
- **Known limit:** the rounding tolerance means **a last-digit error passes the totals check**
  (`2,002,429` → `2,002,428` sums within 10 units over 10 addends). Totals catch shifted, dropped,
  duplicated or misread-magnitude values, not single last digits. Text-layer digits are exact
  copies, so for them this is fine. For OCR, two independent reads must agree (§1.2).
- **Result:**
  - Gold: 96 totals pass, 0 fail, 0 false flags, unflagged 0. Review is 0% on the text-layer docs;
    the 35.5% overall is the image-only pages.
  - ParseBench dev: 31.5% of numbers wrong and unflagged, 28.7% sent to review.
  - ParseBench test: 27.6% wrong and unflagged, 32.7% sent to review.
  - 63% of dev's unflagged errors are gold tables the parser did not output as a matching table
    (merged, split or missed), on pages it did not flag.

*Page check (after m4), a setting `PARSER_PAGE_CHECKS`, default S2:*
- **`stray`:** a multi-cell row with numbers that is outside every table. Single-line text
  (paragraphs, captions, footnotes) never counts.
- **`count_lines`:** PyMuPDF's line finder sees a different number of tables on the page than we output.
- **`mid_header`:** a header-like row (text in ≥ 2 value columns, no numbers, numbers below)
  inside a table body, which suggests stacked tables merged into one.
- **Tuned on dev only:**

  | setting | dev unflagged | dev real* | dev review | test unflagged | test real* | test review |
  |---|---|---|---|---|---|---|
  | S0 stray | 31.5% | 20.0% | 28.7% | 27.6% | 13.3% | 32.7% |
  | S1 + count_lines | 22.5% | 15.6% | 46.0% | 18.5% | 9.0% | 59.1% |
  | **S2 + mid_header (default)** | 21.7% | 15.6% | 48.1% | 16.4% | 7.1% | **64.5%** |
  | S5 + escalate cell flags | 20.2% | 14.5% | 60.3% | 12.0% | 6.0% | 80.6% |

  \*real = not counting numbers whose whole row text matches an output row on the page (ParseBench
  splits cells differently); this is lenient.
- **Result:** S2 is the lowest unflagged with review < 50% **on dev**. **On test no setting except
  S0 stays under 50% review.** Zero unflagged under a 50% review budget is not reachable with these
  signals.
- **Where the gain comes from:** flagging only when the finder sees *more* tables than we output
  (merged tables) gave almost nothing on dev, so the gain comes from pages where we output
  borderless or split tables that the finder doesn't see.
- **Where dev's unflagged numbers sat at m4:** 82% are inside our output tables, but those tables
  are segmented differently from gold (merged or split). Only 1.5% are in multi-cell rows outside
  every table.

**Review flags (target: zero wrong numbers).** A numeric cell is either *verified* or
*flagged*; nothing in between. It is verified only if:
- its source is a text layer that passes the audit, or two OCR reads agree,
- its column assignment is unambiguous, and
- it passes every arithmetic check it takes part in.

Everything else gets a cell flag, and the table gets `needs_review=true` if any hard check
fails. τ is tuned so that `numeric_errors_unflagged` is **0 on the golden set and ParseBench**,
at the cost of more reviews. Review volume is reported, not optimized first. The review queue
lists page, bbox and reason so a reviewer goes straight to the cell.

## 6. Evaluation

### 6.0 What ParseBench covers, and what it doesn't
Checked on the local copy (`parsebench/table.jsonl`, Apache-2.0):

| | |
|---|---|
| Table pages | 503 one-page PDFs (427 `easy`, 76 `hard`); 893 GT tables (1–13 per page); 109k cells, 76k containing digits |
| GT format | HTML per page in `expected_markdown` (`<th>`, `rowspan`/`colspan`, `<strong>`, `<br/>`). **No bboxes, no normalized values.** 8 GT tables are ragged in the source HTML |
| Text layer | 497 have one, 5 have none, 1 garbled, so it barely exercises the OCR path |
| Language | **English only.** No Japanese anywhere in the table GT |
| Cross-page | **None.** Every record is a single page |
| Official metric | GTRM = mean(GriTS, TableRecordMatch), via `run-llama/ParseBench` (`parse-bench run <pipeline> --group table --input_dir ./parsebench`, works offline on the local copy) |
| CPU-only reference scores (Tables) | PyMuPDF4LLM **72.0**, Docling 66.4, LiteParse 42.4, OpenDataLoader 35.2, PyMuPDF text 0.0 |

So ParseBench is the external English benchmark, and the golden set below covers Japanese,
cross-page tables, the OCR path and the exact-number metrics that ParseBench cannot.

### 6.1 Ground truth
1. **Golden set: built** (`gold/`; 23 docs, 702 table rows):
   - Build with `python gold/build_gold.py`. It is deterministic (seed 7).
   - Verify with `python gold/check_gold.py`, which checks that the PDF text inside every gold
     cell bbox equals its `raw_text`, that `_ocr` docs have no text layer, and that spans tile
     each grid exactly once.
   - Contents:
     - **1 real doc:** the sample (`real_ja_balance_sheet`), 45 rows. It is read from the
       verified text layer and re-checked against the statement's own identities (流動資産合計,
       固定資産合計, 資産合計, 純額 rows).
     - **22 synthetic docs, ja and en.** Ground truth comes from the generator's own numbers and
       draw positions, never from parsing:

       | Case | Exercises |
       |---|---|
       | `bs_striped_hierarchy` | striped rows, 3 indent levels, `△0`/`(0)` |
       | `bs_borderless_footnotes` | no ruling, `※1`/`*1` label footnotes, blanks |
       | `bs_ruled_colspan_header` | 2-row header, `col_span=2` |
       | `bill_ruled_wrapped` | invoice with wrapped labels, numbers on the last line, negative discount |
       | `datasheet_rowspan_header` | `row_span=2` + `col_span=3` header, decimals, negatives, blanks |
       | `two_tables_one_page` | two tables with different column counts |
       | `bs_cross_page` | one table over 2 pages, repeated header, `（続き）`/`(continued)` |

     - `_ocr` variants (4 per language) are the same pages rendered to images with no text
       layer, which forces the OCR path.
   - Tags on each doc (`ja`, `ocr`, `cross_page`, ...) give per-stratum scores.
   - **Known limits:**
     - Only 1 real Japanese page; more real filings are the top gap (§9).
     - Synthetic fonts are Droid Sans Fallback (ja) and Helvetica (en), not MS-Mincho.
     - The cross-page case repeats its section labels.
     - There is no garbled-text-layer case (ParseBench has 1).
2. **Self-supervised OCR test.** Take digital PDFs with a text layer that passes the audit,
   rasterize them, strip the layer, and run the OCR path. The text layer is free ground truth
   for OCR CER and numeric exactness at any scale. Add degradations (noise, blur, skew, JPEG,
   low dpi) to mimic real scans.
3. **Synthetic tables.** Render generated HTML tables (random spans, indents, number formats,
   locales, page breaks) to PDF. This gives perfect ground truth for structure and stitching.
4. **XBRL cross-check (financial filings).** EDINET/TDnet (Japan) and SEC EDGAR (US) publish
   machine-readable XBRL for the same statements. Extracted numbers can be matched against XBRL
   facts, which gives exact-number ground truth at scale with no labeling.
5. **Label-free arithmetic pass rate.** On unlabeled filings, measure the share of `is_total`
   rows that reconcile. This works as a production canary.
6. **Public benchmarks** for comparison with published numbers: FinTabNet (financial,
   TEDS), PubTabNet (TEDS), PubTables-1M (GriTS), ICDAR-2013 / cTDaR 2019 (detection), and
   OmniDocBench (multi-language, multi-doc-type).
7. **ParseBench (Kaggle).** It is used two ways:
   - **Official score:** register the parser as a parse-bench provider that returns Markdown with
     HTML tables (`parse_bench.extensions.register_provider`, modeled on the built-in `pymupdf`
     provider), then run `parse-bench run <ours> --group table --input_dir ./parsebench`. The
     provider is written together with the parser.
   - **Our exact-number metrics:** `python eval/parsebench_to_gold.py parsebench/table.jsonl
     eval/parsebench_gold` converts the HTML GT (done: 503 docs). Then `evaluate.py
     eval/parsebench_gold PRED --no-teds` scores numeric exactness, column shifts and unflagged
     errors on 76k real digit cells. Tables are matched by cell-text overlap because the GT has
     no bboxes.
   - **Dev / test split** (`python eval/split_parsebench.py`): seed 7, ~60/40 **by source
     document** (pages of one filing are near-duplicates, so a page split would leak). The result
     is dev 303 pages / 43 documents and test 200 pages / 51 documents. The documents inspected
     while building m2 were forced into dev. **Tune only on dev, report both, never inspect test
     pages.**

### 6.2 Metrics (implemented in `eval/evaluate.py`)
| Metric | Why |
|---|---|
| Table detection P/R/F1 @ IoU 0.5 | missing tables / phantom tables |
| TEDS and TEDS-struct | standard structure+content score (PubTabNet) |
| Cell exact-match accuracy | "text exactly as printed" |
| **Numeric exact-match accuracy** (zero tolerance, Decimal) | the headline number for this project |
| Column-shift count | the specific failure we promise not to have. A shift means the gold value is missing from its column but present in another column of the same row. A cell holding the right text is never a shift, even when the row repeats the value (`1,123 \| 1,123`). Fixed in eval v2: v1 counted those as shifts |
| Missing / extra rows (diff-aligned) | dropped rows don't cascade into false shifts |
| **Unflagged numeric errors** (target 0) | the acceptance metric: wrong or missing numbers whose cell, table **or page** was *not* flagged for review. A page with non-empty `flags` in the prediction (e.g. `no_text_layer`) counts as flagged (eval v2) |
| Per-tag breakdown (`by_tag`) | e.g. `ja` vs `en`, `ocr` vs `text_layer`, `cross_page` |

Run with `python eval/evaluate.py GOLD_DIR PRED_DIR --out report.json [--no-teds]`
(`pip install apted`). Running it with no arguments runs the self-check. Both gold sets scored
against themselves give 1.0 with 0 errors. TEDS is pure Python and slow on ParseBench's
1000+-cell tables, so `--no-teds` is used there, where GriTS comes from the official scorer.
The script is model-agnostic, so any parser is scored the same way once converted to the §2 schema.

## 7. LightOnOCR (now part of the OCR fallback, §1.2)

The first draft excluded it (no GPU). With MPS allowed it runs on the M5 through transformers
5.5.4:
- **Memory:** 6.2 GB process + 7.8 GB MPS.
- **Speed:** ~70–145 s/page with greedy decoding.
- **Version pin:** 5.20.2 silently mis-loads the 1B checkpoint (random LM weights, garbage output).

Its risks from the first draft held up, which is why it never works alone:
- **Silent failures:** empty output on a dense page, cut at the token limit.
- **No per-cell boxes:** it gives boxes per block only.
- **Errors it shares with Tesseract:** the sign rule above.

## 8. Explicitly out of scope (for now)
Charts and figures, handwriting, form checkboxes, semantic mapping of row labels to a
taxonomy (e.g. to XBRL concepts), and translation. Add each only when asked.

## 9. Decisions and open questions

| # | Question | Decision |
|---|---|---|
| 1 | Ground truth | None in-house. ParseBench JSONL (HTML tables) is the external benchmark, and the golden set (§6.1) was built for the rest |
| 2 | Deployment | Offline, CPU only. Heavy OCR, vision and layout models are out of scope, which excludes LightOnOCR |
| 3 | Languages / inputs | Japanese and English; born-digital only (no photos or scans), but OCR is needed for textless pages and regions |
| 4 | Golden set | Built (23 docs). The sample is the first real doc |
| 5 | Values | Kept exactly as printed. Unit and scale are recorded, never applied. Ambiguous unit or scale goes to review |
| 6 | Review target | No wrong numbers: `numeric_errors_unflagged = 0`; review volume is secondary |
| 7 | OCR | Fallback only (missing, garbled or failing text layer), per page or region; MPS/MLX allowed; engine chosen by `ocr_bench/` numbers (Tesseract vs Apple Vision vs LightOnOCR-3, and pairs that accept a number only when two engines agree) |
| 8 | Eval v2 | Column-shift false positive on repeated values fixed; page-level flags count as flagged. Both have self-checks in `evaluate.py` |
| 9 | Empty stub cell | Separate empty cells, no span (ParseBench majority); the synthetic gold was changed to match |
| 10 | ParseBench split | Dev/test by source document, seed 7; tune on dev only |
| 11 | Milestone order | m5 (regions, multi-table, captions, cross-page) before m3: 294/503 ParseBench pages are multi-table, and that's where m2 loses numbers |
| 12 | Text-layer spot check | Vision OCR on a few crops, part of the audit, built in m6 |
| 13 | Header-only columns | Tried in m5: a band for a header cell that misses every band. It created fake columns under centered headers (`Qty`); edge-only had no effect on dev (3,064 → 3,063 shifts), so it was reverted. Empty interior columns stay an open cause of shifts |

Open:
- **More real Japanese pages.** Real filings (TDnet/EDINET 決算短信 or 有価証券報告書 PDFs,
  which also come with XBRL for exact-number checks) would replace the synthetic strata as the
  main Japanese signal. They need to be provided, or downloaded with permission.
- **OCR engine.** Decided: LightOnOCR-3-1B + Tesseract, greedy decoding (§1.2). Sampling did not repeat run to run.

## 10. Notes for the README (from review, 2026-10-23)
- OCR split honesty: the guard rule (LightOnOCR empty/cut) was found on a test page (the VIN page); on
  test it is the guard that gives 0 unflagged, the dash and thin-box rules alone leave test at 250. So
  "test 0" is not a fully clean held-out result.
- Review cost of v3 rules (OCR bench, positional sets): Tesseract dev 9.0% / test 27.8%; Vision 30.6 / 30.4;
  Vision-auto 2.3 / 3.6; LightOnOCR-1B alone 56.0 / 75.8 (its 2,807 -> 12 unflagged on test is whole-page
  flags: review 0% -> 75.8%); LightOnOCR+Tesseract 61.6 / 80.4; LightOnOCR+Vision 67.6 / 79.8.
- The repeated-header (side-by-side) split comes from one dev page (SERFF page275).
- The icon-font audit fix lost 16 exact numbers on one test page.

## 11. Page blocks: nothing on the page is lost (after m7)
Asked at review: the sample's three header lines were missing from the output. `（１）連結貸借対照表`
became the title, but `３．連結財務諸表及び主な注記` and `（訂正後）` were dropped, because only tables
were output.
- **Blocks:** every page now has `blocks` in reading order. Every character not inside a table
  cell is grouped into text lines (the same `rows_of` as tables), and each line is typed:
  - `heading`, by numbering (`第N章`, `N．` level 1, `（N）` level 2, `①` level 3), a short
    bracketed label (`（訂正後）`, `【連結】`), or a font ≥ 1.15× the page median;
  - `unit`, `footnote` (`※`, `*1`, `注`, `Note`), `furniture` (page number);
  - `caption`, for a line stacked directly above a table;
  - `text` otherwise.
- **Tables:** they get `captions`, the whole chain of lines directly above them (nearest within
  40 pt, then each next one up within 1.5 line heights, at most 6). `title` and `unit` are
  unchanged, so no table and no score changed.
- **OCR pages:** these use LightOnOCR's own labeled blocks, run through the same line rules,
  marked `ocr_text_unverified`.
- **Cross-page tables:** later pages carry `{"type": "table", "continued": true}`.
- **Nothing-lost check:** a glyph in no cell and no line flags the page `text_not_captured`. It
  fires on 0 of 530 pages. Two real bugs were found while building it:
  - my first counter double-counted overlapping boxes;
  - a cell's right edge was taken from its last glyph rather than its widest, which only mattered
    for overlapping OCR word boxes. Fixing it changed boxes on 1 test page and 1 dev page; scores
    are identical (0 pages better or worse on dev and test).
- **Limits:** single-column reading order; heading rules, not a model; text blocks are not scored.
