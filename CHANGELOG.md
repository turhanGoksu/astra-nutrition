# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). While the version is 0.x, minor
releases may change the API.

A released version is never changed: fixes ship as a new version.

## [Unreleased]

## [0.3.0] - 2026-10-04

Seventeen common Turkish dishes. Results change for meals that mention them,
hence a minor version.

### Added

- 17 Turkish recipes, computed like the earlier ones (macros divided by the
  cooked weight, every amount a marked assumption): simit, poğaça, kaşarlı
  tost, sucuklu yumurta, çılbır, mantı, tavuk döner, et döner, İskender,
  karnıyarık, kuru fasulye, gözleme, pide, su böreği, ezogelin çorbası, cacık,
  kısır. The table has 158 foods.
- One kind per dish: a plain name means a documented default (`pide` is
  kıymalı, `gözleme` peynirli, `döner` a dürüm); another kind (`patatesli
  gözleme`) stays `unmatched` instead of matching the default. Where USDA has
  no exact ingredient a stated approximation stands in (sucuk, lavaş, yufka).

### Changed

- `kuru fasulye` is now a recipe (beans cooked with onion, tomato paste and
  oil) instead of plain boiled beans.
- Evaluation labels: names that were `none` because their dish was missing now
  have the new recipe as gold (13 v0.1 names, 9 v0.2 dev meals, 5 v0.2 test
  meals). The v0.2 test set has been seen and is not reported again as a
  held-out result; v0.2 dev meals: 109 of 115 foods found.

## [0.2.1] - 2026-10-04

Packaging only: the code, the food table and every result are the same as in
0.2.0.

### Added

- Published on PyPI: `pip install astra-nutrition`. A GitHub release is
  uploaded to TestPyPI and, after a maintainer approves, to PyPI by
  `.github/workflows/publish.yml`, with trusted publishing (no API tokens). The
  uploaded files are the ones attached to the release, not a rebuild.

### Fixed

- README links are absolute, so the image and the changelog link work on PyPI.

## [0.2.0] - 2026-10-03

Turkish dishes, stricter checks on the parser's output, and an honest
held-out measurement.

### Added

- Browser demo with Gradio (`demo/`), run locally with one Docker command
  and also ready as a Hugging Face Docker Space. It installs the hash-checked
  release wheel and runs offline, without the LLM judge.
- Grounding check in the parser: every word of an item name must appear in
  the meal text, so an invented food is rejected with a reason instead of
  counted. On by default (`MealParser(..., check_grounding=False)` turns it
  off). The CLI lists rejected items.
- `totals.rejected`: the number of items the parser checks refused.
- 7 mixed dishes from USDA FNDDS (public domain) that exist there as they
  are: baklava, zeytinyağlı and etli yaprak sarma, etli and zeytinyağlı biber
  dolması, falafel, tabule. FNDDS pieces are US sizes, so these dishes have
  only weight and volume measures: `1 dilim baklava` is `amount_unknown`,
  `100 g baklava` is counted. They change no evaluation decision.
- `foods.csv` has a `source` column naming the dataset and FDC id, and a
  `bare_count` column: a bare count of an FNDDS dish (`2 baklava`) is
  `amount_unknown` instead of two US-sized default portions.
- Meal-time words are dropped from parsed names (`Akşam Biber Dolması` →
  `Biber Dolması`) and read as no amount (`Menemen [sabah]`); an item that is
  only meal-time words (`Öğle Yemeği`) is rejected.
- v0.2 dev meals: 95 meals written by someone other than the alias author,
  labeled per meal and scored end to end (`python -m eval.v02`). Found
  foods went from 64% to 83%, false matches from 4 to 3. Aliases from them:
  `süzme mercimek` (was raw lentils), `biber dolma(sı)` (was raw peppers),
  `fıstıklı baklava`, `soğuk ayran`, `tabule salatası`, `zeytin`, `tavuk
  ızgara`, `somon ızgara`; `1 adet ayran` is a 200 ml cup.
- A fuzzy match that adds a dish word to an alias is refused (`falafel wrap`
  is not falafel, `etli kuru fasulye` not plain beans); serving words such
  as `ızgara` or `soğuk` are allowed. With the judge on, such names go to
  the judge.
- A parsed name that lists table foods without a conjunction is split
  (`Kofte Pilav` -> köfte + pilav) when every part is an exact table name.
  v0.2 dev meals: found foods 83% -> 94%.
- v0.2 test meals: 58 new meals, labeled before the system ran on them and
  evaluated once. Found foods: 57% (v0.1.0: 14%), false matches: 3 (v0.1.0:
  1). Most misses are descriptive words glued to names (`Lahmacun Acili`).
- Amount check: a number in the parsed amount that the user never wrote
  (the parser's `2 adet` for a plain `köfte`) is treated as a missing amount,
  so the item is `estimated` with the default portion and says why.
- Turkish recipe dishes computed from SR Legacy ingredients
  (`data/recipes.csv`, `data/recipe_ingredients.csv`). Macros are divided by
  the cooked weight, which each recipe states with its source; the build
  refuses a cooked weight above the raw total. Recipes: mercimek çorbası,
  ayran, çoban salatası, menemen, ızgara köfte, lahmacun. Grilled meat drips
  fat, so ızgara köfte uses USDA's measured broiled patty instead of raw mince.

### Changed

- `totals.complete` is now false when the parser rejected an item, since a
  rejected item may be a real food that the total leaves out.
- CI runs on a pinned `ubuntu-24.04` instead of the moving `ubuntu-latest`.
- PostgreSQL `foods.fdc_id` may be NULL (recipe dishes); ingest upgrades a
  database created by v0.1.0.
- One v0.1 test name changed with the new aliases: `Zeytin` now matches
  black olives (correct; the alias came from the v0.2 dev meals).
- The fuzzy rule also fixes two v0.1 test errors (`Etli Kuru Fasulye`,
  `canned tuna in oil`), and a test name (`70% dark chocolate`) shaped a
  detail of it, so the v0.1 test numbers are no longer a clean held-out
  result.
- Evaluation: eight dev names (mercimek çorbası, köfte, çoban salatası,
  menemen, ayran, lahmacun) now have their recipe as the gold food instead of
  none; dishes were chosen from dev names and the request log only, and no
  test name changed.

## [0.1.0] - 2026-10-02

First public release (alpha).

### Added

- **Library** (`astra_nutrition`): `Analyzer` turns Turkish or English meal
  text into foods, grams and nutrition, offline after the first model download.
  - Parsing with the fine-tuned 1.5B meal parser
    ([Turhan123/astra-meal-parser-gguf](https://huggingface.co/Turhan123/astra-meal-parser-gguf),
    Q4_K_M) on CPU via llama.cpp. Output is validated and every parse has an
    explicit status; merged items with a conjunction are parsed again.
  - Deterministic amount normalization: Turkish and English units, number
    words and fractions. Weights the model adds in parentheses are used only
    if the user wrote them; an amount that repeats the item's name
    (`1 muz`) reads as a count.
  - Matching: exact, then fuzzy (trigram similarity identical to PostgreSQL
    pg_trgm). Unmatched items are explicit and keep their closest candidate.
  - Honest totals: item statuses `ok`, `estimated`, `amount_unknown`,
    `unmatched`, and `complete` / `includes_estimates` flags on the totals.
  - Bundled food table: 128 foods, 455 Turkish and English aliases and 190
    portions from USDA SR Legacy (CC0), each food traceable to its FDC id.
  - Your own foods from a strictly validated CSV (`--foods`,
    `FoodTable.with_user_foods`).
  - Optional LLM judge (`[judge]` extra): embedding retrieval, then a Groq,
    Gemini or any OpenAI-compatible model picks one of the offered foods or
    none. Only item names are sent.
  - Optional PostgreSQL + pgvector index (`[postgres]` extra) with the same
    results as the in-memory index.
- **CLI** `astra-nutrition` with table and JSON output.
- **Web service** (FastAPI): `/analyze`, `/foods/search`, `/stats/unmatched`,
  `/health`, and a per-item request log in PostgreSQL with best-effort writes.
- **Docker**: a slim image with the model in a volume; the judge is an opt-in
  build.
- **Evaluation** harness with a leakage-safe dev/test split; thresholds were
  tuned on dev only.
- **CI**: lint, unit tests, PostgreSQL parity tests and a clean wheel install.

### Known limitations

- Many Turkish dishes are not in the table yet; they are reported as
  `unmatched`.
- Parser output can differ between llama.cpp builds and machines, even at
  temperature 0.
- Fuzzy matching can match a modified dish to its base food
  (`Etli Kuru Fasulye`).

[Unreleased]: https://github.com/turhanGoksu/astra-nutrition/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/turhanGoksu/astra-nutrition/releases/tag/v0.3.0
[0.2.1]: https://github.com/turhanGoksu/astra-nutrition/releases/tag/v0.2.1
[0.2.0]: https://github.com/turhanGoksu/astra-nutrition/releases/tag/v0.2.0
[0.1.0]: https://github.com/turhanGoksu/astra-nutrition/releases/tag/v0.1.0
