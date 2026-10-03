# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). While the version is 0.x, minor
releases may change the API.

A released version is never changed: fixes ship as a new version.

## [Unreleased]

### Added

- Browser demo with Gradio (`demo/`), run locally with one Docker command
  and also ready as a Hugging Face Docker Space. It installs the hash-checked
  v0.1.0 wheel and runs offline, without the LLM judge.
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

[Unreleased]: https://github.com/turhanGoksu/astra-nutrition/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/turhanGoksu/astra-nutrition/releases/tag/v0.1.0
