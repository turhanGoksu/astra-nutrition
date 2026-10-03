"""Build the curated nutrition table from USDA FoodData Central.

Inputs (both datasets are public domain, downloaded into data/raw/ if missing):
    data/food_selection.csv        foods from USDA SR Legacy (single foods)
    data/food_selection_fndds.csv  dishes from USDA FNDDS (mixed dishes)
    Each row: hand-curated names, aliases and portion specs.
Outputs:
    src/astra_nutrition/data/foods.csv          one row per food, macros per 100 g
    src/astra_nutrition/data/food_portions.csv  one row per (food, unit) with the
                                                source of the grams

Every gram value is traceable: a portion either references a USDA household
measure ("usda:<measure>") or is an explicit assumption (a plain number).

FNDDS portions are US sizes ("1 piece" of baklava is 80 g), so FNDDS dishes
get only weight and volume measures; "adet" and "dilim" stay undefined and
read as an unknown amount instead of a wrong one (Design R).

Usage (from the project root):
    python -m scripts.build_food_table
"""

import csv
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from astra_nutrition.amounts import Unit
from astra_nutrition.text import fold

RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class Dataset:
    """Where a USDA dataset comes from and how its CSVs differ."""

    name: str
    label: str  # shown to users in food_source
    url: str
    directory: Path
    selection: Path
    nutrients: dict[str, str]  # food_nutrient.nutrient_id -> our macro name
    portion_column: str  # food_portion.csv column that names the measure


SR_LEGACY = Dataset(
    name="sr_legacy",
    label="USDA SR Legacy",
    url="https://fdc.nal.usda.gov/fdc-datasets/"
    "FoodData_Central_sr_legacy_food_csv_2018-04.zip",
    directory=RAW_DIR / "sr_legacy" / "FoodData_Central_sr_legacy_food_csv_2018-04",
    selection=Path("data/food_selection.csv"),
    nutrients={"1008": "kcal", "1003": "protein", "1005": "carbs", "1004": "fat"},
    portion_column="modifier",
)
FNDDS = Dataset(
    name="fndds",
    label="USDA FNDDS (FoodData Central 2024-10-31)",
    url="https://fdc.nal.usda.gov/fdc-datasets/"
    "FoodData_Central_survey_food_csv_2024-10-31.zip",
    directory=RAW_DIR / "fndds" / "FoodData_Central_survey_food_csv_2024-10-31",
    selection=Path("data/food_selection_fndds.csv"),
    # FNDDS refers to nutrients by their number (208 = energy), not their id.
    nutrients={"208": "kcal", "203": "protein", "205": "carbs", "204": "fat"},
    portion_column="portion_description",
)
DATASETS = (SR_LEGACY, FNDDS)

# Turkish dishes without a USDA equivalent, computed from SR Legacy ingredients.
RECIPES_PATH = Path("data/recipes.csv")
RECIPE_INGREDIENTS_PATH = Path("data/recipe_ingredients.csv")

# Written into the package so the table ships with it.
PACKAGE_DATA = Path("src/astra_nutrition/data")
FOODS_PATH = PACKAGE_DATA / "foods.csv"
PORTIONS_PATH = PACKAGE_DATA / "food_portions.csv"

MACROS = ("kcal", "protein", "carbs", "fat")
SPEC_UNITS = {
    unit.value: unit
    for unit in Unit
    if unit not in (Unit.GRAM, Unit.MILLILITER, Unit.PORTION)
}
US_CUP_ML = 236.588
FL_OZ_ML = 29.5735
_FL_OZ = re.compile(r"(\d+(?:\.\d+)?)?\s*fl oz")

Portions = list[tuple[str, float]]  # (USDA measure, grams per one unit)


def ensure_usda(dataset: Dataset) -> None:
    """Download and extract a dataset's CSVs if they are not present."""
    if (dataset.directory / "food.csv").exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{dataset.name}.zip"
    print(f"Downloading {dataset.url}")
    urllib.request.urlretrieve(dataset.url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DIR / dataset.name)


def load_usda(
    dataset: Dataset, fdc_ids: set[str]
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, Portions]]:
    """Load descriptions, macros per 100 g and household portions."""
    with open(dataset.directory / "food.csv", encoding="utf-8") as f:
        descriptions = {
            r["fdc_id"]: r["description"]
            for r in csv.DictReader(f)
            if r["fdc_id"] in fdc_ids
        }
    nutrients: dict[str, dict[str, float]] = defaultdict(dict)
    with open(dataset.directory / "food_nutrient.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            macro = dataset.nutrients.get(r["nutrient_id"])
            if r["fdc_id"] in fdc_ids and macro is not None:
                nutrients[r["fdc_id"]][macro] = float(r["amount"])
    portions: dict[str, Portions] = defaultdict(list)
    with open(dataset.directory / "food_portion.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fdc_id"] in fdc_ids:
                # SR Legacy: grams for `amount` units; FNDDS: the description
                # holds the quantity ("1 cup") and `amount` is empty.
                amount = float(r["amount"]) if r["amount"] else 1.0
                grams = float(r["gram_weight"]) / amount
                portions[r["fdc_id"]].append((r[dataset.portion_column].strip(), grams))
    return descriptions, nutrients, portions


def find_portion(prefix: str, portions: Portions) -> tuple[str, float] | None:
    """Find a USDA portion by modifier: exact match first, then prefix."""
    for modifier, grams in portions:
        if modifier.lower() == prefix.lower():
            return modifier, grams
    for modifier, grams in portions:
        if modifier.lower().startswith(prefix.lower()):
            return modifier, grams
    return None


def volume_ml(modifier: str) -> float | None:
    """Volume of a USDA household measure, if it is a volume measure."""
    match = _FL_OZ.search(modifier.lower())
    if match:
        return float(match.group(1) or 1) * FL_OZ_ML
    if "cup" in modifier.lower():
        return US_CUP_ML
    return None


def parse_spec(spec: str) -> list[tuple[str, str]]:
    """Split "piece=usda:large;bowl=200" into [("piece", "usda:large"), ...]."""
    items = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        key, _, value = part.partition("=")
        items.append((key.strip(), value.strip()))
    return items


def recipe_per_100g(
    parts: list[tuple[float, dict[str, float]]], cooked_grams: float
) -> dict[str, float]:
    """Macros per 100 g of a cooked dish from (raw grams, macros per 100 g) parts.

    The macros of the ingredients stay in the pot while water evaporates, so
    they are divided by the cooked weight, not by the raw total (Design B).
    """
    raw_grams = sum(grams for grams, _ in parts)
    if not 0 < cooked_grams <= raw_grams:
        raise ValueError(
            f"cooked weight {cooked_grams:g} g must be above 0 and at most "
            f"the raw total {raw_grams:g} g"
        )
    return {
        macro: round(
            sum(grams * per_100g[macro] for grams, per_100g in parts) / cooked_grams,
            2,
        )
        for macro in MACROS
    }


def resolve_grams(value: str, portions: Portions) -> tuple[str, float, str] | None:
    """(measure, grams, source) for "usda:<measure>" or a plain number."""
    if not value.startswith("usda:"):
        return "", float(value), "assumption"
    found = find_portion(value.removeprefix("usda:"), portions)
    if found is None:
        return None
    measure, grams = found
    return measure, grams, f"usda: {measure}"


def build() -> list[str]:
    """Build both output CSVs. Returns a list of errors (empty on success)."""
    errors: list[str] = []
    foods: list[dict[str, object]] = []
    portion_rows: list[dict[str, object]] = []
    seen_names: dict[str, str] = {}
    with open(RECIPES_PATH, encoding="utf-8") as f:
        recipes = list(csv.DictReader(f))
    with open(RECIPE_INGREDIENTS_PATH, encoding="utf-8") as f:
        ingredients = list(csv.DictReader(f))
    sr_legacy: tuple[dict[str, str], dict[str, dict[str, float]]] = ({}, {})

    for dataset in DATASETS:
        with open(dataset.selection, encoding="utf-8") as f:
            selection = list(csv.DictReader(f))
        ensure_usda(dataset)
        fdc_ids = {row["fdc_id"] for row in selection}
        if dataset is SR_LEGACY:  # recipe ingredients are SR Legacy foods
            fdc_ids |= {row["fdc_id"] for row in ingredients}
        descriptions, nutrients, usda_portions = load_usda(dataset, fdc_ids)
        if dataset is SR_LEGACY:
            sr_legacy = (descriptions, nutrients)

        for row in selection:
            food_id, fdc_id = row["id"], row["fdc_id"]
            if any(food["id"] == food_id for food in foods):
                errors.append(f"{food_id}: id used twice")
                continue
            if fdc_id not in descriptions:
                errors.append(f"{food_id}: fdc_id {fdc_id} not in {dataset.label}")
                continue
            macros = nutrients.get(fdc_id, {})
            if set(macros) != set(MACROS):
                errors.append(f"{food_id}: missing macros {set(MACROS) - set(macros)}")
                continue

            aliases = _claim_names(row, seen_names, errors)

            default = resolve_grams(row["default_grams"], usda_portions[fdc_id])
            if default is None:
                errors.append(f"{food_id}: no USDA portion '{row['default_grams']}'")
                continue

            density: float | None = None
            for key, value in parse_spec(row["portions"]):
                resolved = resolve_grams(value, usda_portions[fdc_id])
                if resolved is None:
                    errors.append(f"{food_id}: no USDA portion matching '{value}'")
                    continue
                measure, grams, source = resolved

                if key == "density":
                    ml = volume_ml(measure) if measure else None
                    if ml is None:
                        errors.append(
                            f"{food_id}: density needs a volume measure, got '{value}'"
                        )
                        continue
                    density = round(grams / ml, 3)
                elif key in SPEC_UNITS:
                    portion_rows.append(
                        {
                            "food_id": food_id,
                            "unit": key,
                            "grams": round(grams, 2),
                            "source": source,
                        }
                    )
                else:
                    errors.append(f"{food_id}: unknown portion unit '{key}'")

            foods.append(
                {
                    "id": food_id,
                    "fdc_id": fdc_id,
                    "usda_description": descriptions[fdc_id],
                    "name_en": row["name_en"],
                    "name_tr": row["name_tr"],
                    "aliases": "|".join(aliases),
                    "kcal_100g": macros["kcal"],
                    "protein_100g": macros["protein"],
                    "carbs_100g": macros["carbs"],
                    "fat_100g": macros["fat"],
                    "default_grams": round(default[1], 2),
                    "density_g_per_ml": density,
                    "note": row["note"],
                    "source": f"{dataset.label}, fdc_id {fdc_id}",
                    # FNDDS defaults are US pieces: "2 baklava" must not mean 2 x 80 g.
                    "bare_count": "unknown" if dataset is FNDDS else "portion",
                }
            )

    for row in recipes:
        food = _recipe_food(row, ingredients, sr_legacy, seen_names, errors)
        if food is None:
            continue
        if any(other["id"] == food["id"] for other in foods):
            errors.append(f"{food['id']}: id used twice")
            continue
        foods.append(food)
        for key, value in parse_spec(row["portions"]):
            if key not in SPEC_UNITS or value.startswith("usda:"):
                errors.append(f"{food['id']}: recipe portions are plain grams per unit")
                continue
            portion_rows.append(
                {
                    "food_id": food["id"],
                    "unit": key,
                    "grams": float(value),
                    "source": "assumption",
                }
            )

    if not errors:
        _write_csv(FOODS_PATH, foods)
        _write_csv(PORTIONS_PATH, portion_rows)
    return errors


def _claim_names(
    row: dict[str, str], seen_names: dict[str, str], errors: list[str]
) -> list[str]:
    """Register a food's names; report any folded name another food owns."""
    aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
    for name in [row["name_en"], row["name_tr"], *aliases]:
        owner = seen_names.setdefault(fold(name), row["id"])
        if owner != row["id"]:
            errors.append(f"{row['id']}: name '{name}' already used by {owner}")
    return aliases


def _recipe_food(
    row: dict[str, str],
    ingredients: list[dict[str, str]],
    sr_legacy: tuple[dict[str, str], dict[str, dict[str, float]]],
    seen_names: dict[str, str],
    errors: list[str],
) -> dict[str, object] | None:
    """A foods.csv row for a recipe dish, or None (with errors) if invalid."""
    descriptions, nutrients = sr_legacy
    recipe_id = row["id"]
    lines = [i for i in ingredients if i["recipe_id"] == recipe_id]
    if not lines:
        errors.append(f"{recipe_id}: recipe has no ingredients")
        return None
    parts: list[tuple[float, dict[str, float]]] = []
    for line in lines:
        macros = nutrients.get(line["fdc_id"], {})
        if line["fdc_id"] not in descriptions or set(macros) != set(MACROS):
            errors.append(f"{recipe_id}: ingredient fdc_id {line['fdc_id']} unusable")
            return None
        parts.append((float(line["grams"]), macros))
    try:
        per_100g = recipe_per_100g(parts, float(row["cooked_grams"]))
    except ValueError as exc:
        errors.append(f"{recipe_id}: {exc}")
        return None

    aliases = _claim_names(row, seen_names, errors)
    raw_grams = sum(grams for grams, _ in parts)
    return {
        "id": recipe_id,
        "fdc_id": "",
        "usda_description": "Recipe: "
        + "; ".join(
            f"{line['grams']} g {descriptions[line['fdc_id']]}" for line in lines
        ),
        "name_en": row["name_en"],
        "name_tr": row["name_tr"],
        "aliases": "|".join(aliases),
        "kcal_100g": per_100g["kcal"],
        "protein_100g": per_100g["protein"],
        "carbs_100g": per_100g["carbs"],
        "fat_100g": per_100g["fat"],
        "default_grams": float(row["default_grams"]),
        "density_g_per_ml": None,
        "note": row["note"],
        "source": (
            f"Recipe from USDA SR Legacy ingredients; cooked weight "
            f"{float(row['cooked_grams']):g} g of {raw_grams:g} g raw "
            f"({row['cooked_grams_source']})"
        ),
        "bare_count": "portion",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    problems = build()
    if problems:
        print("Build failed:", *problems, sep="\n  - ")
        sys.exit(1)
    print(f"Wrote {FOODS_PATH} and {PORTIONS_PATH}")
