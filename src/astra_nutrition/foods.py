"""The food table: per-100 g nutrition, default portions and household units.

Users can add their own foods from a CSV (see ``load_user_foods``). Merging is
strict: an id or a name that already belongs to another food is an error at
load time, unless the user explicitly replaces that food. A silent override
would change results for everyone; a silent duplicate would make the matcher
refuse the name. Both are worse than a clear error.
"""

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from astra_nutrition.amounts import FoodPortions, Unit
from astra_nutrition.tables import read_table
from astra_nutrition.text import fold

USER_UNITS = [u for u in Unit if u not in (Unit.GRAM, Unit.MILLILITER, Unit.PORTION)]
REQUIRED_COLUMNS = [
    "id",
    "name_en",
    "name_tr",
    "kcal_100g",
    "protein_100g",
    "carbs_100g",
    "fat_100g",
    "default_grams",
    "source",
]
OPTIONAL_COLUMNS = [
    "aliases",
    "density_g_per_ml",
    "note",
    *(f"grams_per_{unit.value}" for unit in USER_UNITS),
]
_ID = re.compile(r"[a-z0-9_]+")


class FoodTableError(ValueError):
    """A user food file or merge has problems; ``problems`` lists all of them."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(["Food table problems:", *problems]))


@dataclass(frozen=True)
class Food:
    """One food with nutrition per 100 g (kcal straight from the source data)."""

    id: str
    name_en: str
    name_tr: str
    kcal_100g: float
    protein_100g: float
    carbs_100g: float
    fat_100g: float
    default_grams: float
    density_g_per_ml: float | None
    source: str
    note: str = ""
    count_as_portion: bool = True  # see FoodPortions; False for FNDDS dishes


class FoodTable:
    """Foods and their portions, built from rows in the bundled CSV format."""

    def __init__(
        self, food_rows: list[dict[str, str]], portion_rows: list[dict[str, str]]
    ) -> None:
        self.rows = food_rows  # kept for building indexes over the same foods
        self.portion_rows = portion_rows
        self._foods = {row["id"]: _to_food(row) for row in food_rows}
        units: dict[str, dict[Unit, float]] = defaultdict(dict)
        for row in portion_rows:
            units[row["food_id"]][Unit(row["unit"])] = float(row["grams"])
        self._units = dict(units)

    @classmethod
    def bundled(cls) -> "FoodTable":
        """The table shipped with the package."""
        return cls(read_table("foods.csv"), read_table("food_portions.csv"))

    def __len__(self) -> int:
        return len(self._foods)

    def __contains__(self, food_id: object) -> bool:
        return food_id in self._foods

    def get(self, food_id: str) -> Food:
        return self._foods[food_id]

    def portions(self, food_id: str) -> FoodPortions:
        food = self._foods[food_id]
        return FoodPortions(
            default_grams=food.default_grams,
            unit_grams=self._units.get(food_id, {}),
            density_g_per_ml=food.density_g_per_ml,
            count_as_portion=food.count_as_portion,
        )

    def with_user_foods(
        self, path: str | Path, replace_ids: Iterable[str] = ()
    ) -> "FoodTable":
        """A new table with the foods of a user CSV added (see load_user_foods)."""
        food_rows, portion_rows = load_user_foods(path)
        return self.extend(food_rows, portion_rows, replace_ids)

    def extend(
        self,
        food_rows: list[dict[str, str]],
        portion_rows: list[dict[str, str]],
        replace_ids: Iterable[str] = (),
    ) -> "FoodTable":
        """A new table with extra foods. Conflicts raise FoodTableError.

        ``replace_ids`` lists existing foods the new rows deliberately replace
        (their old names, aliases and portions are dropped).
        """
        replace = set(replace_ids)
        problems = [
            f"replace_ids: {food_id!r} is not in the table"
            for food_id in sorted(replace - set(self._foods))
        ]
        kept = [row for row in self.rows if row["id"] not in replace]
        owners = {fold(name): row["id"] for row in kept for name in _names(row)}
        for row in food_rows:
            if row["id"] in self._foods and row["id"] not in replace:
                problems.append(
                    f"id {row['id']!r} already exists; add it to replace_ids to "
                    "replace it"
                )
            for name in _names(row):
                owner = owners.setdefault(fold(name), row["id"])
                if owner != row["id"]:
                    problems.append(
                        f"name {name!r} of {row['id']!r} is already used by {owner!r}"
                    )
        if problems:
            raise FoodTableError(problems)
        kept_portions = [p for p in self.portion_rows if p["food_id"] not in replace]
        return FoodTable(kept + food_rows, kept_portions + portion_rows)


def load_user_foods(
    path: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read and validate a user food CSV; return (food rows, portion rows).

    Required columns: id, name_en, name_tr, kcal_100g, protein_100g,
    carbs_100g, fat_100g, default_grams, source (where the numbers come from).
    Optional: aliases ("|"-separated), density_g_per_ml, note, and one
    grams_per_<unit> column per household unit (piece, slice, bowl, plate,
    glass, cup, tbsp, tsp, handful). Unknown columns are an error, so a typo
    such as "kcal_100" cannot be ignored silently.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    problems = [f"missing column {c!r}" for c in REQUIRED_COLUMNS if c not in columns]
    known = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
    problems += [f"unknown column {c!r}" for c in columns if c not in known]
    if problems:
        raise FoodTableError(problems)

    food_rows: list[dict[str, str]] = []
    portion_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        where = f"line {line}"
        food_id = (row["id"] or "").strip()
        if not _ID.fullmatch(food_id):
            problems.append(f"{where}: id must use a-z, 0-9 and _ (got {food_id!r})")
        elif food_id in seen_ids:
            problems.append(f"{where}: duplicate id {food_id!r}")
        seen_ids.add(food_id)
        for column in ("name_en", "name_tr", "source"):
            if not (row[column] or "").strip():
                problems.append(f"{where}: {column} is empty")
        for column in ("kcal_100g", "protein_100g", "carbs_100g", "fat_100g"):
            problems += _check_number(row, column, where, allow_zero=True)
        problems += _check_number(row, "default_grams", where, allow_zero=False)
        if (row.get("density_g_per_ml") or "").strip():
            problems += _check_number(row, "density_g_per_ml", where, allow_zero=False)

        food_rows.append(
            {
                "id": food_id,
                "name_en": row["name_en"].strip(),
                "name_tr": row["name_tr"].strip(),
                "aliases": (row.get("aliases") or "").strip(),
                "kcal_100g": row["kcal_100g"],
                "protein_100g": row["protein_100g"],
                "carbs_100g": row["carbs_100g"],
                "fat_100g": row["fat_100g"],
                "default_grams": row["default_grams"],
                "density_g_per_ml": (row.get("density_g_per_ml") or "").strip(),
                "source": row["source"].strip(),
                "note": (row.get("note") or "").strip(),
            }
        )
        for unit in USER_UNITS:
            column = f"grams_per_{unit.value}"
            if (row.get(column) or "").strip():
                problems += _check_number(row, column, where, allow_zero=False)
                portion_rows.append(
                    {
                        "food_id": food_id,
                        "unit": unit.value,
                        "grams": row[column].strip(),
                        "source": f"user: {row['source'].strip()}",
                    }
                )
    if problems:
        raise FoodTableError(problems)
    return food_rows, portion_rows


def _check_number(
    row: dict[str, str], column: str, where: str, allow_zero: bool
) -> list[str]:
    value = (row.get(column) or "").strip()
    try:
        number = float(value)
    except ValueError:
        return [f"{where}: {column} must be a number (got {value!r})"]
    if number < 0 or (number == 0 and not allow_zero):
        limit = ">= 0" if allow_zero else "> 0"
        return [f"{where}: {column} must be {limit} (got {value})"]
    return []


def _names(row: dict[str, str]) -> list[str]:
    aliases = [a for a in (row.get("aliases") or "").split("|") if a]
    return [row["name_en"], row["name_tr"], *aliases]


def _to_food(row: dict[str, str]) -> Food:
    density = row.get("density_g_per_ml") or ""
    source = row.get("source") or (
        f"USDA SR Legacy, fdc_id {row['fdc_id']}" if row.get("fdc_id") else "unknown"
    )
    return Food(
        id=row["id"],
        name_en=row["name_en"],
        name_tr=row["name_tr"],
        kcal_100g=float(row["kcal_100g"]),
        protein_100g=float(row["protein_100g"]),
        carbs_100g=float(row["carbs_100g"]),
        fat_100g=float(row["fat_100g"]),
        default_grams=float(row["default_grams"]),
        density_g_per_ml=float(density) if density else None,
        source=source,
        note=row.get("note", ""),
        count_as_portion=(row.get("bare_count") or "portion") == "portion",
    )
