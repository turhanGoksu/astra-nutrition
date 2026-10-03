"""Deterministic amount normalization: free-text amount -> grams.

Two stages:
1. ``parse_amount`` turns a string into a ``ParsedAmount`` (food-independent).
2. ``to_grams`` converts it using the matched food's ``FoodPortions``.

Policy for unclear amounts (no silent guesses):
- missing or vague ("", "biraz", "some") -> the food's default portion,
  flagged as ``assumed_portion`` so totals can say they include an estimate;
- unparseable (our rules cannot read it) or unconvertible (the food has no
  data for that unit) -> no grams at all, flagged. We never guess there,
  because the user may have given a specific amount we failed to understand.

Weights the model adds in parentheses ("1 dilim (30g)") are the model's own
estimates unless the user wrote them. ``ground_note_weight`` uses them only
when the same number and unit appear in the user's meal text.

The model sometimes repeats the item's name in the amount ("1 muz" for Muz);
whether it does can change with the llama.cpp build, even at temperature 0.
Given the item name, ``parse_amount`` drops that exact trailing name, so the
result does not depend on the machine the model ran on.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from astra_nutrition.text import EATING_VERBS, MEAL_TIME_WORDS, fold


class Unit(StrEnum):
    """Canonical units. Mass is always grams, volume always milliliters."""

    GRAM = "g"
    MILLILITER = "ml"
    PIECE = "piece"
    SLICE = "slice"
    BOWL = "bowl"
    PLATE = "plate"
    GLASS = "glass"
    CUP = "cup"
    TABLESPOON = "tbsp"
    TEASPOON = "tsp"
    PORTION = "portion"
    HANDFUL = "handful"


class AmountKind(StrEnum):
    """What ``parse_amount`` understood."""

    QUANTITY = "quantity"  # a value, optionally with a unit
    MISSING = "missing"  # no amount given
    VAGUE = "vague"  # "biraz", "some", "a few"...
    UNPARSEABLE = "unparseable"  # unknown words or conflicting numbers/units


class AmountStatus(StrEnum):
    """How the grams of an item were obtained."""

    MEASURED = "measured"  # the user gave a mass (g, kg)
    CONVERTED = "converted"  # household unit / volume converted with food data
    ASSUMED_PORTION = "assumed_portion"  # missing/vague -> default portion
    UNCONVERTIBLE = "unconvertible"  # known unit, but no data for this food
    UNPARSEABLE = "unparseable"  # the amount text could not be read


# Vocabularies are written in folded form (see app.text.fold).
# unit alias -> (canonical unit, factor to the canonical unit)
_UNIT_ALIASES: dict[str, tuple[Unit, float]] = {
    **dict.fromkeys(["g", "gr", "gram", "grams", "gramm"], (Unit.GRAM, 1.0)),
    **dict.fromkeys(["kg", "kilo", "kilogram", "kilograms"], (Unit.GRAM, 1000.0)),
    **dict.fromkeys(
        ["ml", "mililitre", "milliliter", "milliliters", "millilitre"],
        (Unit.MILLILITER, 1.0),
    ),
    **dict.fromkeys(
        ["l", "lt", "litre", "liter", "liters", "litres"], (Unit.MILLILITER, 1000.0)
    ),
    **dict.fromkeys(
        ["adet", "tane", "piece", "pieces", "pc", "pcs"], (Unit.PIECE, 1.0)
    ),
    **dict.fromkeys(["dilim", "dilimi", "slice", "slices"], (Unit.SLICE, 1.0)),
    **dict.fromkeys(["kase", "kasesi", "bowl", "bowls"], (Unit.BOWL, 1.0)),
    **dict.fromkeys(["tabak", "tabagi", "plate", "plates"], (Unit.PLATE, 1.0)),
    **dict.fromkeys(["bardak", "bardagi", "glass", "glasses"], (Unit.GLASS, 1.0)),
    **dict.fromkeys(["cup", "cups"], (Unit.CUP, 1.0)),
    # A bare "kaşık" in Turkish usually means a tablespoon ("yemek kaşığı").
    **dict.fromkeys(
        ["kasik", "kasigi", "tbsp", "tablespoon", "tablespoons"], (Unit.TABLESPOON, 1.0)
    ),
    **dict.fromkeys(["tsp", "teaspoon", "teaspoons"], (Unit.TEASPOON, 1.0)),
    **dict.fromkeys(
        ["porsiyon", "portion", "portions", "serving", "servings"], (Unit.PORTION, 1.0)
    ),
    **dict.fromkeys(["avuc", "avucu", "handful", "handfuls"], (Unit.HANDFUL, 1.0)),
}

# Multi-word units, rewritten to a single alias before tokenizing.
_UNIT_PHRASES: dict[str, str] = {
    "yemek kasigi": "tbsp",
    "cay kasigi": "tsp",
    "su bardagi": "glass",
}

_NUMBER_WORDS: dict[str, float] = {
    "bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5,
    "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip
_HALF_WORDS = {"yarim", "half"}  # 0.5 alone, +0.5 after a number
_QUARTER_WORDS = {"ceyrek", "quarter"}
_BUCUK = "bucuk"  # "iki buçuk" = 2.5
_ARTICLES = {"a", "an"}  # imply a value of 1 when no number is given
_VAGUE_WORDS = {"biraz", "az", "bol", "cok", "birkac", "some", "little", "bit", "few"}
_FILLER_WORDS = {
    "of",
    "and",
    "large",
    "medium",
    "small",
    "buyuk",
    "orta",
    "kucuk",
    "boy",
    # "Menemen [sabah]": a meal time or a verb is no amount; the amount is missing.
    *MEAL_TIME_WORDS,
    *EATING_VERBS,
}

_UNICODE_FRACTIONS = {"½": " 0.5 ", "¼": " 0.25 ", "¾": " 0.75 "}
_PARENTHESES = re.compile(r"\(([^)]*)\)")
_DECIMAL_COMMA = re.compile(r"(\d),(\d)")
_TOKEN = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?|[a-z]+")
_MEASURE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]+)")  # a note like "30g", "240 ml"

# Approximate volumes for household units, used only with a known density.
_VOLUME_ML: dict[Unit, float] = {
    Unit.CUP: 240.0,
    Unit.GLASS: 200.0,  # Turkish "su bardağı"
    Unit.TABLESPOON: 15.0,
    Unit.TEASPOON: 5.0,
}


@dataclass(frozen=True)
class ParsedAmount:
    """Food-independent reading of an amount string."""

    kind: AmountKind
    value: float | None = None
    unit: Unit | None = None  # None with a value means a bare count ("2")
    notes: tuple[str, ...] = ()  # text in parentheses, e.g. ("30g",)
    from_note: bool = False  # value/unit taken from a grounded note


@dataclass(frozen=True)
class FoodPortions:
    """Per-food data needed to turn household units into grams."""

    default_grams: float  # one typical serving
    unit_grams: Mapping[Unit, float] = field(default_factory=dict)
    density_g_per_ml: float | None = None  # only for liquids
    # A bare count ("2") without a piece size means that many default portions,
    # unless the default portion is a piece of the wrong size (US FNDDS dishes).
    count_as_portion: bool = True


@dataclass(frozen=True)
class GramsResult:
    """Grams for one item, plus how they were obtained."""

    status: AmountStatus
    grams: float | None
    detail: str


def parse_amount(text: str, item_name: str | None = None) -> ParsedAmount:
    """Parse a free-text amount (Turkish or English) into a ParsedAmount.

    With ``item_name``, the item's own name at the end of the amount is
    dropped ("1 muz" for Muz reads as "1"). Only the exact name is dropped;
    other unknown words still make the amount unparseable.
    """
    notes = tuple(n.strip() for n in _PARENTHESES.findall(text) if n.strip())
    body = _PARENTHESES.sub(" ", text)
    for symbol, replacement in _UNICODE_FRACTIONS.items():
        body = body.replace(symbol, replacement)
    body = _DECIMAL_COMMA.sub(r"\1.\2", fold(body))
    for phrase, alias in _UNIT_PHRASES.items():
        body = re.sub(rf"\b{phrase}\b", alias, body)

    tokens = _TOKEN.findall(body)
    if item_name is not None:
        tokens = _drop_item_name(tokens, item_name)
    if not tokens:
        return ParsedAmount(AmountKind.MISSING, notes=notes)

    unparseable = ParsedAmount(AmountKind.UNPARSEABLE, notes=notes)
    value: float | None = None
    unit: Unit | None = None
    factor = 1.0
    has_article = False
    is_vague = False

    for token in tokens:
        if token[0].isdigit() or token in _NUMBER_WORDS:
            number = _to_number(token)
            if value is not None or number is None:
                return unparseable  # "1-2 dilim", "1/0": do not guess
            value = number
        elif token in _HALF_WORDS:
            value = 0.5 if value is None else value + 0.5
        elif token == _BUCUK:
            if value is None:
                return unparseable
            value += 0.5
        elif token in _QUARTER_WORDS:
            if value is not None:
                return unparseable
            value = 0.25
        elif token in _UNIT_ALIASES:
            if unit is not None:
                return unparseable  # two units: "1 dilim 30 g"
            unit, factor = _UNIT_ALIASES[token]
        elif token in _ARTICLES:
            has_article = True
        elif token in _VAGUE_WORDS:
            is_vague = True
        elif token in _FILLER_WORDS:
            continue
        else:
            return unparseable  # unknown word: never ignore it silently

    if is_vague:
        if value is not None:
            return unparseable  # "2 biraz" makes no sense
        return ParsedAmount(AmountKind.VAGUE, unit=unit, notes=notes)
    if value is None:
        if unit is None and not has_article:
            return ParsedAmount(AmountKind.MISSING, notes=notes)
        value = 1.0  # "bowl", "a cup": one of the unit
    return ParsedAmount(AmountKind.QUANTITY, value * factor, unit, notes)


def ungrounded_numbers(amount_text: str, meal_text: str) -> list[float]:
    """Numbers in an amount that the user never wrote in the meal text.

    The parser sometimes invents a quantity ("2 adet" for a plain "köfte");
    such a number is not a measurement. Numbers are compared by value, so
    "iki", "2" and "two" are the same, "yarım" is 0.5 and "a cup" means 1. A 1
    also counts as written when its unit is in the text ("kase mercimek").
    """
    meal_tokens = _tokens(meal_text)
    written = set(_number_values(meal_tokens))
    if set(meal_tokens) & _ARTICLES:
        written.add(1.0)
    amount_tokens = _tokens(_PARENTHESES.sub(" ", amount_text))
    unit_written = any(t in _UNIT_ALIASES and t in meal_tokens for t in amount_tokens)
    return [
        value
        for value in _number_values(amount_tokens, combine_halves=False)
        if value not in written and not (value == 1.0 and unit_written)
    ]


def ground_note_weight(amount: ParsedAmount, meal_text: str) -> ParsedAmount:
    """Replace the amount with a parenthetical mass/volume the user wrote.

    "0.5 adet (30g)" keeps "0.5 adet" unless "30 g" appears in ``meal_text``.
    This is a string heuristic, not proof: if another item in the same meal
    has exactly that weight, the note is (wrongly) treated as grounded.
    """
    text = _DECIMAL_COMMA.sub(r"\1.\2", fold(meal_text))
    for note in amount.notes:
        match = _MEASURE.fullmatch(_DECIMAL_COMMA.sub(r"\1.\2", fold(note)))
        if match is None:
            continue
        number, alias = match.groups()
        unit_and_factor = _UNIT_ALIASES.get(alias)
        if unit_and_factor is None or unit_and_factor[0] not in (
            Unit.GRAM,
            Unit.MILLILITER,
        ):
            continue
        same_unit = "|".join(
            a for a, uf in _UNIT_ALIASES.items() if uf == unit_and_factor
        )
        pattern = rf"(?<![\d.]){re.escape(number)}\s*(?:{same_unit})\b"
        if re.search(pattern, text):
            unit, factor = unit_and_factor
            return ParsedAmount(
                AmountKind.QUANTITY,
                float(number) * factor,
                unit,
                amount.notes,
                from_note=True,
            )
    return amount


def to_grams(amount: ParsedAmount, portions: FoodPortions) -> GramsResult:
    """Convert a parsed amount to grams using the matched food's portion data."""
    if amount.kind in (AmountKind.MISSING, AmountKind.VAGUE):
        return GramsResult(
            AmountStatus.ASSUMED_PORTION,
            portions.default_grams,
            f"{amount.kind} amount -> default portion",
        )
    if amount.kind == AmountKind.UNPARSEABLE or amount.value is None:
        return GramsResult(AmountStatus.UNPARSEABLE, None, "amount not understood")

    value, unit = amount.value, amount.unit
    if unit == Unit.GRAM:
        detail = "mass from grounded note" if amount.from_note else "mass given"
        return GramsResult(AmountStatus.MEASURED, _round(value), detail)
    if unit is None:
        # A bare count: pieces if the food is countable, otherwise portions.
        if Unit.PIECE in portions.unit_grams:
            unit = Unit.PIECE
        elif portions.count_as_portion:
            unit = Unit.PORTION
        else:
            return GramsResult(
                AmountStatus.UNCONVERTIBLE, None, "a count, but no known piece size"
            )
    if unit == Unit.PORTION:
        return GramsResult(
            AmountStatus.CONVERTED,
            _round(value * portions.default_grams),
            f"{value:g} x default portion",
        )
    if unit in portions.unit_grams:
        return GramsResult(
            AmountStatus.CONVERTED,
            _round(value * portions.unit_grams[unit]),
            f"{value:g} x {unit} ({portions.unit_grams[unit]:g} g)",
        )

    volume_ml = value if unit == Unit.MILLILITER else None
    if unit in _VOLUME_ML:
        volume_ml = value * _VOLUME_ML[unit]
    if volume_ml is not None and portions.density_g_per_ml is not None:
        return GramsResult(
            AmountStatus.CONVERTED,
            _round(volume_ml * portions.density_g_per_ml),
            f"{volume_ml:g} ml x density {portions.density_g_per_ml:g}",
        )
    return GramsResult(
        AmountStatus.UNCONVERTIBLE, None, f"no gram data for unit '{unit}'"
    )


def _tokens(text: str) -> list[str]:
    """Folded tokens, normalized the way parse_amount reads an amount."""
    for symbol, replacement in _UNICODE_FRACTIONS.items():
        text = text.replace(symbol, replacement)
    text = _DECIMAL_COMMA.sub(r"\1.\2", fold(text))
    for phrase, alias in _UNIT_PHRASES.items():
        text = re.sub(rf"\b{phrase}\b", alias, text)
    return _TOKEN.findall(text)


def _number_values(tokens: list[str], combine_halves: bool = True) -> list[float]:
    """Values of the numbers in tokens; with ``combine_halves``, "iki buçuk"
    and "one and a half" also give 2.5 and 1.5."""
    values: list[float] = []
    for i, token in enumerate(tokens):
        if token[0].isdigit() or token in _NUMBER_WORDS:
            number = _to_number(token)
            if number is None:
                continue
            values.append(number)
            following = tokens[i + 1 : i + 4]
            if combine_halves and (_BUCUK in following or _HALF_WORDS & set(following)):
                values.append(number + 0.5)
        elif token in _HALF_WORDS:
            values.append(0.5)
        elif token in _QUARTER_WORDS:
            values.append(0.25)
    return values


def _drop_item_name(tokens: list[str], item_name: str) -> list[str]:
    """Drop the item's name from the end of the tokens if something precedes it.

    An amount that is only the name ("muz") is left as is: there is no amount
    in it to read, and dropping it would turn it into a quiet default portion.
    """
    name = _TOKEN.findall(fold(item_name))
    if name and len(tokens) > len(name) and tokens[-len(name) :] == name:
        return tokens[: -len(name)]
    return tokens


def _to_number(token: str) -> float | None:
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    if "/" in token:
        numerator, denominator = token.split("/")
        return float(numerator) / float(denominator) if float(denominator) else None
    return float(token)


def _round(grams: float) -> float:
    return round(grams, 1)
