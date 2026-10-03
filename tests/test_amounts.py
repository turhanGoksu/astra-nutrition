"""Unit tests for amount parsing and gram conversion."""

import pytest

from astra_nutrition.amounts import (
    AmountKind,
    AmountStatus,
    FoodPortions,
    ParsedAmount,
    Unit,
    ground_note_weight,
    parse_amount,
    to_grams,
    ungrounded_numbers,
)

Q = AmountKind.QUANTITY


@pytest.mark.parametrize(
    ("text", "kind", "value", "unit"),
    [
        # Real parser outputs observed in Step 2
        ("100g", Q, 100, Unit.GRAM),
        ("2 pieces", Q, 2, Unit.PIECE),
        ("1", Q, 1, None),
        ("bowl", Q, 1, Unit.BOWL),
        ("two slices", Q, 2, Unit.SLICE),
        ("glass", Q, 1, Unit.GLASS),
        ("1 dilim (30g)", Q, 1, Unit.SLICE),
        ("0.5 adet (30g)", Q, 0.5, Unit.PIECE),
        ("1 cup (240ml)", Q, 1, Unit.CUP),
        ("bir tabak", Q, 1, Unit.PLATE),
        ("200 ml", Q, 200, Unit.MILLILITER),
        ("3 adet", Q, 3, Unit.PIECE),
        # Units and conversions to canonical units
        ("100 gr", Q, 100, Unit.GRAM),
        ("1,5 kg", Q, 1500, Unit.GRAM),
        ("0,5 lt", Q, 500, Unit.MILLILITER),
        ("2 yemek kaşığı", Q, 2, Unit.TABLESPOON),
        ("1 çay kaşığı", Q, 1, Unit.TEASPOON),
        ("1 su bardağı", Q, 1, Unit.GLASS),
        ("3 kaşık", Q, 3, Unit.TABLESPOON),
        ("1 porsiyon", Q, 1, Unit.PORTION),
        ("bir avuç", Q, 1, Unit.HANDFUL),
        # Number words, fractions, Turkish casing
        ("İki Bardak", Q, 2, Unit.GLASS),
        ("iki buçuk dilim", Q, 2.5, Unit.SLICE),
        ("yarım", Q, 0.5, None),
        ("çeyrek", Q, 0.25, None),
        ("1/2 kase", Q, 0.5, Unit.BOWL),
        ("½ bardak", Q, 0.5, Unit.GLASS),
        ("half a cup", Q, 0.5, Unit.CUP),
        ("one and a half cups", Q, 1.5, Unit.CUP),
        ("a cup", Q, 1, Unit.CUP),
        ("2 büyük", Q, 2, None),
        ("0", Q, 0, None),
        # Missing / vague
        ("", AmountKind.MISSING, None, None),
        ("   ", AmountKind.MISSING, None, None),
        ("biraz", AmountKind.VAGUE, None, None),
        ("some", AmountKind.VAGUE, None, None),
        ("a few", AmountKind.VAGUE, None, None),
        ("birkaç dilim", AmountKind.VAGUE, None, Unit.SLICE),
        # Unparseable: never guess
        ("1-2 dilim", AmountKind.UNPARSEABLE, None, None),
        ("1 dilim 30 g", AmountKind.UNPARSEABLE, None, None),
        ("bir tutam", AmountKind.UNPARSEABLE, None, None),
        ("2 fincan", AmountKind.UNPARSEABLE, None, None),
        ("1/0", AmountKind.UNPARSEABLE, None, None),
        ("çok yorgunum, hiçbir şey yiyemedim.", AmountKind.UNPARSEABLE, None, None),
    ],
)
def test_parse_amount(
    text: str, kind: AmountKind, value: float | None, unit: Unit | None
) -> None:
    parsed = parse_amount(text)
    assert (parsed.kind, parsed.value, parsed.unit) == (kind, value, unit)


def test_parenthetical_weight_is_kept_as_note_but_not_used() -> None:
    parsed = parse_amount("1 dilim (30g)")
    assert parsed.notes == ("30g",)
    assert (parsed.value, parsed.unit) == (1, Unit.SLICE)


@pytest.mark.parametrize(
    ("text", "item_name", "kind", "value", "unit"),
    [
        # The model repeats the item's name (seen in the Docker build, Step 12)
        ("1 muz", "Muz", Q, 1, None),
        ("2 rafadan", "Rafadan", Q, 2, None),
        ("15 hazelnuts", "Hazelnuts", Q, 15, None),
        ("1 elma (150g)", "Elma", Q, 1, None),
        ("1 dilim kaşar peyniri", "Kaşar Peyniri", Q, 1, Unit.SLICE),
        # Only the exact name, only at the end, and only after an amount
        ("1 elma", "Muz", AmountKind.UNPARSEABLE, None, None),
        ("muz 1", "Muz", AmountKind.UNPARSEABLE, None, None),
        ("muz", "Muz", AmountKind.UNPARSEABLE, None, None),
        ("bir parça dil peyniri", "Dil Peyniri", AmountKind.UNPARSEABLE, None, None),
    ],
)
def test_parse_amount_drops_the_item_name(
    text: str, item_name: str, kind: AmountKind, value: float | None, unit: Unit | None
) -> None:
    parsed = parse_amount(text, item_name=item_name)
    assert (parsed.kind, parsed.value, parsed.unit) == (kind, value, unit)


def test_item_name_is_kept_without_the_item_name_argument() -> None:
    assert parse_amount("1 muz").kind == AmountKind.UNPARSEABLE


EGG = FoodPortions(default_grams=100, unit_grams={Unit.PIECE: 50})
RICE = FoodPortions(default_grams=150, unit_grams={Unit.PLATE: 200})
AYRAN = FoodPortions(default_grams=200, density_g_per_ml=1.03)


@pytest.mark.parametrize(
    ("text", "portions", "status", "grams"),
    [
        ("100g", RICE, AmountStatus.MEASURED, 100),
        ("2 adet", EGG, AmountStatus.CONVERTED, 100),
        ("2", EGG, AmountStatus.CONVERTED, 100),  # bare count -> pieces
        ("2", RICE, AmountStatus.CONVERTED, 300),  # not countable -> portions
        ("1 tabak", RICE, AmountStatus.CONVERTED, 200),
        ("1,5 porsiyon", RICE, AmountStatus.CONVERTED, 225),
        ("200 ml", AYRAN, AmountStatus.CONVERTED, 206),
        ("1 glass", AYRAN, AmountStatus.CONVERTED, 206),
        ("", RICE, AmountStatus.ASSUMED_PORTION, 150),
        ("biraz", RICE, AmountStatus.ASSUMED_PORTION, 150),
        ("1 dilim", EGG, AmountStatus.UNCONVERTIBLE, None),
        ("200 ml", RICE, AmountStatus.UNCONVERTIBLE, None),  # no density
        ("bir tutam", RICE, AmountStatus.UNPARSEABLE, None),
    ],
)
def test_to_grams(
    text: str, portions: FoodPortions, status: AmountStatus, grams: float | None
) -> None:
    result = to_grams(parse_amount(text), portions)
    assert (result.status, result.grams) == (status, grams)
    assert result.detail


def test_unparseable_never_falls_back_to_default_portion() -> None:
    result = to_grams(ParsedAmount(AmountKind.UNPARSEABLE), RICE)
    assert result.grams is None


SIMIT = FoodPortions(default_grams=110, unit_grams={Unit.PIECE: 110})
CHEESE = FoodPortions(default_grams=30, unit_grams={Unit.SLICE: 25})
TEA = FoodPortions(default_grams=200, density_g_per_ml=1.0)


@pytest.mark.parametrize(
    ("meal_text", "amount_text", "portions", "status", "grams"),
    [
        # The model invented "(30g)": ignored, table portion used instead.
        ("kahvaltıda yarım simit yedim", "0.5 adet (30g)", SIMIT, "converted", 55),
        # The user wrote the weight: the note is grounded and used.
        ("1 dilim (30 g) beyaz peynir", "1 dilim (30g)", CHEESE, "measured", 30),
        ("1 dilim beyaz peynir, 30 gr", "1 dilim (30g)", CHEESE, "measured", 30),
        # "300g" and "130g" must not ground a "30g" note.
        ("1 dilim peynir ve 300g yoğurt", "1 dilim (30g)", CHEESE, "converted", 25),
        ("1 dilim peynir ve 130g yoğurt", "1 dilim (30g)", CHEESE, "converted", 25),
        # A number without the same unit is not grounding ("30 dakika").
        ("1 dilim peynir, 30 dakika yürüdüm", "1 dilim (30g)", CHEESE, "converted", 25),
        # Volume notes follow the same rule.
        ("a cup of black tea", "1 cup (240ml)", TEA, "converted", 240),
        ("240 ml black tea", "1 cup (240ml)", TEA, "converted", 240),
    ],
)
def test_ground_note_weight(
    meal_text: str,
    amount_text: str,
    portions: FoodPortions,
    status: str,
    grams: float,
) -> None:
    amount = ground_note_weight(parse_amount(amount_text), meal_text)
    result = to_grams(amount, portions)
    assert (result.status, result.grams) == (status, grams)


def test_grounded_note_is_visible_in_detail() -> None:
    amount = ground_note_weight(parse_amount("1 dilim (30g)"), "1 dilim (30 g) peynir")
    assert amount.from_note
    assert to_grams(amount, CHEESE).detail == "mass from grounded note"


def test_known_limitation_weight_of_another_item_grounds_the_note() -> None:
    # Documented heuristic limit: "30 g" belongs to the olives, not the simit,
    # but string matching cannot tell. Kept as a test so the behavior is explicit.
    amount = ground_note_weight(
        parse_amount("0.5 adet (30g)"), "yarım simit ve 30 g zeytin"
    )
    assert to_grams(amount, SIMIT).grams == 30


@pytest.mark.parametrize(
    ("amount", "meal_text", "missing"),
    [
        ("2 adet", "çoban salatası ve köfte", [2.0]),  # invented by the parser
        ("1 porsiyon (250ml)", "mercimek çorbası", [1.0]),
        ("2 adet", "akşam 2 köfte", []),
        ("2", "iki yumurta", []),  # number words
        ("0.5 adet", "yarım ekmek", []),
        ("2.5 dilim", "iki buçuk dilim ekmek", []),
        ("1.5 cups", "one and a half cups of rice", []),
        ("1 cup", "a cup of milk", []),  # an article means 1
        ("1 kase", "kase mercimek çorbası", []),  # the unit is written
        ("1,5 porsiyon", "1,5 porsiyon pilav", []),
        ("200 gr", "200gr tavuk", []),
        ("biraz", "biraz peynir", []),
    ],
)
def test_ungrounded_numbers(amount: str, meal_text: str, missing: list[float]) -> None:
    assert ungrounded_numbers(amount, meal_text) == missing


def test_bare_count_without_a_trusted_piece_size_is_unknown() -> None:
    us_pieces = FoodPortions(default_grams=80, count_as_portion=False)
    result = to_grams(parse_amount("2"), us_pieces)
    assert (result.status, result.grams) == (AmountStatus.UNCONVERTIBLE, None)
    assert to_grams(parse_amount("200 g"), us_pieces).grams == 200


@pytest.mark.parametrize("text", ["sabah", "Akşam", "yedim", "lunch"])
def test_a_meal_time_or_verb_is_no_amount(text: str) -> None:
    assert parse_amount(text).kind == AmountKind.MISSING
