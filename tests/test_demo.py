"""Tests for the demo's table and totals (no Gradio, no model)."""

from astra_nutrition import Analyzer
from astra_nutrition.parser import MealParser
from demo.app import result_rows, summary
from tests.test_parser import FakeLlm

ANALYZER = Analyzer()


def test_rows_show_every_item_and_dashes_for_what_is_not_counted() -> None:
    result = ANALYZER.analyze_items(
        [("Yumurta", "2"), ("Pilav", "biraz"), ("Kokoreç", "1 dilim")]
    )
    assert result_rows(result) == [
        ["Yumurta", "2", "ok", "Egg", 100.0, 143.0],
        ["Pilav", "biraz", "estimated*", "Cooked white rice", 150.0, 195.0],
        ["Kokoreç", "1 dilim", "unmatched", "-", "-", "-"],
    ]


def test_summary_says_what_is_estimated_and_what_is_left_out() -> None:
    text = summary(ANALYZER.analyze_items([("Pilav", "biraz"), ("Kokoreç", "1 dilim")]))
    assert "**Total: 195 kcal**" in text
    assert "1 item(s) use a default portion" in text
    assert "1 unmatched (not in the food table yet)" in text


def test_summary_of_a_complete_meal_has_no_caveats() -> None:
    text = summary(ANALYZER.analyze_items([("Tavuk Göğsü", "200 g")]))
    assert text.startswith("**Total: 330 kcal**")
    assert "estimate" not in text and "Not counted" not in text


def test_summary_without_items_shows_the_parser_status() -> None:
    assert summary(ANALYZER.analyze_items([])).startswith("No food items found")


def test_summary_names_rejected_items() -> None:
    fake = FakeLlm(
        '{"items": [{"name": "Muz", "amount": "1"}, {"name": "Ekmek", "amount": "1"}]}'
    )
    result = Analyzer(parser=MealParser(fake)).analyze("1 muz")
    text = summary(result)
    assert "1 rejected by the parser checks" in text
    assert "Rejected: `Ekmek` (not in the meal text: ekmek)" in text
