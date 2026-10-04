"""Tests for the public Analyzer API (bundled table, fake parser, no model)."""

import json

import pytest

import astra_nutrition.analyzer as analyzer_module
from astra_nutrition import Analyzer, FoodTable, ItemStatus
from astra_nutrition.parser import MealParser, ParseStatus
from tests.test_parser import FakeLlm

ANALYZER = Analyzer()


def test_statuses_grams_and_totals() -> None:
    result = ANALYZER.analyze_items(
        [
            ("Yumurta", "2"),
            ("Pilav", "biraz"),
            ("Kokoreç", "1 dilim"),
            ("Muz", "bir tutam"),
        ]
    )
    rows = [(i.name, i.status, i.grams) for i in result.items]
    assert rows == [
        ("Yumurta", ItemStatus.OK, 100.0),  # 2 x 50 g
        ("Pilav", ItemStatus.ESTIMATED, 150.0),  # vague -> default portion
        ("Kokoreç", ItemStatus.UNMATCHED, None),  # not in the table
        ("Muz", ItemStatus.AMOUNT_UNKNOWN, None),  # unreadable amount
    ]
    totals = result.totals
    assert totals.kcal == 338.0  # 143 + 195: unmatched and unknown not counted
    assert (totals.counted, totals.estimated, totals.unmatched) == (2, 1, 1)
    assert totals.amount_unknown == 1
    assert totals.includes_estimates is True
    assert totals.complete is False


def test_amount_that_repeats_the_item_name_gives_the_same_grams() -> None:
    result = ANALYZER.analyze_items([("Muz", "1"), ("Muz", "1 muz")])
    assert [(i.status, i.grams) for i in result.items] == [
        (ItemStatus.OK, 118.0),
        (ItemStatus.OK, 118.0),
    ]


def test_unmatched_item_keeps_its_best_candidate_but_no_nutrition() -> None:
    item = ANALYZER.analyze_items([("Kokoreç", "1 dilim")]).items[0]
    assert item.nutrition is None
    assert item.food_id is None
    assert item.best_candidate_food_id is not None  # visible, never counted


def test_complete_meal_without_estimates() -> None:
    totals = ANALYZER.analyze_items([("Tavuk Göğsü", "200 g")]).totals
    assert (totals.complete, totals.includes_estimates) == (True, False)
    assert totals.kcal == 330.0


def test_empty_meal() -> None:
    totals = ANALYZER.analyze_items([]).totals
    assert (totals.kcal, totals.items, totals.counted) == (0.0, 0, 0)


def test_model_added_weight_is_used_only_if_the_user_wrote_it() -> None:
    invented = ANALYZER.analyze_items(
        [("Beyaz Peynir", "1 dilim (30g)")], meal_text="1 dilim beyaz peynir"
    )
    written = ANALYZER.analyze_items(
        [("Beyaz Peynir", "1 dilim (40g)")], meal_text="1 dilim (40 g) beyaz peynir"
    )
    assert invented.items[0].grams == 30.0  # table slice, the note is ignored
    assert written.items[0].grams == 40.0  # grounded note
    assert written.items[0].amount_detail == "mass from grounded note"


def test_totals_are_rounded_once_after_summing() -> None:
    row = {
        "id": "tiny",
        "name_en": "Tiny food",
        "name_tr": "Minik",
        "aliases": "",
        "kcal_100g": "16",
        "protein_100g": "0",
        "carbs_100g": "0",
        "fat_100g": "0",
        "default_grams": "1",
        "density_g_per_ml": "",
        "note": "",
    }
    analyzer = Analyzer(table=FoodTable([row], []))
    result = analyzer.analyze_items([("Minik", "1 g")] * 3)
    assert [i.nutrition.kcal for i in result.items] == [0.2, 0.2, 0.2]  # 0.16 each
    assert result.totals.kcal == 0.5  # round(0.48), not 0.2 + 0.2 + 0.2 = 0.6


def test_analyze_uses_the_parser_and_reports_its_status() -> None:
    fake = FakeLlm(
        '{"items": [{"name": "Muz", "amount": "1"}, {"name": "", "amount": "2"}]}'
    )
    analyzer = Analyzer(parser=MealParser(fake))
    result = analyzer.analyze("1 muz")
    assert result.parse_status == ParseStatus.PARTIAL
    assert len(result.rejected_items) == 1  # the blank name, reported
    assert result.items[0].food_id == "banana"
    assert result.items[0].grams == 118.0  # 1 medium banana
    assert (result.totals.rejected, result.totals.complete) == (1, False)


def test_rejected_food_makes_the_total_incomplete() -> None:
    fake = FakeLlm(
        '{"items": [{"name": "Muz", "amount": "1"},'
        ' {"name": "Ekmek", "amount": "1 dilim"}]}'
    )
    result = Analyzer(parser=MealParser(fake)).analyze("1 muz")
    assert [i.name for i in result.items] == ["Muz"]  # every item counted...
    assert result.totals.kcal == 105.0
    assert result.totals.rejected == 1  # ...but the invented bread is reported
    assert result.totals.complete is False


def test_parser_is_loaded_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_download() -> None:
        raise AssertionError("model requested")

    monkeypatch.setattr(analyzer_module, "default_model_path", no_download)
    analyzer = Analyzer()  # must not download anything
    analyzer.analyze_items([("Muz", "1")])  # no parser needed either
    with pytest.raises(AssertionError, match="model requested"):
        _ = analyzer.parser


class KeywordEmbedder:
    """Tiny deterministic embedder: 'yo...' texts point one way, others another."""

    model_name = "keyword"
    signature = "keyword"
    dimension = 2

    def embed(self, texts):
        import numpy as np

        return np.array(
            [[1.0, 0.0] if "yo" in t.lower() else [0.0, 1.0] for t in texts],
            dtype=np.float32,
        )


class FirstCandidateProvider:
    """An LlmProvider of our own: always picks the first candidate."""

    name = "mine"
    model = "rule"

    def complete_json(self, system: str, user: str) -> str:
        first = user.split("- id: ")[1].split(" ")[0]
        return json.dumps({"food_id": first})


def test_with_judge_accepts_a_custom_provider_and_embedder() -> None:
    analyzer = Analyzer.with_judge(FirstCandidateProvider(), embedder=KeywordEmbedder())
    item = analyzer.analyze_items([("Yohurt", "1 kase")]).items[0]
    assert item.match_method == "llm"
    assert item.food_id is not None and "yogurt" in item.food_id


def test_with_judge_needs_a_model_name_for_named_providers() -> None:
    with pytest.raises(ValueError, match="model is required"):
        Analyzer.with_judge("groq", api_key="k", embedder=KeywordEmbedder())


def test_with_judge_explains_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"astra-nutrition\[judge\]"):
        Analyzer.with_judge("groq", api_key="k", model="m")


def test_amount_the_user_never_wrote_is_an_estimate() -> None:
    invented = ANALYZER.analyze_items(
        [("Köfte", "2 adet")], meal_text="çoban salatası ve köfte"
    ).items[0]
    written = ANALYZER.analyze_items([("Köfte", "2 adet")], meal_text="2 köfte").items[
        0
    ]
    assert (invented.status, invented.grams) == (ItemStatus.ESTIMATED, 180.0)
    assert (
        invented.amount_detail == "'2 adet' is not in the meal text -> default portion"
    )
    assert (written.status, written.grams) == (ItemStatus.OK, 60.0)


def test_a_count_of_us_sized_dishes_is_unknown_but_local_pieces_count() -> None:
    items = ANALYZER.analyze_items(
        [("Baklava", "2"), ("Lahmacun", "2")], meal_text="2 baklava, 2 lahmacun"
    ).items
    assert [(i.status, i.grams) for i in items] == [
        (ItemStatus.AMOUNT_UNKNOWN, None),  # FNDDS: one piece is 80 g in the US
        (ItemStatus.OK, 250.0),  # recipe: a piece is 1/8 of the batch
    ]


@pytest.mark.parametrize(
    ("name", "food_id"),
    [
        # False matches found in the v0.2 dev meals (raw lentils, raw pepper)
        ("Süzme Mercimek", "mercimek_corbasi"),
        ("Biber Dolma", "stuffed_pepper_meat"),
        # Misses found there
        ("Fıstıklı Baklava", "baklava"),
        ("Soğuk Ayran", "ayran"),
        ("Tabule Salatası", "tabbouleh"),
        ("Zeytin", "black_olives"),
        ("Tavuk Izgara", "chicken_breast"),
    ],
)
def test_v02_dev_names_match_the_right_dish(name: str, food_id: str) -> None:
    assert ANALYZER.analyze_items([(name, "")]).items[0].food_id == food_id


def _parsed(*items: tuple[str, str]) -> Analyzer:
    """An analyzer whose fake parser returns these (name, amount) items."""
    output = json.dumps({"items": [{"name": n, "amount": a} for n, a in items]})
    return Analyzer(parser=MealParser(FakeLlm(output)))


def test_foods_listed_without_a_conjunction_are_split() -> None:
    result = _parsed(("Kofte Pilav", "1 porsiyon")).analyze(
        "aksam 1 porsiyon kofte pilav"
    )
    rows = [(i.name, i.food_id, i.status, i.note) for i in result.items]
    assert rows == [
        ("Kofte", "izgara_kofte", ItemStatus.OK, "split from 'Kofte Pilav'"),
        ("Pilav", "rice", ItemStatus.ESTIMATED, "split from 'Kofte Pilav'"),
    ]


@pytest.mark.parametrize(
    ("name", "food_id"),
    [
        ("Sebzeli Pilav", None),  # "sebzeli" is no food: a different dish
        ("Su Muhallebisi", None),  # "su" is water, "muhallebisi" no food
        ("Kuru Fasulye", "kuru_fasulye"),  # a table name as a whole
    ],
)
def test_names_that_are_not_lists_stay_whole(name: str, food_id: str | None) -> None:
    items = _parsed((name, "")).analyze(name.lower()).items
    assert [(i.name, i.food_id) for i in items] == [(name, food_id)]


@pytest.mark.parametrize(
    ("name", "food_id"),
    [
        # Design V1: a plain name means the documented default kind
        ("Pide", "pide"),  # with minced meat
        ("Gözleme", "gozleme"),  # with cheese
        ("Tost", "kasarli_tost"),
        ("Tavuk Döner", "tavuk_doner"),  # as a dürüm
        ("Döner", "et_doner"),
        ("Gevrek", "simit"),  # the İzmir name
        # another kind is never matched to the default: it stays unmatched
        ("Patatesli Gözleme", None),
        ("Kaşarlı Pide", None),
        ("Etli Kuru Fasulye", None),
    ],
)
def test_dish_kinds_default_or_stay_unmatched(name: str, food_id: str | None) -> None:
    assert ANALYZER.analyze_items([(name, "")]).items[0].food_id == food_id
