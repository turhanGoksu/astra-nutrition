"""Unit tests for the meal parser wrapper (no real model needed)."""

from typing import Any

import pytest

from astra_nutrition.parser import (
    MealParser,
    ParseStatus,
    looks_merged,
    strip_meal_words,
    ungrounded_words,
    validate_output,
)
from astra_nutrition.prompts import SYSTEM_PROMPT


class FakeLlm:
    """Returns canned completions (or raises) instead of running a model.

    ``content`` may be one string or a list returned in order, one per call.
    """

    def __init__(
        self,
        content: str | list[str] = "",
        finish_reason: str = "stop",
        error: Exception | None = None,
    ) -> None:
        self.contents = [content] if isinstance(content, str) else list(content)
        self.finish_reason = finish_reason
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.contents[min(len(self.calls), len(self.contents)) - 1]
        return {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": self.finish_reason,
                }
            ]
        }


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ('{"items": [{"name": "Muz", "amount": "1"}]}', ParseStatus.SUCCESS),
        ('{"items": []}', ParseStatus.EMPTY),
        (
            '{"items": [{"name": "Muz", "amount": "1"},'
            ' {"name": "", "amount": "2 adet"}]}',
            ParseStatus.PARTIAL,
        ),
        ('{"items": [{"name": "  ", "amount": "2 adet"}]}', ParseStatus.INVALID_OUTPUT),
        ('{"items": [{"name": "Muz"}]}', ParseStatus.INVALID_OUTPUT),
        ("Tabii, işte haftalık diyet listeniz", ParseStatus.INVALID_OUTPUT),
        ('{"foods": []}', ParseStatus.INVALID_OUTPUT),
        ('[{"name": "Muz", "amount": "1"}]', ParseStatus.INVALID_OUTPUT),
    ],
)
def test_validate_output_status(raw_output: str, expected: ParseStatus) -> None:
    assert validate_output(raw_output).status == expected


def test_blank_name_is_rejected_explicitly_not_dropped() -> None:
    result = validate_output(
        '{"items": [{"name": "Muz", "amount": "1"}, {"name": "", "amount": "2 adet"}]}'
    )
    assert [item.name for item in result.items] == ["Muz"]
    assert result.rejected_items[0].raw == {"name": "", "amount": "2 adet"}
    assert "blank" in result.rejected_items[0].reason


def test_empty_amount_is_kept_for_the_normalizer_to_flag() -> None:
    result = validate_output('{"items": [{"name": "Pilav", "amount": ""}]}')
    assert result.status == ParseStatus.SUCCESS
    assert result.items[0].amount == ""


def test_whitespace_is_stripped() -> None:
    result = validate_output('{"items": [{"name": "  Muz ", "amount": " 1 "}]}')
    assert (result.items[0].name, result.items[0].amount) == ("Muz", "1")


def test_parse_uses_model_card_settings_without_grammar_by_default() -> None:
    fake = FakeLlm('{"items": []}')
    MealParser(fake).parse("2 yumurta")

    call = fake.calls[0]
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "2 yumurta"}
    assert call["temperature"] == 0
    assert call["stop"] == ["<|im_end|>"]
    assert call["response_format"] is None


def test_parse_with_grammar_sends_json_schema() -> None:
    fake = FakeLlm('{"items": []}')
    MealParser(fake, use_grammar=True).parse("2 yumurta")

    response_format = fake.calls[0]["response_format"]
    assert response_format["type"] == "json_object"
    assert "items" in response_format["schema"]["properties"]


def test_truncated_output_is_invalid_even_if_it_looks_like_json() -> None:
    fake = FakeLlm('{"items": [{"name": "Muz", "amount": "1"}]}', "length")
    result = MealParser(fake, max_tokens=20).parse("1 muz")
    assert result.status == ParseStatus.INVALID_OUTPUT
    assert "truncated" in (result.error or "")


def test_inference_exception_becomes_error_status() -> None:
    fake = FakeLlm(error=RuntimeError("llama_decode failed"))
    result = MealParser(fake).parse("1 muz")
    assert result.status == ParseStatus.ERROR
    assert "RuntimeError" in (result.error or "")
    assert result.latency_ms is not None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Yoğurt with honey and walnuts", True),
        ("Yulaf Ezmesi Süt İle", True),
        ("tavuk, pilav", True),
        ("Bread & butter", True),
        ("mac and cheese", True),  # flagged; the second parse keeps it whole
        ("Vegetable soup", False),  # "ve" only as a whole word
        ("chicken noodle soup", False),
        ("Kuru Fasulye", False),
    ],
)
def test_looks_merged(name: str, expected: bool) -> None:
    assert looks_merged(name) is expected


MERGED = '{"items": [{"name": "Yoğurt with honey and walnuts", "amount": "1 bowl"}]}'
SPLIT = (
    '{"items": [{"name": "Yoğurt", "amount": "1 adet (150g)"},'
    ' {"name": "Honey", "amount": "1 tane"}, {"name": "Walnuts", "amount": "1 adet"}]}'
)


def test_merged_item_is_split_first_part_keeps_original_amount() -> None:
    fake = FakeLlm([MERGED, SPLIT])
    result = MealParser(fake).parse("1 bowl yoğurt with honey and walnuts")

    assert len(fake.calls) == 2
    assert fake.calls[1]["messages"][1]["content"] == "Yoğurt with honey and walnuts"
    # Invented amounts from the second parse are dropped, not trusted.
    assert [(i.name, i.amount) for i in result.items] == [
        ("Yoğurt", "1 bowl"),
        ("Honey", ""),
        ("Walnuts", ""),
    ]
    assert result.resplits[0].original.name == "Yoğurt with honey and walnuts"
    assert result.resplits[0].names == ["Yoğurt", "Honey", "Walnuts"]


def test_real_dish_with_conjunction_is_kept_whole() -> None:
    dish = '{"items": [{"name": "Mac and Cheese", "amount": "1 plate"}]}'
    fake = FakeLlm([dish, dish])
    result = MealParser(fake).parse("a plate of mac and cheese")
    assert [(i.name, i.amount) for i in result.items] == [("Mac and Cheese", "1 plate")]
    assert result.resplits == []


def test_failed_second_parse_keeps_the_original_item() -> None:
    fake = FakeLlm([MERGED, "not json"])
    result = MealParser(fake).parse("yoğurt with honey and walnuts")
    assert [i.name for i in result.items] == ["Yoğurt with honey and walnuts"]


def test_no_merge_word_means_no_second_call() -> None:
    fake = FakeLlm('{"items": [{"name": "Tavuk Göğsü", "amount": "100g"}]}')
    MealParser(fake).parse("100g tavuk göğsü")
    assert len(fake.calls) == 1


def test_resplit_can_be_disabled() -> None:
    fake = FakeLlm([MERGED, SPLIT])
    parser = MealParser(fake, resplit_merged=False)
    result = parser.parse("yoğurt with honey and walnuts")
    assert len(fake.calls) == 1
    assert len(result.items) == 1


@pytest.mark.parametrize(
    ("name", "meal_text", "missing"),
    [
        ("Kaşar Peyniri", "1 dilim KAŞAR PEYNİRİ", []),  # Turkish case folding
        ("70% dark chocolate", "three squares of 70% dark chocolate", []),
        ("Hava", "bugün hava çok güzel", []),  # grounded; matching finds no food
        ("Elma", "elmas yüzük aldım", ["elma"]),  # whole words only
        ("Yumurta", "kahvaltı yaptım", ["yumurta"]),  # an invented food
        ("lettuce", "tuna salad with letuce", ["lettuce"]),  # the known cost
    ],
)
def test_ungrounded_words(name: str, meal_text: str, missing: list[str]) -> None:
    assert ungrounded_words(name, meal_text) == missing


TWO_ITEMS = (
    '{"items": [{"name": "Yumurta", "amount": "2"},'
    ' {"name": "Ekmek", "amount": "1 dilim"}]}'
)


def test_item_not_in_the_meal_text_is_rejected_loudly() -> None:
    result = MealParser(FakeLlm(TWO_ITEMS)).parse("kahvaltıda 2 yumurta yedim")
    assert result.status == ParseStatus.PARTIAL
    assert [i.name for i in result.items] == ["Yumurta"]
    assert result.rejected_items[0].raw == {"name": "Ekmek", "amount": "1 dilim"}
    assert result.rejected_items[0].reason == "not in the meal text: ekmek"


def test_only_invented_items_make_the_parse_invalid() -> None:
    output = '{"items": [{"name": "Yumurta", "amount": "1 porsiyon"}]}'
    result = MealParser(FakeLlm(output)).parse("kahvaltı yaptım")
    assert result.status == ParseStatus.INVALID_OUTPUT
    assert result.items == []
    assert result.error == "no item name appears in the meal text"


def test_names_from_a_second_parse_are_checked_too() -> None:
    invented = (
        '{"items": [{"name": "Yoğurt", "amount": "1"},'
        ' {"name": "Banana", "amount": "1"}]}'
    )
    result = MealParser(FakeLlm([MERGED, invented])).parse("yoğurt with honey")
    assert [i.name for i in result.items] == ["Yoğurt"]
    assert [r.raw["name"] for r in result.rejected_items] == ["Banana"]


def test_grounding_can_be_disabled() -> None:
    result = MealParser(FakeLlm(TWO_ITEMS), check_grounding=False).parse("2 yumurta")
    assert result.status == ParseStatus.SUCCESS
    assert [i.name for i in result.items] == ["Yumurta", "Ekmek"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Akşam Biber Dolması", "Biber Dolması"),
        ("Oglen Falafel Wrap", "Falafel Wrap"),
        ("Tabule Yedim", "Tabule"),
        ("Ara Öğün Badem", "Badem"),
        ("Sabah Menemen Ekmek Bandim", "Menemen Ekmek"),
        ("Öğle Yemeği", ""),
        ("Gece Yarısı Çorbası", "Yarısı Çorbası"),  # only leading words go
        ("Mercimek Çorbası", "Mercimek Çorbası"),
    ],
)
def test_strip_meal_words(name: str, expected: str) -> None:
    assert strip_meal_words(name) == expected


def test_meal_time_words_leave_the_name_and_a_meal_word_alone_is_rejected() -> None:
    output = (
        '{"items": [{"name": "Akşam Falafel", "amount": "2"},'
        ' {"name": "Öğle Yemeği", "amount": ""}]}'
    )
    result = MealParser(FakeLlm(output)).parse("öğle yemeği: akşam 2 falafel")
    assert [i.name for i in result.items] == ["Falafel"]
    assert result.rejected_items[0].reason == "only meal-time words, no food"
    assert result.status == ParseStatus.PARTIAL
