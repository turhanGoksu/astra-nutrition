"""Unit tests for the matching strategies (fake index, no DB or model)."""

from collections.abc import Sequence

import numpy as np
import pytest

from astra_nutrition.matcher import (
    Candidate,
    FoodDetails,
    FoodMatcher,
    JudgeVerdict,
    MatchConfig,
    MatchMethod,
    Strategy,
    added_words,
)


class FakeEmbedder:
    """Encodes each text as its position in a list, so the fake index can
    look up which text a vector came from."""

    model_name = "fake"
    signature = "fake|prefix=''|lowercase=True"
    dimension = 1

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            self.texts.append(text)
            vectors.append([len(self.texts) - 1])
        return np.array(vectors, dtype=np.float32)


class FakeIndex:
    """Preset candidates per query, in the shape PgFoodIndex returns."""

    def __init__(
        self,
        embedder: FakeEmbedder,
        exact: dict[str, list[tuple[str, str]]] | None = None,
        fuzzy: dict[str, list[tuple[str, str, float]]] | None = None,
        nearest: dict[str, list[tuple[str, str, float]]] | None = None,
    ) -> None:
        self._embedder = embedder
        self._exact = exact or {}
        self._fuzzy = fuzzy or {}
        self._nearest = nearest or {}

    def exact(self, folded: str) -> list[Candidate]:
        return [
            Candidate(f, a, 1.0, MatchMethod.EXACT)
            for f, a in self._exact.get(folded, [])
        ]

    def fuzzy(self, folded: str, k: int) -> list[Candidate]:
        return [
            Candidate(f, a, s, MatchMethod.FUZZY)
            for f, a, s in self._fuzzy.get(folded, [])[:k]
        ]

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        text = self._embedder.texts[int(vector[0])]
        return [
            Candidate(f, a, s, MatchMethod.EMBEDDING)
            for f, a, s in self._nearest.get(text, [])[:k]
        ]

    def details(self, food_ids: list[str]) -> dict[str, FoodDetails]:
        return {f: FoodDetails(f, f.title(), f, 100.0) for f in food_ids}


class FakeJudge:
    """Returns a preset verdict and records what it was shown."""

    def __init__(self, verdict: JudgeVerdict) -> None:
        self.verdict = verdict
        self.seen: list[tuple[str, list[str]]] = []

    def choose(self, item_name: str, candidates: list[FoodDetails]) -> JudgeVerdict:
        self.seen.append((item_name, [c.food_id for c in candidates]))
        return self.verdict


def make_matcher(
    strategy: Strategy, judge: FakeJudge | None = None, **index_data: dict
) -> FoodMatcher:
    embedder = FakeEmbedder()
    index = FakeIndex(embedder, **index_data)
    config = MatchConfig(strategy, fuzzy_threshold=0.5, embedding_threshold=0.93)
    return FoodMatcher(index, embedder, config, judge=judge)


# Real scores measured on our table (see Step 6 discussion).
DATA = {
    "exact": {"tavuk gogsu": [("chicken_breast", "Tavuk göğsü")]},
    "fuzzy": {
        "tavuk gosu": [("chicken_breast", "tavuk gogsu", 0.64)],
        "mercimek corbasi": [("lentils", "mercimek", 0.53)],
        "baklava": [("honey", "bal", 0.20)],
    },
    "nearest": {
        "tavuk gösü": [("chicken_breast", "Tavuk göğsü", 0.80)],
        "mercimek çorbası": [("tomato_soup", "Domates çorbası", 0.857)],
        "baklava": [("couscous", "Kuskus", 0.929)],
        "chicken breast fillet": [("chicken_breast", "Chicken breast", 0.95)],
    },
}


def test_design_c_exact_stage_wins_first() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("Tavuk Göğsü")
    assert (result.matched, result.food_id, result.method) == (
        True,
        "chicken_breast",
        MatchMethod.EXACT,
    )


def test_design_c_fuzzy_catches_typos() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("tavuk gösü")
    assert (result.food_id, result.method, result.similarity) == (
        "chicken_breast",
        MatchMethod.FUZZY,
        0.64,
    )


def test_fuzzy_match_that_adds_a_dish_word_is_refused() -> None:
    # "mercimek çorbası" contains the alias "mercimek" (lentils) and adds
    # "çorbası": soup is another dish, so the fuzzy match is refused (v0.2).
    result = make_matcher(Strategy.HYBRID, **DATA).match("mercimek çorbası")
    assert result.matched is False
    assert result.note == "adds 'corbasi' to 'mercimek': another dish?"


@pytest.mark.parametrize(
    ("name", "alias", "added"),
    [
        ("Falafel Wrap", "Falafel", ["wrap"]),
        ("Etli Kuru Fasulye", "kuru fasulye", ["etli"]),
        ("Tavuk Göğsü Izgara", "Tavuk göğsü", []),  # a serving word
        ("70% dark chocolate", "Dark chocolate", []),  # not a word
        ("Mercimek Corba", "Mercimek çorbası", []),  # does not contain the alias
    ],
)
def test_added_words(name: str, alias: str, added: list[str]) -> None:
    assert added_words(name, alias) == added


def test_design_c_below_both_thresholds_is_unmatched_with_best_candidate() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("baklava")
    assert result.matched is False
    assert result.food_id is None
    assert result.best_candidate is not None
    assert result.best_candidate.food_id == "couscous"  # logged, never used


def test_design_c_embedding_stage_matches_above_threshold() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("chicken breast fillet")
    assert (result.food_id, result.method) == ("chicken_breast", MatchMethod.EMBEDDING)


def test_design_a_never_uses_fuzzy_or_embedding() -> None:
    matcher = make_matcher(Strategy.EXACT, **DATA)
    assert matcher.match("Tavuk Göğsü").matched is True
    assert matcher.match("tavuk gösü").matched is False
    assert matcher.match("tavuk gösü").best_candidate is None


def test_design_b_skips_exact_and_fuzzy() -> None:
    matcher = make_matcher(Strategy.EMBEDDING, **DATA)
    result = matcher.match("tavuk gösü")  # fuzzy would match; B ignores it
    assert result.matched is False
    assert result.best_candidate is not None
    assert result.best_candidate.method == MatchMethod.EMBEDDING


def test_one_name_pointing_to_two_foods_is_refused() -> None:
    data = {"exact": {"peynir": [("feta", "peynir"), ("mozzarella", "peynir")]}}
    result = make_matcher(Strategy.HYBRID, **data).match("peynir")
    assert result.matched is False


def test_blank_name_is_unmatched() -> None:
    assert make_matcher(Strategy.HYBRID, **DATA).match("   ").matched is False


# Judge path (Step 8): candidates come from embeddings, the judge decides.
JUDGE_DATA = {
    "nearest": {
        "Kuru Üzüm": [
            ("grapes", "Üzüm", 0.94),
            ("grapes", "grapes", 0.93),  # second alias of the same food
            ("dried_figs", "Kuru incir", 0.90),
        ],
        "Yohurt": [("yogurt", "Yoğurt", 0.92), ("greek_yogurt", "greek yogurt", 0.9)],
    }
}


def test_judge_sees_distinct_foods_and_its_choice_is_used() -> None:
    judge = FakeJudge(JudgeVerdict("yogurt"))
    result = make_matcher(Strategy.HYBRID, judge, **JUDGE_DATA).match("Yohurt")
    assert (result.food_id, result.method, result.similarity) == (
        "yogurt",
        MatchMethod.LLM,
        0.92,
    )
    assert judge.seen == [("Yohurt", ["yogurt", "greek_yogurt"])]


def test_judge_none_keeps_item_unmatched_with_best_candidate() -> None:
    judge = FakeJudge(JudgeVerdict(None))
    result = make_matcher(Strategy.HYBRID, judge, **JUDGE_DATA).match("Kuru Üzüm")
    assert result.matched is False
    assert result.best_candidate is not None
    assert result.best_candidate.food_id == "grapes"
    assert judge.seen[0][1] == ["grapes", "dried_figs"]  # grapes shown once


def test_judge_error_keeps_item_unmatched_with_a_note() -> None:
    judge = FakeJudge(JudgeVerdict(None, "HTTPStatusError: 429"))
    result = make_matcher(Strategy.HYBRID, judge, **JUDGE_DATA).match("Kuru Üzüm")
    assert result.matched is False
    assert "429" in (result.note or "")


def test_judge_id_outside_the_candidates_is_refused() -> None:
    judge = FakeJudge(JudgeVerdict("raisins"))  # never offered: hallucination
    result = make_matcher(Strategy.HYBRID, judge, **JUDGE_DATA).match("Kuru Üzüm")
    assert result.matched is False
    assert "unknown food id" in (result.note or "")


def test_judge_is_not_asked_when_exact_or_fuzzy_already_matched() -> None:
    judge = FakeJudge(JudgeVerdict("grapes"))
    result = make_matcher(Strategy.HYBRID, judge, **DATA).match("tavuk gösü")
    assert result.method == MatchMethod.FUZZY
    assert judge.seen == []
