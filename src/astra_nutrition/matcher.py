"""Food matching: map a parsed item name to a food in the nutrition table.

Three strategies share one interface so they can be evaluated on the same
labeled data (Step 7):
- Design A (EXACT):     folded name equals a folded alias;
- Design B (EMBEDDING): nearest alias by cosine similarity >= threshold;
- Design C (HYBRID):    exact -> fuzzy (pg_trgm) -> embedding, each gated.

Optionally (Step 8) Design C replaces its embedding threshold with an LLM
judge: embeddings retrieve the top candidates, the judge decides whether one
of them is the same food or none.

Below the thresholds, or when the judge says none or fails, an item is
UNMATCHED. We never fall back to the nearest neighbor: a wrong food is
silent wrong data, an unmatched item is visible.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from astra_nutrition.embeddings import Embedder
from astra_nutrition.text import fold

# Words that say how a food is served, not which food it is: a fuzzy match may
# add them to an alias ("Tavuk Göğsü Izgara"). Folded.
SERVING_WORDS = frozenset(
    {
        "izgara", "haslanmis", "haslama", "firinda", "firin", "taze", "soguk",
        "sicak", "sade", "grilled", "boiled", "baked", "fresh", "cold", "hot",
        "plain",
    }
)  # fmt: skip


_WORD = re.compile(r"[a-z]+")


def added_words(name: str, alias: str) -> list[str]:
    """Words a name adds to an alias it fully contains, serving words aside.

    "falafel wrap" adds "wrap" to "falafel", "etli kuru fasulye" adds "etli" to
    "kuru fasulye": probably a different dish, so a fuzzy match is refused.
    """
    # Letters only: "70%" in "70% dark chocolate" is not another dish.
    name_words, alias_words = _WORD.findall(fold(name)), _WORD.findall(fold(alias))
    if not set(alias_words) <= set(name_words):
        return []
    return [w for w in name_words if w not in alias_words and w not in SERVING_WORDS]


class Strategy(StrEnum):
    """Matching strategy (the design labels used in the evaluation)."""

    EXACT = "exact"  # Design A
    EMBEDDING = "embedding"  # Design B
    HYBRID = "hybrid"  # Design C


class MatchMethod(StrEnum):
    """Which stage produced a match."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    EMBEDDING = "embedding"
    LLM = "llm"  # embedding candidate confirmed by the LLM judge


@dataclass(frozen=True)
class Candidate:
    """One possible food for a query, with its score from one stage."""

    food_id: str
    alias: str
    similarity: float
    method: MatchMethod


@dataclass(frozen=True)
class FoodDetails:
    """What the LLM judge sees about a candidate food."""

    food_id: str
    name_en: str
    name_tr: str
    kcal_100g: float


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's decision: a candidate food id, "none" (None), or an error."""

    food_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one item name."""

    query: str
    matched: bool
    food_id: str | None = None
    alias: str | None = None
    similarity: float | None = None
    method: MatchMethod | None = None
    # When unmatched: the closest candidate we refused (logged to reveal
    # coverage gaps). Never used for nutrition.
    best_candidate: Candidate | None = None
    note: str | None = None  # e.g. why the judge path failed


class FoodIndex(Protocol):
    """Lookups over food aliases (Postgres in production, a fake in tests)."""

    def exact(self, folded: str) -> list[Candidate]: ...

    def fuzzy(self, folded: str, k: int) -> list[Candidate]: ...

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]: ...

    def details(self, food_ids: list[str]) -> dict[str, FoodDetails]: ...


class Judge(Protocol):
    """Decides whether one of the candidate foods is the same as the item."""

    def choose(self, item_name: str, candidates: list[FoodDetails]) -> JudgeVerdict: ...


@dataclass(frozen=True)
class MatchConfig:
    """Strategy and thresholds. Thresholds must be tuned on the dev set only."""

    strategy: Strategy = Strategy.HYBRID
    fuzzy_threshold: float = 0.5
    # None (or anything above 1.0) turns the embedding-threshold stage off.
    embedding_threshold: float | None = 0.9
    judge_candidates: int = 5  # distinct foods shown to the judge

    @property
    def embedding_stage_on(self) -> bool:
        if self.strategy == Strategy.EXACT:
            return False
        return self.embedding_threshold is not None and self.embedding_threshold <= 1.0


class FoodMatcher:
    """Matches item names to foods with one of the three strategies."""

    def __init__(
        self,
        index: FoodIndex,
        embedder: Embedder | None,
        config: MatchConfig,
        judge: Judge | None = None,
    ) -> None:
        needs_embedder = judge is not None or config.embedding_stage_on
        if needs_embedder and embedder is None:
            raise ValueError(
                "this configuration needs an embedder (the judge or the embedding "
                "stage is on): pip install 'astra-nutrition[judge]'"
            )
        self._index = index
        self._embedder = embedder
        self._judge = judge
        self.config = config

    def match(self, name: str) -> MatchResult:
        """Return the matched food, or an explicit unmatched result."""
        folded = fold(name)
        if not folded:
            return MatchResult(query=name, matched=False)
        strategy = self.config.strategy

        if strategy in (Strategy.EXACT, Strategy.HYBRID):
            hits = self._index.exact(folded)
            if len({hit.food_id for hit in hits}) == 1:
                return _matched(name, hits[0])
            if hits:  # one name pointing to two foods: refuse, never guess
                return MatchResult(query=name, matched=False, best_candidate=hits[0])
            if strategy == Strategy.EXACT:
                return MatchResult(query=name, matched=False)

        best: Candidate | None = None
        note: str | None = None
        if strategy == Strategy.HYBRID:
            fuzzy = self._index.fuzzy(folded, k=1)
            if fuzzy:
                best = fuzzy[0]
                if best.similarity >= self.config.fuzzy_threshold:
                    extra = added_words(name, best.alias)
                    if not extra:
                        return _matched(name, best)
                    note = f"adds {' '.join(extra)!r} to {best.alias!r}: another dish?"
            if self._judge is not None:
                return self._ask_judge(name)

        if not self.config.embedding_stage_on:
            return MatchResult(
                query=name, matched=False, best_candidate=best, note=note
            )
        nearest = self._index.nearest(self._embed(name), k=1)
        if nearest:
            best = nearest[0]
            assert self.config.embedding_threshold is not None
            if best.similarity >= self.config.embedding_threshold:
                return _matched(name, best)
        return MatchResult(query=name, matched=False, best_candidate=best, note=note)

    def _embed(self, name: str) -> np.ndarray:
        assert self._embedder is not None  # checked in __init__
        return self._embedder.embed([name])[0]

    def _ask_judge(self, name: str) -> MatchResult:
        """Retrieve distinct candidate foods and let the judge decide."""
        assert self._judge is not None
        k = self.config.judge_candidates
        by_food: dict[str, Candidate] = {}
        for cand in self._index.nearest(self._embed(name), k=4 * k):
            by_food.setdefault(cand.food_id, cand)  # keep each food's best alias
            if len(by_food) == k:
                break
        if not by_food:
            return MatchResult(query=name, matched=False)
        best = next(iter(by_food.values()))

        details = self._index.details(list(by_food))
        verdict = self._judge.choose(name, [details[f] for f in by_food])
        if verdict.food_id is not None and verdict.food_id not in by_food:
            # An id we never offered is a hallucination: refuse it.
            verdict = JudgeVerdict(None, f"unknown food id {verdict.food_id!r}")
        if verdict.error is not None:
            return MatchResult(
                query=name,
                matched=False,
                best_candidate=best,
                note=f"judge error: {verdict.error}",
            )
        if verdict.food_id is None:
            return MatchResult(query=name, matched=False, best_candidate=best)
        chosen = by_food[verdict.food_id]
        return MatchResult(
            query=name,
            matched=True,
            food_id=chosen.food_id,
            alias=chosen.alias,
            similarity=chosen.similarity,
            method=MatchMethod.LLM,
        )

    def candidates(self, name: str, k: int = 5) -> list[Candidate]:
        """Top-k candidates from every stage, for inspection and debugging."""
        folded = fold(name)
        if not folded:
            return []
        found = [*self._index.exact(folded), *self._index.fuzzy(folded, k)]
        if self._embedder is not None:
            found += self._index.nearest(self._embed(name), k)
        return found


def _matched(query: str, candidate: Candidate) -> MatchResult:
    return MatchResult(
        query=query,
        matched=True,
        food_id=candidate.food_id,
        alias=candidate.alias,
        similarity=candidate.similarity,
        method=candidate.method,
    )
