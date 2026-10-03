"""Public API: meal text in, foods, grams and nutrition out.

    from astra_nutrition import Analyzer
    result = Analyzer().analyze("2 yumurta, biraz pilav")
    print(result.totals.kcal, result.totals.complete)

Every item gets an explicit status, and the totals say whether they include
estimates or leave items out. Nothing is guessed silently.
"""

import threading
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from astra_nutrition.amounts import (
    AmountKind,
    AmountStatus,
    GramsResult,
    ParsedAmount,
    ground_note_weight,
    parse_amount,
    to_grams,
    ungrounded_numbers,
)
from astra_nutrition.embeddings import Embedder
from astra_nutrition.foods import FoodTable
from astra_nutrition.index.memory import MemoryFoodIndex
from astra_nutrition.matcher import (
    Candidate,
    FoodIndex,
    FoodMatcher,
    Judge,
    MatchConfig,
    MatchMethod,
    Strategy,
)
from astra_nutrition.model import default_model_path
from astra_nutrition.parser import MealParser, ParsedItem, ParseStatus, RejectedItem
from astra_nutrition.text import fold

if TYPE_CHECKING:  # httpx is only installed with the [judge] extra
    from astra_nutrition.judge import LlmProvider

# Chosen on the dev set (lambda = 3): exact -> fuzzy (pg_trgm >= 0.60), with
# the embedding-threshold stage off. See the evaluation in the README.
DEFAULT_MATCH_CONFIG = MatchConfig(
    Strategy.HYBRID, fuzzy_threshold=0.60, embedding_threshold=None
)
# Candidate retrieval for the judge, chosen on the dev set (recall@5 16/16).
# e5 models expect the "query: " prefix on every text.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_PREFIX = "query: "


class ItemStatus(StrEnum):
    """How one item contributes to the totals."""

    OK = "ok"  # food found, grams measured or converted
    ESTIMATED = "estimated"  # food found, vague/missing amount: default portion
    AMOUNT_UNKNOWN = "amount_unknown"  # food found, amount unreadable: not counted
    UNMATCHED = "unmatched"  # food not in the table: not counted


class Nutrition(BaseModel):
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float


class ItemResult(BaseModel):
    """One food item of the meal."""

    name: str
    amount: str
    status: ItemStatus
    food_id: str | None = None
    food_name_en: str | None = None
    food_name_tr: str | None = None
    food_source: str | None = None  # where the nutrition numbers come from
    match_method: MatchMethod | None = None
    match_similarity: float | None = None
    # When unmatched: the closest food we refused (for debugging, never counted).
    best_candidate_food_id: str | None = None
    grams: float | None = None
    amount_status: AmountStatus | None = None
    amount_detail: str | None = None
    nutrition: Nutrition | None = None
    note: str | None = None


class Totals(BaseModel):
    """Sums over counted items (ok + estimated), with honesty flags."""

    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    items: int
    counted: int
    estimated: int
    amount_unknown: int
    unmatched: int
    includes_estimates: bool  # some counted grams are default portions
    complete: bool  # every item is counted and the parser rejected none
    rejected: int = 0  # items the parser checks refused (see rejected_items)


class AnalysisResult(BaseModel):
    meal_text: str | None
    parse_status: ParseStatus | None  # None when items were given directly
    parse_error: str | None = None
    rejected_items: list[RejectedItem] = Field(default_factory=list)
    items: list[ItemResult]
    totals: Totals


class Analyzer:
    """Analyzes meals against a food table. Create once, reuse (thread-safe).

    Defaults: the bundled table, an in-memory index, the dev-tuned matching
    configuration, no LLM judge, and the GGUF parser loaded on first use.
    """

    def __init__(
        self,
        *,
        table: FoodTable | None = None,
        index: FoodIndex | None = None,
        parser: MealParser | None = None,
        model_path: str | Path | None = None,
        embedder: Embedder | None = None,
        judge: Judge | None = None,
        match_config: MatchConfig = DEFAULT_MATCH_CONFIG,
        n_threads: int | None = None,
    ) -> None:
        self.table = table or FoodTable.bundled()
        index = index or MemoryFoodIndex(self.table.rows, embedder)
        self._index = index
        self._matcher = FoodMatcher(index, embedder, match_config, judge=judge)
        self._parser = parser
        self._model_path = Path(model_path) if model_path else None
        self._n_threads = n_threads
        self._parser_lock = threading.Lock()

    @classmethod
    def with_judge(
        cls,
        provider: "str | LlmProvider",
        *,
        model: str | None = None,
        api_key: str | None = None,
        rpm: int | None = None,
        base_url: str | None = None,
        json_mode: bool = True,
        embedder: Embedder | None = None,
        table: FoodTable | None = None,
        model_path: str | Path | None = None,
        n_threads: int | None = None,
    ) -> "Analyzer":
        """An Analyzer whose unresolved items go to an LLM judge.

        Needs: pip install 'astra-nutrition[judge]'. Off unless asked for:
        item names (never the meal text) are sent to ``provider``, which can
        be "groq", "gemini", "openai-compatible" (any OpenAI-compatible URL,
        including a local Ollama server that keeps everything on the
        machine), or your own LlmProvider object.
        """
        try:
            from astra_nutrition.judge import FoodJudge, RateLimiter, build_provider

            if embedder is None:
                from astra_nutrition.embeddings import SentenceTransformerEmbedder

                embedder = SentenceTransformerEmbedder(
                    DEFAULT_EMBEDDING_MODEL, prefix=DEFAULT_EMBEDDING_PREFIX
                )
        except ImportError as exc:
            raise ImportError(
                "The LLM judge needs extra packages: "
                "pip install 'astra-nutrition[judge]'"
            ) from exc

        if isinstance(provider, str):
            if not model:
                raise ValueError("model is required when provider is a name")
            llm = build_provider(
                provider, api_key, model, base_url=base_url, json_mode=json_mode
            )
        else:
            llm = provider
        judge = FoodJudge(llm, RateLimiter(rpm) if rpm else None)
        return cls(
            table=table,
            embedder=embedder,
            judge=judge,
            model_path=model_path,
            n_threads=n_threads,
        )

    @property
    def parser(self) -> MealParser:
        """The meal parser, loaded (and downloaded if needed) on first use."""
        with self._parser_lock:
            if self._parser is None:
                path = self._model_path or default_model_path()
                self._parser = MealParser.from_path(path, n_threads=self._n_threads)
            return self._parser

    def candidates(self, name: str, k: int = 5) -> list[Candidate]:
        """Top-k candidates from every matching stage, for inspection."""
        return self._matcher.candidates(name, k)

    def analyze(self, meal_text: str) -> AnalysisResult:
        """Parse a meal description and compute its nutrition."""
        parsed = self.parser.parse(meal_text)
        items = [split for item in parsed.items for split in self._split_list(item)]
        result = self.analyze_items([item for item, _ in items], meal_text=meal_text)
        for item_result, (_, original) in zip(result.items, items, strict=True):
            if original is not None and item_result.note is None:
                item_result.note = f"split from {original!r}"
        result.parse_status = parsed.status
        result.parse_error = parsed.error
        result.rejected_items = parsed.rejected_items
        if parsed.rejected_items:
            # A rejected item may be a real food: the total leaves it out.
            result.totals.rejected = len(parsed.rejected_items)
            result.totals.complete = False
        return result

    def _split_list(self, item: ParsedItem) -> list[tuple[ParsedItem, str | None]]:
        """Split a name that lists foods without a conjunction ("Kofte Pilav").

        Only when the whole name is not a table name and every part is an
        exact one; "Sebzeli Pilav" stays whole, since "sebzeli" is no food.
        The first part keeps the amount; the others get none (an estimate).
        Returns (item, original name if split) pairs.
        """
        if self._index.exact(fold(item.name)):
            return [(item, None)]
        parts = self._exact_parts(item.name.split())
        if parts is None or len(parts) < 2:
            return [(item, None)]
        return [(ParsedItem(name=parts[0], amount=item.amount), item.name)] + [
            (ParsedItem(name=part, amount=""), item.name) for part in parts[1:]
        ]

    def _exact_parts(self, words: list[str]) -> list[str] | None:
        """Cover the words with exact table names, longest first, or None."""
        parts, start = [], 0
        while start < len(words):
            for end in range(len(words), start, -1):
                part = " ".join(words[start:end])
                if len({hit.food_id for hit in self._index.exact(fold(part))}) == 1:
                    parts.append(part)
                    start = end
                    break
            else:
                return None
        return parts

    def analyze_items(
        self,
        items: Iterable[ParsedItem | tuple[str, str]],
        meal_text: str | None = None,
    ) -> AnalysisResult:
        """Compute nutrition for already-parsed (name, amount) items.

        With ``meal_text``, weights the parser added in parentheses are used
        only if the user actually wrote them.
        """
        results: list[ItemResult] = []
        raw: list[tuple[float, float, float, float]] = []
        for item in items:
            if isinstance(item, tuple):
                item = ParsedItem(name=item[0], amount=item[1])
            result, values = self._analyze_item(item, meal_text)
            results.append(result)
            if values is not None:
                raw.append(values)
        return AnalysisResult(
            meal_text=meal_text,
            parse_status=None,
            items=results,
            totals=_totals(results, raw),
        )

    def _analyze_item(
        self, item: ParsedItem, meal_text: str | None
    ) -> tuple[ItemResult, tuple[float, float, float, float] | None]:
        match = self._matcher.match(item.name)
        if not match.matched or match.food_id is None:
            best = match.best_candidate
            return ItemResult(
                name=item.name,
                amount=item.amount,
                status=ItemStatus.UNMATCHED,
                best_candidate_food_id=best.food_id if best else None,
                note=match.note,
            ), None

        food = self.table.get(match.food_id)
        amount = parse_amount(item.amount, item_name=item.name)
        invented = False
        if meal_text is not None:
            amount = ground_note_weight(amount, meal_text)
            # A quantity the user never wrote is the parser's guess, not a
            # measurement: use the default portion and flag it as an estimate.
            invented = (
                amount.kind == AmountKind.QUANTITY
                and not amount.from_note
                and bool(ungrounded_numbers(item.amount, meal_text))
            )
            if invented:
                amount = ParsedAmount(AmountKind.MISSING, notes=amount.notes)
        grams = to_grams(amount, self.table.portions(food.id))
        if invented:
            grams = GramsResult(
                grams.status,
                grams.grams,
                f"'{item.amount}' is not in the meal text -> default portion",
            )
        result = ItemResult(
            name=item.name,
            amount=item.amount,
            status=ItemStatus.AMOUNT_UNKNOWN,
            food_id=food.id,
            food_name_en=food.name_en,
            food_name_tr=food.name_tr,
            food_source=food.source,
            match_method=match.method,
            match_similarity=match.similarity,
            grams=grams.grams,
            amount_status=grams.status,
            amount_detail=grams.detail,
        )
        if grams.grams is None:
            return result, None

        factor = grams.grams / 100
        values = (
            food.kcal_100g * factor,
            food.protein_100g * factor,
            food.carbs_100g * factor,
            food.fat_100g * factor,
        )
        result.status = (
            ItemStatus.ESTIMATED
            if grams.status == AmountStatus.ASSUMED_PORTION
            else ItemStatus.OK
        )
        result.nutrition = _nutrition(values)
        return result, values


def _nutrition(values: tuple[float, float, float, float]) -> Nutrition:
    kcal, protein, carbs, fat = values
    return Nutrition(
        kcal=round(kcal, 1),
        protein_g=round(protein, 1),
        carbs_g=round(carbs, 1),
        fat_g=round(fat, 1),
    )


def _totals(
    items: list[ItemResult], raw: list[tuple[float, float, float, float]]
) -> Totals:
    # Sum unrounded values, round once at the end (no accumulated rounding).
    sums = [sum(column) for column in zip(*raw, strict=True)] if raw else [0.0] * 4
    summed = _nutrition((sums[0], sums[1], sums[2], sums[3]))
    count = {status: 0 for status in ItemStatus}
    for item in items:
        count[item.status] += 1
    counted = count[ItemStatus.OK] + count[ItemStatus.ESTIMATED]
    return Totals(
        **summed.model_dump(),
        items=len(items),
        counted=counted,
        estimated=count[ItemStatus.ESTIMATED],
        amount_unknown=count[ItemStatus.AMOUNT_UNKNOWN],
        unmatched=count[ItemStatus.UNMATCHED],
        includes_estimates=count[ItemStatus.ESTIMATED] > 0,
        complete=counted == len(items),
    )
