"""Meal parser: GGUF model inference with validated JSON output.

Defense in depth:
- optionally, a grammar (built from ``MealParseSchema``) guarantees the SHAPE;
- Pydantic validation checks the MEANING we can check here (e.g. blank names);
- names lose meal-time words the model glued to them ("Akşam Falafel");
- grounding: every word of an item name must appear in the meal text, so an
  invented food is rejected instead of counted;
- every call ends in an explicit ``ParseStatus``, never a silent guess.

Grammar-constrained decoding is OFF by default (provisional decision, small
sample): on 4 real meals it produced identical output at ~50% higher latency,
and on a non-meal input it turned a loud ``invalid_output`` into a silent
``success`` with an invented food. Re-evaluate with a larger parser eval set.
"""

import json
import logging
import re
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from astra_nutrition.prompts import SYSTEM_PROMPT
from astra_nutrition.text import EATING_VERBS, MEAL_TIME_WORDS, fold

logger = logging.getLogger(__name__)


class ParseStatus(StrEnum):
    """Outcome of one parse call."""

    SUCCESS = "success"  # all items valid
    PARTIAL = "partial"  # some items valid, some rejected
    EMPTY = "empty"  # valid output with no items
    INVALID_OUTPUT = "invalid_output"  # not JSON, wrong shape, or no valid item
    ERROR = "error"  # inference raised an exception


class ParsedItem(BaseModel):
    """One food item extracted by the model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    amount: str  # may be empty: amount normalization flags it, never guesses

    # A field_validator (unlike Field(min_length=1)) is NOT part of the JSON
    # schema, so the grammar does not force the model to invent a name.
    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("name must not be blank")
        return value


class MealParseSchema(BaseModel):
    """Expected output shape. Its JSON schema becomes the decoding grammar."""

    items: list[ParsedItem]


class RejectedItem(BaseModel):
    """A raw item from the model output that failed validation."""

    raw: Any
    reason: str


class Resplit(BaseModel):
    """A merged item that a second parse split into several items."""

    original: ParsedItem
    names: list[str]


class ParseResult(BaseModel):
    """Result of parsing one meal description."""

    status: ParseStatus
    items: list[ParsedItem] = Field(default_factory=list)
    rejected_items: list[RejectedItem] = Field(default_factory=list)
    resplits: list[Resplit] = Field(default_factory=list)
    raw_output: str | None = None
    error: str | None = None
    latency_ms: float | None = None


# Words that suggest the model merged several foods into one name, e.g.
# "Yoğurt with honey and walnuts". Matched on folded, whole-word tokens.
_MERGE_WORDS = {"with", "and", "ve", "ile"}
_MERGE_SYMBOLS = ("&", "+", ",")


def looks_merged(name: str) -> bool:
    """True if a parsed name may contain several foods."""
    tokens = set(re.findall(r"[a-z]+", fold(name)))
    return bool(tokens & _MERGE_WORDS) or any(s in name for s in _MERGE_SYMBOLS)


_WORD = re.compile(r"[a-z0-9]+")


def ungrounded_words(name: str, meal_text: str) -> list[str]:
    """Words of a parsed name that do not appear in the meal text.

    Both are folded (Turkish letters, case), and whole words are compared.
    """
    meal_words = set(_WORD.findall(fold(meal_text)))
    return [word for word in _WORD.findall(fold(name)) if word not in meal_words]


def check_grounding(result: ParseResult, meal_text: str) -> ParseResult:
    """Move items whose name is not in the meal text to ``rejected_items``.

    Strict on purpose: a similarity rule would let an invented "Elma" through
    on "elmas". The cost, measured on the 220 eval items: 5 are rejected, 2 of
    them correct typo fixes by the model ("letuce" -> "lettuce"). They are
    rejected loudly, never counted silently.
    """
    kept: list[ParsedItem] = []
    rejected: list[RejectedItem] = []
    for item in result.items:
        missing = ungrounded_words(item.name, meal_text)
        if missing:
            reason = f"not in the meal text: {', '.join(missing)}"
            rejected.append(RejectedItem(raw=item.model_dump(), reason=reason))
        else:
            kept.append(item)
    return _keep(result, kept, rejected, "no item name appears in the meal text")


def strip_meal_words(name: str) -> str:
    """Drop meal-time words before a name and eating verbs after it.

    "Akşam Biber Dolması" -> "Biber Dolması", "Tabule Yedim" -> "Tabule".
    """
    words = name.split()
    while words and fold(words[0]) in MEAL_TIME_WORDS:
        words.pop(0)
    while words and fold(words[-1]) in EATING_VERBS:
        words.pop()
    return " ".join(words)


def drop_meal_words(result: ParseResult) -> ParseResult:
    """Clean item names; an item that was only meal-time words is rejected."""
    kept: list[ParsedItem] = []
    rejected: list[RejectedItem] = []
    for item in result.items:
        name = strip_meal_words(item.name)
        if name:
            kept.append(item.model_copy(update={"name": name}))
        else:
            reason = "only meal-time words, no food"
            rejected.append(RejectedItem(raw=item.model_dump(), reason=reason))
    return _keep(result, kept, rejected, "no item names a food")


def _keep(
    result: ParseResult,
    kept: list[ParsedItem],
    rejected: list[RejectedItem],
    error_if_none: str,
) -> ParseResult:
    """Keep the valid items; rejected ones are reported, never dropped silently."""
    if rejected:
        result.rejected_items.extend(rejected)
        if kept:
            result.status = ParseStatus.PARTIAL
        else:
            result.status = ParseStatus.INVALID_OUTPUT
            result.error = error_if_none
    result.items = kept
    return result


class ChatModel(Protocol):
    """The subset of ``llama_cpp.Llama`` we use (lets tests inject a fake)."""

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]: ...


def validate_output(raw_output: str) -> ParseResult:
    """Validate raw model text into a ParseResult, item by item."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return ParseResult(
            status=ParseStatus.INVALID_OUTPUT,
            raw_output=raw_output,
            error=f"not valid JSON: {exc}",
        )

    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return ParseResult(
            status=ParseStatus.INVALID_OUTPUT,
            raw_output=raw_output,
            error="expected an object with an 'items' list",
        )

    items: list[ParsedItem] = []
    rejected: list[RejectedItem] = []
    for raw_item in data["items"]:
        try:
            items.append(ParsedItem.model_validate(raw_item))
        except ValidationError as exc:
            reason = "; ".join(err["msg"] for err in exc.errors())
            rejected.append(RejectedItem(raw=raw_item, reason=reason))

    if not items and not rejected:
        status = ParseStatus.EMPTY
    elif not items:
        status = ParseStatus.INVALID_OUTPUT
    elif rejected:
        status = ParseStatus.PARTIAL
    else:
        status = ParseStatus.SUCCESS

    return ParseResult(
        status=status, items=items, rejected_items=rejected, raw_output=raw_output
    )


class MealParser:
    """Thread-safe wrapper around the GGUF meal parser. Load once, reuse."""

    def __init__(
        self,
        llm: ChatModel,
        use_grammar: bool = False,
        max_tokens: int = 512,
        resplit_merged: bool = True,
        check_grounding: bool = True,
    ) -> None:
        self._llm = llm
        self._resplit_merged = resplit_merged
        self._check_grounding = check_grounding
        # llama.cpp keeps one KV cache per model instance: never run two
        # generations on it at the same time.
        self._lock = threading.Lock()
        self._max_tokens = max_tokens
        self._response_format = (
            {"type": "json_object", "schema": MealParseSchema.model_json_schema()}
            if use_grammar
            else None
        )

    @classmethod
    def from_path(
        cls,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        use_grammar: bool = False,
        resplit_merged: bool = True,
        check_grounding: bool = True,
    ) -> "MealParser":
        """Load the GGUF model from disk (slow: call once at startup)."""
        # Imported here so `import astra_nutrition` stays fast and light.
        from llama_cpp import Llama

        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only, same as inside Docker
            chat_format="chatml",
            verbose=False,
        )
        return cls(
            llm,
            use_grammar=use_grammar,
            resplit_merged=resplit_merged,
            check_grounding=check_grounding,
        )

    def parse(self, meal_text: str) -> ParseResult:
        """Parse one meal description. Never raises: failures become a status.

        Items whose name looks merged ("X with Y and Z") are parsed once more
        on their own (Design R). Real dishes ("mac and cheese") come back as a
        single item and are kept. After a split, the first item keeps the
        original amount; the others get an empty amount, which amount
        normalization flags as an assumed portion instead of guessing.

        Then meal-time words are dropped from names ("Akşam Falafel" ->
        "Falafel"), and items whose name is not in ``meal_text`` are rejected
        (see ``check_grounding``), including names from a second parse.
        """
        start = time.perf_counter()
        result = self._parse_once(meal_text)
        if self._resplit_merged and result.items:
            items: list[ParsedItem] = []
            for item in result.items:
                parts = self._split(item)
                if len(parts) > 1:
                    result.resplits.append(Resplit(original=item, names=parts))
                    items.append(ParsedItem(name=parts[0], amount=item.amount))
                    items += [ParsedItem(name=name, amount="") for name in parts[1:]]
                else:
                    items.append(item)
            result.items = items
        if result.items:
            result = drop_meal_words(result)
        if self._check_grounding and result.items:
            result = check_grounding(result, meal_text)
        result.latency_ms = _elapsed_ms(start)
        return result

    def _split(self, item: ParsedItem) -> list[str]:
        """Names from a second parse of a merged-looking item (or just its own)."""
        if not looks_merged(item.name):
            return [item.name]
        second = self._parse_once(item.name)
        if second.status != ParseStatus.SUCCESS or len(second.items) < 2:
            return [item.name]
        return [part.name for part in second.items]

    def _parse_once(self, meal_text: str) -> ParseResult:
        """One model call plus validation."""
        start = time.perf_counter()
        try:
            with self._lock:
                response = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": meal_text},
                    ],
                    temperature=0,
                    stop=["<|im_end|>"],
                    max_tokens=self._max_tokens,
                    response_format=self._response_format,
                )
            choice = response["choices"][0]
            raw_output = choice["message"]["content"] or ""
        except Exception as exc:  # isolate inference failures from the caller
            logger.exception("Parser inference failed")
            return ParseResult(
                status=ParseStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(start),
            )

        if choice.get("finish_reason") == "length":
            result = ParseResult(
                status=ParseStatus.INVALID_OUTPUT,
                raw_output=raw_output,
                error=f"output truncated at max_tokens={self._max_tokens}",
            )
        else:
            result = validate_output(raw_output)
        result.latency_ms = _elapsed_ms(start)
        return result


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
