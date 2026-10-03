"""End-to-end evaluation on the v0.2 meals: meal text -> foods.

Someone other than the alias author wrote these meals without looking at the
table. Per meal, the labels list the foods a correct system finds ("a|b":
either is right; "?a": allowed but not required, e.g. the yogurt on mantı).
- dev  (data/eval/meals_v02.txt): used to find and fix problems, so its
  numbers are not a held-out result;
- test (data/eval/meals_v02_test.txt): labeled and committed before the
  system ever ran on it, evaluated once, nothing fixed afterwards.

Scoring is per food, through the whole pipeline (parser, name and amount
checks, matching), so fixes that change parsed names are measured too:
- found: a gold food that some item matched (any amount status);
- missed: a gold food no item matched;
- false match: a matched food that no gold food allows (silent wrong data).

Model completions are cached (eval/results/v02_llm_cache.json), so reruns are
fast and do not depend on the machine's llama.cpp build.

Usage (from the project root):
    python -m eval.v02 --label baseline            # dev set
    python -m eval.v02 --set test --label final     # test set, once
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from astra_nutrition import Analyzer
from astra_nutrition.parser import MealParser

SETS = {
    "dev": (Path("data/eval/meals_v02.txt"), Path("data/eval/labels_v02.csv")),
    "test": (
        Path("data/eval/meals_v02_test.txt"),
        Path("data/eval/labels_v02_test.csv"),
    ),
}
RESULTS_DIR = Path("eval/results")
CACHE_PATH = RESULTS_DIR / "v02_llm_cache.json"


class CachedChatModel:
    """Replays chat completions from a JSON file; runs the model on a miss."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        self._llm: Any = None

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        key = json.dumps(
            [kwargs["messages"], kwargs.get("response_format")], ensure_ascii=False
        )
        if key not in self._cache:
            if self._llm is None:
                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=str(get_settings().model_path),
                    n_ctx=2048,
                    n_gpu_layers=0,
                    chat_format="chatml",
                    verbose=False,
                )
            self._cache[key] = self._llm.create_chat_completion(**kwargs)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        return self._cache[key]


def score(
    gold: list[set[str]], matched: list[str], optional: list[set[str]] = ()
) -> dict[str, list[str]]:
    """Found, missed and falsely matched foods for one meal.

    Optional foods are never missed, and matching them is no false match.
    """
    found = [alts for alts in gold if alts & set(matched)]
    allowed = set().union(*gold, *optional)
    return {
        "found": ["|".join(sorted(a)) for a in found],
        "missed": ["|".join(sorted(a)) for a in gold if a not in found],
        "false_matches": [food for food in matched if food not in allowed],
    }


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--set", choices=list(SETS), default="dev")
    args.add_argument("--label", required=True, help="name for the results file")
    parsed = args.parse_args()
    label, meals_path, labels_path = parsed.label, *SETS[parsed.set]

    meals = meals_path.read_text(encoding="utf-8").splitlines()
    with open(labels_path, encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    analyzer = Analyzer(parser=MealParser(CachedChatModel(CACHE_PATH)))

    rows, totals = [], {"found": 0, "missed": 0, "false_matches": 0}
    for meal, label_row in zip(meals, labels, strict=True):
        items = [g for g in label_row["gold"].split(";") if g]
        gold = [set(g.split("|")) for g in items if not g.startswith("?")]
        optional = [set(g[1:].split("|")) for g in items if g.startswith("?")]
        result = analyzer.analyze(meal)
        matched = [i.food_id for i in result.items if i.food_id is not None]
        scored = score(gold, matched, optional)
        for key in totals:
            totals[key] += len(scored[key])
        rows.append(
            {
                "line": int(label_row["line"]),
                "meal": meal,
                **scored,
                "items": [
                    [i.name, i.amount, i.status.value, i.food_id] for i in result.items
                ],
                "rejected": [r.reason for r in result.rejected_items],
            }
        )

    gold_total = totals["found"] + totals["missed"]
    summary = {
        "set": parsed.set,
        "label": label,
        "meals": len(meals),
        "gold_foods": gold_total,
        "found": totals["found"],
        "recall": round(totals["found"] / gold_total, 3),
        "false_matches": totals["false_matches"],
    }
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row["missed"] or row["false_matches"]:
            print(
                f"  {row['line']:3} {row['meal'][:40]:40} missed={row['missed']} "
                f"false={row['false_matches']}"
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"v02_{parsed.set}_{label}.json"
    out.write_text(
        json.dumps({"summary": summary, "meals": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
