"""Gradio demo of astra-nutrition (local Docker, or a Hugging Face Space).

The demo installs the released wheel (see requirements.txt), not this
repository, so it shows exactly what `pip install astra-nutrition==0.3.0` gives. It runs
offline on the CPU: no LLM judge, no API keys, and the meal text is not
stored. A public demo with the judge would spend one shared key for every
visitor and give different results at busy times.

Run it with Docker (see Dockerfile), or from this folder with
requirements.txt installed:
    python app.py        # http://127.0.0.1:7860
"""

from astra_nutrition import AnalysisResult, Analyzer, ItemStatus

MAX_CHARS = 1000  # same limit as the web API
COLUMNS = ["Item", "Amount", "Status", "Food", "Grams", "kcal"]
STATUS_LABEL = {
    ItemStatus.OK: "ok",
    ItemStatus.ESTIMATED: "estimated*",
    ItemStatus.AMOUNT_UNKNOWN: "amount unknown",
    ItemStatus.UNMATCHED: "unmatched",
}
# Two meals the table covers fully, two that show estimates, an unreadable
# amount and a gap honestly.
EXAMPLES = [
    "1 kase yoğurt, bir avuç ceviz ve 2 yemek kaşığı bal",
    "2 boiled eggs, 100 g rice and an apple",
    "öğlen 1 kase mercimek çorbası, biraz pilav ve 1 dilim baklava",
    "akşam 2 lahmacun, 1 bardak ayran ve biraz cacık",
]
INTRO = """\
# astra-nutrition

Turkish / English meal text → foods, grams and nutrition. A fine-tuned 1.5B
parser runs **on this server's CPU**: no API keys, and your text is not stored.
Every item gets a status, and the total says what it leaves out.

[GitHub](https://github.com/turhanGoksu/astra-nutrition) ·
[Model](https://huggingface.co/Turhan123/astra-meal-parser-gguf) ·
Not medical or dietary advice.
"""


def result_rows(result: AnalysisResult) -> list[list[object]]:
    """One table row per item; "-" where nothing was counted."""
    return [
        [
            item.name,
            item.amount,
            STATUS_LABEL[item.status],
            item.food_name_en or "-",
            item.grams if item.grams is not None else "-",
            item.nutrition.kcal if item.nutrition else "-",
        ]
        for item in result.items
    ]


def summary(result: AnalysisResult) -> str:
    """Totals in Markdown, saying what is estimated and what is not counted."""
    if not result.items:
        return f"No food items found (parser status: `{result.parse_status}`)."
    t = result.totals
    lines = [
        f"**Total: {t.kcal:g} kcal** · protein {t.protein_g:g} g · "
        f"carbs {t.carbs_g:g} g · fat {t.fat_g:g} g"
    ]
    if t.includes_estimates:
        lines.append(f"\\* {t.estimated} item(s) use a default portion (estimate).")
    if not t.complete:
        rejected = f", {t.rejected} rejected by the parser checks" if t.rejected else ""
        lines.append(
            f"Not counted: {t.unmatched} unmatched (not in the food table yet), "
            f"{t.amount_unknown} with an unreadable amount{rejected}."
        )
    for item in result.rejected_items:
        name = item.raw.get("name") if isinstance(item.raw, dict) else item.raw
        lines.append(f"Rejected: `{name}` ({item.reason})")
    return "\n\n".join(lines)


def build_demo(analyzer: Analyzer):
    """The Gradio UI. Gradio is imported here so tests need only the helpers."""
    import gradio as gr

    def analyze(meal_text: str) -> tuple[list[list[object]], str]:
        text = meal_text.strip()
        if not text:
            raise gr.Error("Write a meal first.")
        if len(text) > MAX_CHARS:
            raise gr.Error(f"Please keep it under {MAX_CHARS} characters.")
        result = analyzer.analyze(text)
        return result_rows(result), summary(result)

    with gr.Blocks(title="astra-nutrition") as demo:
        gr.Markdown(INTRO)
        meal = gr.Textbox(
            label="Meal",
            placeholder="2 yumurta, 1 dilim ekmek ve bir muz",
            max_length=MAX_CHARS,
            lines=2,
        )
        button = gr.Button("Analyze", variant="primary")
        table = gr.Dataframe(headers=COLUMNS, interactive=False)
        totals = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=meal)
        button.click(analyze, meal, [table, totals], api_name="analyze")
        meal.submit(analyze, meal, [table, totals], api_name=False)
    return demo


if __name__ == "__main__":
    analyzer = Analyzer()
    analyzer.analyze("1 muz")  # download and load the model before the first visitor
    # One model on a small CPU: requests wait in line, and a full line is
    # refused with a message instead of growing without limit.
    demo = build_demo(analyzer).queue(default_concurrency_limit=1, max_size=20)
    demo.launch(server_name="0.0.0.0", server_port=7860)
