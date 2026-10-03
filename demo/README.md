---
title: astra-nutrition
emoji: 🥗
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Turkish / English meal text to foods, grams and nutrition
---

# astra-nutrition demo

Try [astra-nutrition](https://github.com/turhanGoksu/astra-nutrition) in the
browser: write a meal in Turkish or English and get foods, grams and
nutrition, with a status for every item.

- Runs the released **v0.2.0** wheel (hash-checked) on this Space's CPU.
- Offline: the optional LLM judge is off, there are no API keys, and the meal
  text is not stored. Install the library to use the judge with your own key.
- One model on a small CPU: requests wait in line. The first request after the
  Space wakes up waits for the model to load.

Nutrition values are estimates from public USDA data (SR Legacy, FNDDS) and
documented Turkish recipes, not medical or dietary advice.
