"""Pre-compute and freeze chat answers for the demo.

Runs the coordinator live (real Gemini) over a set of demo properties × the
questions a presenter is likely to ask, writing each answer through to
``Data/processed/chat_cache.json`` via the coordinator's own cache.

On demo day, run the app with ``DEMO_MODE=1`` so the chat is served entirely
from this file — no live LLM call, no network risk.

Usage (from the repo root, with the project env):
    python scripts/build_demo_cache.py
"""

from __future__ import annotations

import os
from dataclasses import asdict

# Build in live mode so run_chat writes through to the cache.
os.environ.pop("DEMO_MODE", None)

from airbnb_iip.agents import chat_cache  # noqa: E402
from airbnb_iip.agents.coordinator import run_chat  # noqa: E402
from airbnb_iip.decision.engine import Property, compute_scenario  # noqa: E402

# Representative demo properties — one per city, cheap + premium districts.
DEMO_PROPERTIES = [
    Property(city="madrid", district="Salamanca", size_m2=85, bedrooms=2,
             bathrooms=1.0, accommodates=4, room_type="Entire home/apt",
             has_ac=True, has_elevator=True),
    Property(city="barcelona", district="Eixample", size_m2=95, bedrooms=3,
             bathrooms=2.0, accommodates=5, room_type="Entire home/apt",
             has_ac=True, has_elevator=True, has_balcony=True),
    Property(city="malaga", district="Centro", size_m2=70, bedrooms=2,
             bathrooms=1.0, accommodates=3, room_type="Entire home/apt",
             has_ac=True),
]

# Questions a presenter is likely to ask (covers every coordinator intent).
QUESTIONS = [
    "Why this recommendation?",
    "Should I sell or keep it on Airbnb?",
    "What net revenue will I earn?",
    "What's the projected occupancy?",
    "What nightly price does the model predict?",
    "What about regulations here?",
    "Show me comparable listings.",
    "How can I improve the property?",
    "What if I added air conditioning?",
]

GENERAL_QUESTIONS = ["hello", "what can you do?"]


def main() -> None:
    chat_cache.reload()
    count = 0
    for prop in DEMO_PROPERTIES:
        pdict = asdict(prop)
        scen = asdict(compute_scenario(prop))
        label = f"{prop.district}, {prop.city}"
        print(f"\n[{label}] computing {len(QUESTIONS)} answers…")
        for q in QUESTIONS:
            res = run_chat(q, property=pdict, scenario=scen, use_llm=True)
            count += 1
            print(f"  · [{res['intent']:11}] {q}")

    for q in GENERAL_QUESTIONS:
        run_chat(q, use_llm=True)
        count += 1

    print(f"\nCached {count} answers → {chat_cache.CACHE_PATH}")
    print("Run the demo with DEMO_MODE=1 to serve these without any live LLM call.")


if __name__ == "__main__":
    main()
