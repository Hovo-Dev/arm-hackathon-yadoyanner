#!/usr/bin/env python3
"""Minimal DeepSeek call through the OpenRouter gateway (OpenAI-compatible).

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python example.py
    python example.py "Summarise this idea in two sentences: ..."
"""

import os
import sys

from openai import OpenAI

MODEL = "deepseek/deepseek-v4-pro"
BASE_URL = "https://openrouter.ai/api/v1"

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("OPENROUTER_API_KEY is not set. See README.md.")

client = OpenAI(api_key=api_key, base_url=BASE_URL)

prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
)

print(response.choices[0].message.content)

usage = response.usage
if usage:
    print(
        f"\n[tokens] prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} total={usage.total_tokens}",
        file=sys.stderr,
    )
