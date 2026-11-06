"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.8-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
 
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage

# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Supports new SDK (google-genai) with fallback to legacy google-generativeai."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    start_time = time.time()
 
    try:
        # Option A: New Google GenAI SDK (preferred)
        from google import genai
        from google.genai import types
 
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency = time.time() - start_time
 
        text = response.text or ""
        meta = response.usage_metadata
        usage = {
            "input_tokens": (meta.prompt_token_count or 0) if meta else 0,
            "output_tokens": (meta.candidates_token_count or 0) if meta else 0,
        }
        return text, latency, usage
 
    except (ImportError, Exception):
        # Option B: Fallback to legacy google-generativeai SDK
        import google.generativeai as genai
 
        genai.configure(api_key=api_key)
        model_inst = genai.GenerativeModel(model)
        config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        start_time = time.time()
        response = model_inst.generate_content(prompt, generation_config=config)
        latency = time.time() - start_time
 
        text = response.text or ""
        try:
            input_tokens = model_inst.count_tokens(prompt).total_tokens
            output_tokens = model_inst.count_tokens(text).total_tokens
        except Exception:
            # Heuristic fallback if token service fails
            input_tokens = int(len(prompt.split()) * 1.5)
            output_tokens = int(len(text.split()) * 1.5)
 
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    text = "".join(getattr(block, "text", "") or "" for block in response.content)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    calls = {
        "gpt4o": (OPENAI_MODEL, call_openai(prompt, model=OPENAI_MODEL)),
        "gpt4o_mini": (OPENAI_MINI_MODEL, call_openai(prompt, model=OPENAI_MINI_MODEL)),
        "gemini_flash": (GEMINI_MODEL, call_gemini(prompt, model=GEMINI_MODEL)),
    }

    result = {}
    for key, (model_name, (text, latency, usage)) in calls.items():
        rates = PRICING_1M_TOKENS[model_name]
        cost = (
            usage["input_tokens"] * rates["input"]
            + usage["output_tokens"] * rates["output"]
        ) / 1_000_000
        result[key] = {
            "response": text,
            "latency": latency,
            "cost": cost,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }
    return result
 
# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
    from google.genai import types
 
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
 
    # history: list of {"role": "user"|"model", "text": str}
    history: list[dict] = []
 
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
 
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
 
        history.append({"role": "user", "text": user_input})
        # Keep last 3 turns = 6 messages
        history = history[-6:]
 
        formatted_history = [
            types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
            for m in history
        ]
 
        print("Gemini: ", end="", flush=True)
        reply = ""
        try:
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=formatted_history,
            )
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    reply += chunk.text
            print()
        except Exception as e:
            print(f"\n[Error] {e}")
            history.pop()  # drop the user message that failed
            continue
 
        history.append({"role": "model", "text": reply})
        history = history[-6:]
 


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # type: ignore[misc]
 


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for p in prompts:
        r = compare_models(p)
        r["prompt"] = p
        results.append(r)
    return results
 
# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    def _clean(s: str, limit: int | None = None) -> str:
        s = str(s).replace("\n", " ").replace("|", "\\|").strip()
        if limit is not None and len(s) > limit:
            s = s[:limit] + "..."
        return s
    
    display = {"gpt4o": "GPT-4o", "gpt4o_mini": "GPT-4o-Mini", "gemini_flash": "Gemini-Flash"}
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        prompt = _clean(item.get("prompt", ""), 40)
        for key in ("gpt4o", "gpt4o_mini", "gemini_flash"):
            if key not in item:
                continue
            s = item[key]
            lines.append(
                f"| {prompt} | {display.get(key, key)} | {_clean(s['response'], 50)} "
                f"| {s['latency']:.2f}s "
                f"| {s['input_tokens']}/{s['output_tokens']} "
                f"| ${s['cost']:.6f} |"
            )
    return "\n".join(lines)
 

# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 3.8 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
