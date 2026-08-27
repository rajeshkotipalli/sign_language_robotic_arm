"""
sentence_builder.py
Uses the Claude API to convert a list of imperfectly-spelled words
from ASL recognition into a clean, natural English sentence.
"""

import requests
import json


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def build_sentence(words: list[str]) -> str:
    """
    Takes a list of words spelled out via ASL (may have errors/missing letters)
    and returns a corrected natural sentence.

    Args:
        words: e.g. ["HELO", "MY", "NAM", "IS", "RAJESH"]

    Returns:
        e.g. "Hello, my name is Rajesh."
    """
    if not words:
        return ""

    words_str = " ".join(w for w in words if w.strip())

    prompt = f"""You are an ASL (American Sign Language) communication assistant.
The following text was spelled out letter by letter using ASL hand signs. 
Due to recognition errors, some letters may be missing or wrong.

Spelled text: {words_str}

Your job:
1. Correct spelling errors and reconstruct the most likely intended sentence.
2. Return ONLY the corrected natural English sentence.
3. Do NOT add any explanation, quotes, or extra text — just the sentence.
4. Keep it concise and natural.
5. If you cannot determine the meaning, return the input text cleaned up."""

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=8
        )
        data = resp.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", words_str).strip()
        return words_str
    except Exception:
        # Fallback: just join the words cleanly
        return words_str


def build_sentence_quick(words: list[str]) -> str:
    """
    Lightweight offline fallback — joins words and applies basic corrections.
    Used when the Claude API is not available.
    """
    if not words:
        return ""
    # Basic cleanup: lowercase, strip empties, rejoin
    cleaned = [w.capitalize() for w in words if w.strip()]
    return " ".join(cleaned) + ("." if cleaned else "")
