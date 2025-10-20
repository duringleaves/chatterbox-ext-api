"""ChatGPT-assisted text preprocessing for VO scripts."""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import HTTPException
from openai import AsyncOpenAI

from ..config import settings

logger = logging.getLogger(__name__)


class TextProcessingError(Exception):
    """Raised when the OpenAI service returns an unexpected response."""


class TextPreprocessor:
    def __init__(self, api_key: Optional[str]) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is not configured.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = f"""
            You are an expert radio producer and copywriter. You will be given a JSON array
            of strings and must return a JSON object {{"processed_lines": [...]}} with the
            processed strings in the same order.

            Rules (apply in this exact order):
            1. DO NOT CHANGE WORDS OR PHRASES INSIDE QUOTATION MARKS.
            2. For station call signs starting with 'W' or 'K':
               - If the call sign is a real English word (e.g., KISS, COOL, ROCK) keep it whole.
               - Otherwise spell out the individual letters separated by spaces (e.g., KIIS -> "k i i s").
            3. Spell out numbers naturally. Examples:
               - 100 -> "one hundred"
               - 0 -> "oh"
               - 95.5 -> "ninety-five five" (omit 'point' unless in quotes)
               - 100.3 -> "one hundred point three"
               - 103.9 -> "one oh three nine"
               - 101.1 -> "one oh one point one"
               - Phone numbers should be human-friendly ("800 520 1027" -> "eight hundred, five two oh, one oh two seven").
            4. Spell out abbreviations that should be read letter-by-letter (e.g., "L.A." -> "l a", "DJ" -> "d j").
            5. Replace ellipses with "..." and ensure each line ends with punctuation.
            6. If context implies the festival brand "Lyve", spell "live" as "lyve".
            7. Convert text to lowercase.
            8. Do not add new words unless the phrase is shorter than two words; in that case,
               prepend "{settings.prepend_phrase}." before the processed phrase.

            Respond ONLY with valid JSON in the shape {{"processed_lines": [...]}}.
        """

    async def process_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(texts)},
                ],
            )
        except Exception as exc:  # pragma: no cover - network error path
            logger.error("OpenAI request failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise TextProcessingError("OpenAI returned an empty response")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - response format issue
            logger.error("Failed to parse OpenAI response: %s\nContent: %s", exc, content)
            raise TextProcessingError("OpenAI returned invalid JSON")

        processed = payload.get("processed_lines")
        if not isinstance(processed, list) or len(processed) != len(texts):
            raise TextProcessingError(
                "OpenAI returned an unexpected number of lines"
            )
        return processed


try:
    text_preprocessor = TextPreprocessor(settings.openai_api_key)
except ValueError:
    logger.warning("OPENAI_API_KEY not configured; /scripts/analyze will be unavailable.")
    text_preprocessor = None
