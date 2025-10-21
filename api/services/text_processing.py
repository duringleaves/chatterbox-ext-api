"""ChatGPT-assisted text preprocessing for VO scripts."""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from ..config import settings

logger = logging.getLogger(__name__)


class TextProcessingError(Exception):
    """Raised when the OpenAI service returns an unexpected response."""


class TextPreprocessor:
    """Service that normalizes voice-over scripts via OpenAI."""

    def __init__(self, api_key: Optional[str]) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is not configured.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.logger = logger
        self.system_prompt = f"""
            You are an expert radio producer and copywriter. You will be given a JSON array of strings.
            Process each string in the array according to the following rules and return a JSON array of the processed strings in the same order.

            Rules (apply in this exact order):
            1. DO NOT MODIFY WORDS OR PHRASES IN QUOTATION MARKS! They should be left exactly as-is, and this rule overrides all others.

            2. For radio station call signs starting with "W" or "K":
            - If the call sign is a REAL ENGLISH WORD, keep it as a word: KISS stays "kiss", COOL stays "cool", ROCK stays "rock"
            - If the call sign is NOT a real English word, separate the letters with spaces: KIIS becomes "k i i s", WKRP becomes "w k r p", KLOS becomes "k l o s"
            - Common real words to keep as words: KISS, COOL, ROCK, STAR, LOVE, HITS, GOLD, JACK, MIKE, DAVE
            - Common non-words to spell out: KIIS, KLOS, KROQ, WKRP, WXYZ, KPWR, WEBN

            3. Spell out numbers in natural language:
            - 100 becomes 'one hundred'
            - 0 becomes 'oh'
            - 95.5 becomes "ninety-five five" (omit 'point' unless in quotes)
            - 100.3 becomes "one hundred point three" (add 'point' unless in quotes)
            - 103.9 becomes "one oh three nine" (omit 'point' unless in quotes)
            - 101.1 becomes "one oh one point one" (include 'point' unless in quotes)
            - Phone numbers should be read like a human reads them (eg. '800 520 1027' becomes 'eight hundred, five two oh, one oh two seven')

            4. Separate letters for abbreviations that should be spelled out:
            - "LA's number one" becomes "l a's number one"
            - "DJ" becomes "d j"

            5. Replace ellipses with three periods (...) and add a period to the end of any line that does not end with punctuation.

            6. If the context for "live" implies "lyve" (e.g., 'see them live on stage'), spell it as "lyve".

            7. Make all letters lowercase

            8. Do not add any other words or text, UNLESS a given phrase is SHORTER THAN 2 WORDS. In that case, add the phrase "{settings.prepend_phrase}." before the processed phrase.

            IMPORTANT: For call signs, distinguish between real words and letter combinations:
            - Real words like KISS, COOL, ROCK should stay as words: "kiss fm", "cool fm", "rock fm"
            - Letter combinations like KIIS, WKRP, KLOS should be spelled out: "k i i s fm", "w k r p", "k l o s"

            Respond ONLY with a valid JSON object in the format: {{"processed_lines": ["line 1 result", "line 2 result", ...]}}
        """

    async def process_batch(self, texts: List[str], log: Optional[logging.Logger] = None) -> List[str]:
        log = log or self.logger
        if not texts:
            return []

        log.info("Pre-processing batch of %s lines with OpenAI.", len(texts))
        user_content = json.dumps(texts)
        log.debug("OpenAI request payload: %s", user_content)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover - network failure
            log.error("Error during OpenAI batch processing call: %s", exc)
            raise TextProcessingError(f"OpenAI API batch call failed: {exc}") from exc

        response_content = response.choices[0].message.content if response.choices else None
        if not response_content:
            log.error("OpenAI returned an empty response")
            raise TextProcessingError("OpenAI returned an empty response")

        log.debug("OpenAI response payload: %s", response_content)

        try:
            processed_data = json.loads(response_content)
        except json.JSONDecodeError as exc:
            log.error("Failed to decode JSON response from OpenAI: %s\nResponse: %s", exc, response_content)
            raise TextProcessingError(f"OpenAI returned invalid JSON: {exc}") from exc

        processed_lines = processed_data.get("processed_lines", [])
        if len(processed_lines) != len(texts):
            raise TextProcessingError(
                f"OpenAI returned a different number of lines ({len(processed_lines)}) than expected ({len(texts)})."
            )

        log.info("Successfully processed batch of %s lines.", len(texts))
        return processed_lines

    async def process_text(self, text: str, log: Optional[logging.Logger] = None) -> str:
        processed = await self.process_batch([text], log=log)
        return processed[0] if processed else text


try:
    text_preprocessor = TextPreprocessor(settings.openai_api_key)
except ValueError as exc:
    logger.warning("TextPreprocessor service not initialized: %s", exc)
    text_preprocessor = None
