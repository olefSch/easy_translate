from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List, Optional

from ollama import Client

from .base_translator import BaseTranslator, TranslationError

logger = logging.getLogger(__name__)


class LLMTranslator(BaseTranslator):
    """
    Translator that drives any Ollama‑hosted model via the Ollama Python client.
    """

    DEFAULT_TEMPLATE = textwrap.dedent(
        """\
        Translate the following sentence from {source_lang} to {target_lang}:

        {text}

        Return ONLY the translated sentence—no quotes, labels, explanations or extra whitespace.
    """
    )

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        num_predict: int = 512,
        source_lang: str = "English",
        target_lang: str = "German",
        stop: Optional[List[str]] = None,
        client: Optional[Client] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initialize an Ollama-based translator.

        Args:
            model_name (str): Ollama model ID (e.g. "llama3.1:8b").
            num_predict (int): Maximum number of tokens to predict per call.
            source_lang (str): Name of the source language.
            target_lang (str): Name of the target language.
            stop (Optional[List[str]]): Optional list of stop sequences; defaults to ["—"].
            client (Optional[Client]): Optional pre-configured Ollama Client; if None, constructs a new one.
            prompt_template (Optional[str]): Optional prompt template with placeholders
                {source_lang}, {target_lang}, {text}. If None uses DEFAULT_TEMPLATE.

        Raises:
            ValueError: If model_name, source_lang, or target_lang are empty,
                or if num_predict is not positive.
        """
        # Basic validation
        self._validate_language_pair(source_lang, target_lang)
        self._validate_positive("num_predict", num_predict)

        # Use provided client or create a new one
        self.client: Client = client or Client()
        self.model_name: str = model_name
        self.num_predict: int = num_predict

        # Stop sequences to tell the model when to stop generating
        self.stop: List[str] = stop if stop is not None else ["—"]
        self.source_lang: str = source_lang
        self.target_lang: str = target_lang

        # Prompt template for constructing the translation prompt
        self.prompt_template: str = prompt_template or self.DEFAULT_TEMPLATE

    def translate(
        self,
        text: str,
    ) -> str:
        """Translate the given text via the Ollama LLM.

        Args:
            text (str): A single sentence to translate.

        Returns:
            str: The translated sentence (stripped of surrounding whitespace).

        Raises:
            TranslationError: On any failure from the Ollama client.
        """
        # Short-circuit on empty input
        if not text:
            logger.debug("Empty input text; returning empty string.")
            return ""

        # Build the prompt by filling in the template with the language and text
        prompt = self.prompt_template.format(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            text=text,
        )
        logger.debug("LLMTranslator prompt:\n%s", prompt)

        try:
            # Call the Ollama model with the prompt and generation settings
            response: Dict[str, Any] = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": self.num_predict,
                    "stop": self.stop,
                },
            )
        except Exception as e:
            # Wrap any errors from Ollama client in a custom TranslationError
            logger.error(
                "Ollama client error for model %s: %s", self.model_name, e
            )
            raise TranslationError(f"Ollama error: {e}") from e

        # Extract the raw response string from the model output
        translated = response.get("response", "")
        if not isinstance(translated, str):
            logger.error("Invalid response format: %r", response)
            raise TranslationError(f"Invalid response format: {response!r}")

        # Strip extra whitespace and return
        result = translated.strip()
        logger.debug("LLMTranslator result: %s", result)
        return result
