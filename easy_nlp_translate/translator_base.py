import logging
from abc import ABC, abstractmethod
from typing import Optional

from langdetect import detect

from .config import available_language_codes
from .exceptions import DetectionError

logger = logging.getLogger(__name__)


class TranslatorBase(ABC):
    """
    Abstract base class for all translator implementations.

    This class defines the core interface for translation services,
    including language validation, language detection, and the abstract
    `translate` method that concrete subclasses must implement.

    Attributes:
        LANGUAGE_CODES (List[str]): A list of supported language codes.
            Concrete subclasses might override this.
        source_lang (Optional[str]): The source language code for translation.
            If None, auto-detection is typically attempted.
        target_lang (str): The target language code for translation.
    """

    LANGUAGE_CODES: list[str] = (
        available_language_codes  # needs to be overwritten in smaller huggingface model classes
    )

    def __init__(self, target_lang: str, source_lang: Optional[str] = None):
        """
        Initializes the translator with target and optional source languages.

        Args:
            target_lang: The language code for the target translation
                language (e.g., 'fr' for French).
            source_lang: The language code of the source text
                (e.g., 'en' for English). Defaults to None, implying
                auto-detection will be attempted by subclasses or specific methods.
        """
        self._validate_language_pair(source_lang, target_lang)

        self.source_lang = source_lang
        self.target_lang = target_lang

        logger.info(
            f"{self.__class__.__name__} initialized with source language: "
            f"{self.source_lang if self.source_lang is not None else 'auto-detect'} "
            f"and target language: {self.target_lang}"
        )

    def _validate_langauge(self, lang: str):
        """
        Validates if the given language code is supported.

        Args:
            lang: The language code string to validate (e.g., 'en', 'fr').

        Raises:
            ValueError: If the language code `lang` is not found in
                `self.LANGUAGE_CODES`.
        """
        if lang not in self.LANGUAGE_CODES:
            raise ValueError(
                f"Language '{lang}' is not supported. Supported languages are: {self.LANGUAGE_CODES}"
            )

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text using langdetect.

        Args:
            text: The text whose language is to be detected.

        Returns:
            The detected language code (e.g., 'en', 'fr').

        Raises:
            ValueError: If the text is empty or invalid for detection.
            LangDetectException: If language detection by the `langdetect`
                library fails for other reasons (e.g., text too short, no features).
            ValueError: If the detected language is not in `self.LANGUAGE_CODES`.
        """
        TranslatorBase._validate_basic_text_to_translate(text)

        try:
            lang = detect(text)
        except Exception as e:
            raise DetectionError(
                f"Language detection failed for text snippet '{text[:50]}...': Original error: {str(e)}"
            ) from e

        self._validate_langauge(lang)

        return lang

    def _validate_language_pair(
        self, source_lang: Optional[str], target_lang: str
    ):
        """
        Validates the source and target language pair.

        Ensures that the target language is supported. If a source language
        is provided, it is also validated. It also checks if auto-detection
        seems feasible if no source language is given, and ensures source and
        target languages are not identical.

        Args:
            source_lang: The source language code (e.g., 'en').
                Can be None if auto-detection is intended.
            target_lang: The target language code (e.g., 'fr').

        Raises:
            ValueError: If `target_lang` or `source_lang` (if provided)
                are not in `self.LANGUAGE_CODES`.
            ValueError: If `source_lang` and `target_lang` are the same.
            NotImplementedError: If `source_lang` is None and language
                auto-detection capability (tested by `self.detect_language("test")`)
                is not properly implemented or fails its basic test.
        """
        self._validate_langauge(target_lang)
        if source_lang is not None:
            self._validate_langauge(source_lang)

        if source_lang == target_lang:
            raise ValueError("Source and target languages cannot be the same.")

    @staticmethod
    def _validate_basic_text_to_translate(text: str):
        """
        Validate the text to be translated.

        Args:
            text (str): The text to be validated.

        Raises:
            ValueError: If the text is empty or not a string.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to translate must be a non-empty string.")

    @abstractmethod
    def translate(self, text: str) -> str:
        """
        Translate the given text from source language to target language.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )

    def translate_batch(self, texts: list) -> list:
        """
        Translate a batch of texts from source language to target language.

        Args:
            texts (list): A list of texts to be translated.

        Returns:
            list: A list of translated texts.
        """
        for text in texts:
            self._validate_basic_text_to_translate(text)

        return [self.translate(text) for text in texts]
