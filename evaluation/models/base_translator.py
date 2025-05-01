from abc import ABC, abstractmethod

class TranslationError(Exception):
    """Base exception for all translation failures."""
    pass


class BaseTranslator(ABC):
    """
    Abstract base class for translator models. Ensures a consistent
    interface for all translators, whether local LLM-based or via an API.
    """

    @abstractmethod
    def translate(self, text: str) -> str:
        """
        Translate a single piece of text from the source language
        to the target language.
        """
        pass
    
    @staticmethod
    def _validate_non_empty(name: str, value: str) -> None:
        """Raise ValueError if `value` is empty or only whitespace."""
        if not value or not value.strip():
            raise ValueError(f"`{name}` must be a non-empty string")

    @staticmethod
    def _validate_positive(name: str, value: int) -> None:
        """Raise ValueError if `value` is not a positive integer."""
        if value <= 0:
            raise ValueError(f"`{name}` must be > 0 (got {value})")

    def _validate_language_pair(
        self, source_lang: str, target_lang: str
    ) -> None:
        """Raise ValueError if either language code is empty."""
        self._validate_non_empty("source_lang", source_lang)
        self._validate_non_empty("target_lang", target_lang)

    def _validate_generation_params(
        self, max_length: int, num_beams: int
    ) -> None:
        """Raise ValueError if generation params are invalid."""
        self._validate_positive("max_length", max_length)
        self._validate_positive("num_beams", num_beams)