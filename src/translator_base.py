from abc import ABC, abstractmethod
from typing import Optional 

class TranslatorBase(ABC):

    LANGUAGE_CODE_PAIRS: dict[str, str] | list = NotImplemented

    def __init__(self, target_lang: str, source_lang: Optional[str] = None): 
        """
        Initialize the translator with optional source and required target languages.

        Args:
            target_lang (str): The target language code (e.g., 'fr' for French).
            source_lang (Optional[str], optional): The source language code (e.g., 'en' for English). 
                                                   Defaults to None (implying auto-detection).
        """
        self._validate_language_pair(source_lang, target_lang)

        self.source_lang = source_lang
        self.target_lang = target_lang

        print(f"Translator initialized with source language: {self.source_lang if self.source_lang is not None else 'auto'} and target language: {self.target_lang}")



    @staticmethod
    def _validate_language_pair(source_lang: Optional[str], target_lang: str):
        """
        Validate the source and target language pair.

        Args:
            source_lang (Optional[str]): The source language code.
            target_lang (str): The target language code.

        Raises:
            NotImplementedError: If the source language is not provided and auto-detection is not implemented.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

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
        raise NotImplementedError("This method should be implemented in subclasses.")

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
