from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional 

class TranslatorBase(ABC):

    def __init__(self, target_lang: str, source_lang: Optional[str] = None): 
        """
        Initialize the translator with optional source and required target languages.

        Args:
            target_lang (str): The target language code (e.g., 'fr' for French).
            source_lang (Optional[str], optional): The source language code (e.g., 'en' for English). 
                                                   Defaults to None (implying auto-detection).
        """
        self.source_lang = source_lang
        self.target_lang = target_lang

        print(f"Translator initialized with source language: {self.source_lang if self.source_lang is not None else 'auto'} and target language: {self.target_lang}")

    @abstractmethod
    def translate(self, text: str) -> str:
        """
        Translate the given text from source language to target language.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text.
        """
        pass

    @abstractmethod
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        Args:
            text (str): The text whose language is to be detected.

        Returns:
            str: The detected language code.
        """
        pass

    @abstractmethod
    def translate_batch(self, texts: list) -> list:
        """
        Translate a batch of texts from source language to target language.

        Args:
            texts (list): A list of texts to be translated.

        Returns:
            list: A list of translated texts.
        """
        pass

    @abstractmethod
    def _generate(self, input: str) -> Iterable:
        """
        Generate a translation for the given input.

        Args:
            input (str): The input text to be translated.

        Returns:
            str: The generated model output
        """
        pass

    @abstractmethod
    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-process the generated output.

        Args:
            output (str): The generated output to be post-processed.

        Returns:
            str: The post-processed output.
        """
        pass
