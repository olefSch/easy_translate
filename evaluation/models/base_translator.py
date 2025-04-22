from abc import ABC, abstractmethod

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
