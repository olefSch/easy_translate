from .translator_base import TranslatorBase
from typing import Optional, Union

import torch


class HuggingFaceTranslator(TranslatorBase):
    """
    A base class for Hugging Face-based translators, inheriting from TranslatorBase.
    """

    def __init__(self, target_lang: str, source_lang: Optional[str] = None):
        """
        Initialize the Hugging Face translator with optional source and required target languages.

        Args:
            target_lang (str): The target language code (e.g., 'fr' for French).
            source_lang (Optional[str], optional): The source language code (e.g., 'en' for English). 
                                                   Defaults to None (implying auto-detection).
        """
        super().__init__(target_lang, source_lang)
        self.device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
