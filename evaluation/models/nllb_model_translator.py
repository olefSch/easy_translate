import logging
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .base_translator import BaseTranslator, TranslationError

logger = logging.getLogger(__name__)


class NllbTranslator(BaseTranslator):
    """
    Translator using Hugging Face’s NLLB-200 for multilingual translation.
    """

    MODEL_NAME: str = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        device: Union[str, torch.device] = "cpu",
        max_length: int = 512,
        num_beams: int = 4,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the NLLB translator.

        Args:
            source_lang (str): Source language code (e.g. "eng_Latn").
            target_lang (str): Target language code (e.g. "deu_Latn").
            device (Union[str, torch.device]): "cpu", "cuda", or torch.device.
                Defaults to "cpu".
            max_length (int): Maximum length of generated sequences.
                Defaults to 512.
            num_beams (int): Number of beams for beam search.
                Defaults to 4.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Extra kwargs for
                `AutoTokenizer.from_pretrained`. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]]): Extra kwargs for
                `AutoModelForSeq2SeqLM.from_pretrained`. Defaults to None.

        Raises:
            ValueError: If `source_lang` or `target_lang` are empty, or if
                `max_length` or `num_beams` are not positive integers.

        """
        self._validate_language_pair(source_lang, target_lang)
        self._validate_generation_params(max_length, num_beams)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = torch.device(device)

        tk_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, **tk_kwargs
        )
        self.tokenizer.source_lang = self.source_lang

        md_kwargs = model_kwargs or {}
        self.model: PreTrainedModel = (
            AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME, **md_kwargs)
            .to(self.device)
            .eval()
        )

    def translate(self, text: str) -> str:
        """Translate a single sentence using the NLLB model.

        Args:
            text (str): The input sentence in the source language.

        Returns:
            str: The translated sentence.

        Raises:
            TranslationError: If `text` is empty or generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise TranslationError("Input text must be a non-empty string")

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        forced_bos = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        # Generate
        try:
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        except Exception as e:
            raise TranslationError(f"NLLB generation failed: {e}") from e

        # Decode and clean up
        translated = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return translated
