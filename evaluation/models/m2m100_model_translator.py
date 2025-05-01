import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (M2M100ForConditionalGeneration, M2M100Tokenizer,
                          PreTrainedModel, PreTrainedTokenizer)

from .base_translator import BaseTranslator

logger = logging.getLogger(__name__)


class M2M100Translator(BaseTranslator):
    """
    Translator using Hugging Face’s M2M100 model for multilingual translation
    """

    MODEL_NAME: str = "facebook/m2m100_418M"

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "de",
        device: Union[str, torch.device] = "cpu",
        max_length: int = 512,
        num_beams: int = 4,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the M2M100_418M translator.

        Args:
            source_lang (str): Source language code (e.g. "en").
            target_lang (str): Target language code (e.g. "de").
            device (Union[str, torch.device], optional): "cpu", "cuda",
                or a torch.device. Defaults to "cpu".
            max_length (int, optional): Maximum length of generated sequences.
                Defaults to 512.
            num_beams (int, optional): Number of beams for beam‐search.
                Defaults to 4.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Extra kwargs for
                `M2M100Tokenizer.from_pretrained`. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]]): Extra kwargs for
                `M2M100ForConditionalGeneration.from_pretrained`. Defaults to None.

        Raises:
            ValueError: If source_lang or target_lang are empty,
                or if max_length or num_beams are not positive.
        """
        if not source_lang or not target_lang:
            raise ValueError("Both `source_lang` and `target_lang` are required")
        if max_length <= 0:
            raise ValueError("`max_length` must be > 0")
        if num_beams <= 0:
            raise ValueError("`num_beams` must be > 0")

        self.device = torch.device(device)

        tk_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = M2M100Tokenizer.from_pretrained(
            self.MODEL_NAME, **tk_kwargs
        )
        self.tokenizer.src_lang = source_lang

        md_kwargs = model_kwargs or {}
        self.model: PreTrainedModel = (
            M2M100ForConditionalGeneration.from_pretrained(self.MODEL_NAME, **md_kwargs)
            .to(self.device)
            .eval()
        )

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.num_beams = num_beams

    def translate(self, text: str) -> str:
        """Translate a single text string from source_lang → target_lang.

        Args:
            text (str): Input sentence in the source language.

        Returns:
            str: Translated sentence (no special tokens).

        Raises:
            ValueError: If `text` is empty.
            RuntimeError: If generation fails.
        """
        if not text:
            raise ValueError("Input `text` must not be empty")

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        forced_bos = self.tokenizer.get_lang_id(self.target_lang)

        # Generate
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )
        except Exception as e:
            logger.error("M2M100 generation error: %s", e)
            raise RuntimeError(f"Translation failed: {e}") from e

        # Decode and clean up
        translated = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return translated
