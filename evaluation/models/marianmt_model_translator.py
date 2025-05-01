import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (MarianMTModel, MarianTokenizer, PreTrainedModel,
                          PreTrainedTokenizer)

from .base_translator import BaseTranslator, TranslationError

logger = logging.getLogger(__name__)


class MarianTranslator(BaseTranslator):
    """
    Translator using Hugging Face’s MarianMT model for multilingual translation
    """

    MODEL_NAME_TEMPLATE = "Helsinki-NLP/opus-mt-{source}-{target}"

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "de",
        device: Union[str, torch.device] = "cpu",
        max_length: int = 512,
        num_beams: int = 4,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the MarianMT translator.

        Args:
            source_lang (str): Name of the source language (for consistency).
            target_lang (str): Name of the target language.
            device (Union[str, torch.device]): "cpu", "cuda", or a torch.device.
                Defaults to "cpu".
            max_length (int): Maximum length of generated sequences.
                Defaults to 512.
            num_beams (int): Number of beams for beam search.
                Defaults to 4.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Extra kwargs passed to
                `MarianTokenizer.from_pretrained`. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]]): Extra kwargs passed to
                `MarianMTModel.from_pretrained`. Defaults to None.

        Raises:
            ValueError: If `source_lang` or `target_lang` are empty,
                or if `max_length` or `num_beams` are not positive.
        """
        self._validate_language_pair(source_lang, target_lang)
        self._validate_generation_params(max_length, num_beams)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = torch.device(device)

        model_name = self.MODEL_NAME_TEMPLATE.format(
            source=source_lang, target=target_lang
        )

        tk_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = MarianTokenizer.from_pretrained(
            model_name, **tk_kwargs
        )

        md_kwargs = model_kwargs or {}
        self.model: PreTrainedModel = (
            MarianMTModel.from_pretrained(model_name, **md_kwargs)
            .to(self.device)
            .eval()
        )

    def translate(self, text: str) -> str:
        """Translate a single sentence using the MarianMT model.

        Args:
            text (str): The input sentence in the source language.

        Returns:
            str: The translated sentence.

        Raises:
            TranslationError: If `text` is empty or generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise TranslationError("Input text must be a non-empty string")

        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(
            self.device
        )

        # Generate
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )
        except Exception as e:
            logger.error("MarianMT generation error: %s", e)
            raise TranslationError(f"Translation failed: {e}") from e

        # Decode and clean up
        tranlated = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return tranlated
