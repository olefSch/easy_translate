import logging
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

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
        # Validate provided language codes and generation settings
        self._validate_language_pair(source_lang, target_lang)
        self._validate_generation_params(max_length, num_beams)

        # Store config and target device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = torch.device(device)

        # Format the model name using the language codes
        model_name = self.MODEL_NAME_TEMPLATE.format(
            source=source_lang, target=target_lang
        )

        # Load tokenizer with optional customization
        tk_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = MarianTokenizer.from_pretrained(
            model_name, **tk_kwargs
        )

        # Load model with optional customization and put it on the specified device
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
        # Ensure non-empty input
        if not isinstance(text, str) or not text.strip():
            raise TranslationError("Input text must be a non-empty string")

        # Tokenize input and send to device
        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(
            self.device
        )

        # Generate translated tokens with beam search
        try:
            with torch.no_grad():  # Disable gradient computation for faster inference
                output_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,  # Stop beam search early if all beams end
                )
        except Exception as e:
            # Wrap any errors from the model in a custom TranslationError
            logger.error("MarianMT generation error: %s", e)
            raise TranslationError(f"Translation failed: {e}") from e

        # Decode token IDs into human-readable text and clean whitespace
        tranlated = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return tranlated
