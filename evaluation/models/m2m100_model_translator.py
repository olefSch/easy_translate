import logging
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .base_translator import BaseTranslator, TranslationError

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
            max_length (int): Maximum length of generated sequences.
                Defaults to 512.
            num_beams (int): Number of beams for beam‐search.
                Defaults to 4.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Extra kwargs for
                `M2M100Tokenizer.from_pretrained`. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]]): Extra kwargs for
                `M2M100ForConditionalGeneration.from_pretrained`. Defaults to None.

        Raises:
            ValueError: If source_lang or target_lang are empty,
                or if max_length or num_beams are not positive.
        """
        # Validate provided language codes and generation settings
        self._validate_language_pair(source_lang, target_lang)
        self._validate_generation_params(max_length, num_beams)

        # Prepare target device
        self.device = torch.device(device)

        # Load tokenizer with optional customization
        tk_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = M2M100Tokenizer.from_pretrained(
            self.MODEL_NAME, **tk_kwargs
        )
        self.tokenizer.src_lang = source_lang  # Set source language

        # Load model with optional customization and put it on the specified device
        md_kwargs = model_kwargs or {}
        self.model: PreTrainedModel = (
            M2M100ForConditionalGeneration.from_pretrained(self.MODEL_NAME, **md_kwargs)
            .to(self.device)
            .eval()
        )

        # Store config for use during generation
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.num_beams = num_beams

    def translate(self, text: str) -> str:
        """Translate a single sentence using the M2M100 model.

        Args:
            text (str): Input sentence in the source language.

        Returns:
            str: Translated sentence.

        Raises:
            TranslationError: If `text` is empty or generation fails.
        """
        # Ensure non-empty input
        if not isinstance(text, str) or not text.strip():
            raise TranslationError("Input text must be a non-empty string")

        # Tokenize input and send to device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        # Get ID of target language to force decoder to generate in that language
        forced_bos = self.tokenizer.get_lang_id(self.target_lang)

        # Generate translated tokens with beam search
        try:
            with torch.no_grad():  # Disable gradient computation for faster inference
                output_ids = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,  # Stop beam search early if all beams end
                )
        except Exception as e:
            # Wrap any errors from the model in a custom TranslationError
            logger.error("M2M100 generation error: %s", e)
            raise TranslationError(f"Translation failed: {e}") from e

        # Decode token IDs into human-readable text and clean whitespace
        translated = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return translated
