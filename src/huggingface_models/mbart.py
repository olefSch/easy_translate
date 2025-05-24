import logging
from typing import Optional, Any

import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

from ..huggingface_translator_base import HuggingFaceTranslator
from ..translator_base import TranslatorBase
from ..config import generic_to_mbart_code_map

logger = logging.getLogger(__name__)


class MBARTTranslator(HuggingFaceTranslator):
    """
    MBART model for translation.
    """

    MODEL_NAME: str = "facebook/mbart-large-50-many-to-many-mmt"
    CODE_MAPPER: dict[str, str] = generic_to_mbart_code_map

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _convert_lang_code(self, lang_code: str) -> str:
        """
        Convert a generic language code to the MBART-specific code.

        Args:
            lang_code (str): The generic language code.

        Returns:
            str: The MBART-specific language code.
        """
        if lang_code not in self.CODE_MAPPER:
            raise ValueError(
                f"Language code '{lang_code}' is not supported. Supported codes: {list(self.CODE_MAPPER.keys())}"
            )
        return self.CODE_MAPPER[lang_code]

    def _init_tokenizer(
        self, tokenizer_kwargs: Optional[dict[str, Any]] = None
    ) -> MBart50Tokenizer:
        """
        Initialize the tokenizer.

        Args:
            tokenizer_kwargs (Optional[dict[str, Any]]): Additional arguments for the tokenizer.
        """
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        tokenizer_kwargs["model_max_length"] = self.max_length

        tokenizer = MBart50Tokenizer.from_pretrained(
            self.MODEL_NAME, **tokenizer_kwargs
        )

        if self.source_lang is not None:
            tokenizer.src_lang = self._convert_lang_code(self.source_lang)
            logger.info(f"Setting source language to: {tokenizer.src_lang}")

        return tokenizer

    def _init_model(
        self, model_kwargs: Optional[dict[str, Any]] = None
    ) -> MBartForConditionalGeneration:
        """
        Initialize the model.

        Args:
            model_kwargs (Optional[dict[str, Any]]): Additional arguments for the model.
        """
        if model_kwargs is None:
            model_kwargs = {}

        return MBartForConditionalGeneration.from_pretrained(
            self.MODEL_NAME, **model_kwargs
        )

    def translate(self, text: str) -> str:
        """
        Translate the input text.

        Args:
            text (str): The text to translate.

        Returns:
            str: The translated text.
        """
        TranslatorBase._validate_basic_text_to_translate(text)

        if self.source_lang is None:
            self.tokenizer.src_lang = self._convert_lang_code(
                self.detect_language(text)
            )
            logger.info(f"Detected source language: {self.tokenizer.src_lang}")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        logger.debug(f"Tokenized inputs: {inputs}")

        forced_bos_token_id = self.tokenizer.lang_code_to_id.get(
            self._convert_lang_code(self.target_lang)
        )
        logger.info(
            f"Using forced_bos_token_id: {forced_bos_token_id} for target language: {self.target_lang}"
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=self.num_beams,
                max_length=self.max_length,
                early_stopping=True,
            )
            logger.debug(f"Generated token IDs: {outputs}")

        output = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

        logger.debug(f"Output: {output}")

        return output
