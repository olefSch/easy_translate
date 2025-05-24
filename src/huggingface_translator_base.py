import logging
from .translator_base import TranslatorBase
from typing import Optional, Union, Any

from transformers import PreTrainedTokenizer, PreTrainedModel

import torch

logger = logging.getLogger(__name__)


class HuggingFaceTranslator(TranslatorBase):
    """
    A base class for Hugging Face-based translators, inheriting from TranslatorBase.
    """

    def __init__(
        self,
        target_lang: str,
        source_lang: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "cuda"
        if torch.cuda.is_available()
        else "cpu",
        max_length: Optional[int] = 512,
        num_beams: Optional[int] = 4,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initializes the HuggingFaceTranslator with target and optional source languages,
        device, maximum length, number of beams, and additional arguments for the tokenizer and model.

        Args:
            target_lang (str): The target language code for translation (e.g., 'fr' for French).
            source_lang (Optional[str]): The source language code for translation (e.g., 'en' for English).
                Defaults to None, implying auto-detection will be attempted.
            device (Optional[Union[str, torch.device]]): The device to use for the model.
                Defaults to "cuda" if available, otherwise "cpu".
            max_length (Optional[int]): The maximum length of the generated text.
                Defaults to 512.
            num_beams (Optional[int]): The number of beams for beam search.
                Defaults to 4.
            tokenizer_kwargs (Optional[dict[str, Any]]): Additional arguments for the tokenizer.
            model_kwargs (Optional[dict[str, Any]]): Additional arguments for the model.
        """
        super().__init__(target_lang, source_lang)

        self._validate_generation_parameters(max_length, num_beams)

        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams

        self.tokenizer: PreTrainedTokenizer = self._init_tokenizer(
            tokenizer_kwargs
        )
        self.model: PreTrainedModel = self._init_model(model_kwargs)

        logger.info(
            f"Initialized {self.__class__.__name__} with target language '{target_lang}'"
        )

    def _validate_generation_parameters(
        self, max_length: Optional[int], num_beams: Optional[int]
    ):
        """
        Validates the generation parameters.
        Args:
            max_length (Optional[int]): The maximum length of the generated text.
            num_beams (Optional[int]): The number of beams for beam search.
        Raises:
            ValueError: If max_length or num_beams is not a positive integer.
        """
        if max_length is not None and (
            not isinstance(max_length, int) or max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer.")
        if num_beams is not None and (
            not isinstance(num_beams, int) or num_beams <= 0
        ):
            raise ValueError("num_beams must be a positive integer.")

    def _init_tokenizer(
        self, tokenizer_kwargs: Optional[dict[str, Any]] = None
    ) -> PreTrainedTokenizer:
        """
        Initialize the tokenizer.

        Args:
            tokenizer_kwargs (Optional[dict[str, Any]]): Additional arguments for the tokenizer.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )

    def _init_model(
        self, model_kwargs: Optional[dict[str, Any]] = None
    ) -> PreTrainedModel:
        """
        Initialize the model.

        Args:
            model_kwargs (Optional[dict[str, Any]]): Additional arguments for the model.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )
