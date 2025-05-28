import logging
from typing import Any, Optional, overload, Literal

from .translator_base import TranslatorBase
from .huggingface_models import MBARTTranslator
from .llm_translator_base import LLMTranslator
from .llm_provider import (
    GeminiTranslator,
    GPTTranslator,
    ClaudeTranslator,
    OllamaTranslator,
)

logger = logging.getLogger(__name__)

TRANSLATOR_REGISTRY: dict[str, type[TranslatorBase]] = {
    "mbart": MBARTTranslator,
    "gemini": GeminiTranslator,
    "gpt": GPTTranslator,
    "claude": ClaudeTranslator,
    "ollama": OllamaTranslator,
}


@overload
def initialize_translator(
    translator_name: Literal["mbart"],
    target_lang: str,
    source_lang: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = 512,
    num_beams: Optional[int] = 4,
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> MBARTTranslator: ...


@overload
def initialize_translator(
    translator_name: Literal["gemini", "gpt", "claude", "ollama"],
    model_name: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    prompt_type: str = "default",
    costum_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> LLMTranslator: ...


def initialize_translator(
    translator_name: str,
    *args: Any,
    **kwargs: Any,
) -> TranslatorBase:
    """
    Initialize a translator based on the provided name.

    Args:
        translator_name (str): The name of the translator to initialize.
        *args: Positional arguments for the translator's constructor.
        **kwargs: Keyword arguments for the translator's constructor.

    Returns:
        TranslatorBase: An instance of the specified translator.
    """
    if translator_name not in TRANSLATOR_REGISTRY:
        raise ValueError(
            f"Unknown translator name: {translator_name}. Available: {list(TRANSLATOR_REGISTRY.keys())}"
        )

    translator_class = TRANSLATOR_REGISTRY[translator_name]

    return translator_class(*args, **kwargs)
