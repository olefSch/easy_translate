import logging
from typing import Any

from .translator_base import TranslatorBase
from .huggingface_models import MBARTTranslator

logger = logging.getLogger(__name__)

TRANSLATOR_REGISTRY: dict[str, type[TranslatorBase]] = {
    "mbart": MBARTTranslator,
}


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
