"""Global configuration for translation evaluation."""

from functools import partial
from pathlib import Path
from typing import List

from models import (LLMTranslator, M2M100Translator, MarianTranslator,
                    MBartTranslator, NllbTranslator, T5Translator)

MODEL_REGISTRY = {
    "nllb": NllbTranslator,
    "m2m100": M2M100Translator,
    "mbart50": MBartTranslator,
    "marian": MarianTranslator,
    "t5": T5Translator,
    "llama3.2": partial(LLMTranslator, model_name="llama3.2:3b"),
    "llama3.1": partial(LLMTranslator, model_name="llama3.1:8b"),
    "gemma": partial(LLMTranslator, model_name="gemma3:4b"),
    "phi3": partial(LLMTranslator, model_name="phi3:3.8b"),
    "mistral": partial(LLMTranslator, model_name="mistral:7b"),
}

MODELS_TO_EVALUATE: List[str] = [
    "mistral",
    "m2m100",
    "marian",
    "t5",
    "nllb",
    "mbart50",
    "llama3.1",
    "llama3.2",
    "gemma",
    "phi3",
]

LANGUAGE_MAPPING_PATH: Path = Path("configs/language_mappings.yaml")

DATASET_NAME: str = "wmt19"
DATASET_SPLIT: str = "train[:1000]"

OUTPUT_DIR: Path = Path("reports")
