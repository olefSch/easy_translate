from typing import Iterable
import pytest
import os

from unittest.mock import MagicMock
from transformers import PreTrainedTokenizer, PreTrainedModel

from easy_nlp_translate.translator_base import TranslatorBase
from easy_nlp_translate.llm_translator_base import LLMTranslator
from easy_nlp_translate.huggingface_translator_base import (
    HuggingFaceTranslator,
)
from easy_nlp_translate.huggingface_models import MBARTTranslator


# --- initilize_translator function
@pytest.fixture(autouse=True)
def unregister_ollama_translator_for_test(monkeypatch):
    """
    Unregister the 'ollama' translator from the registry for testing purposes.
    This fixture ensures that the 'ollama' translator is not available during tests,
    allowing for testing of other translators without interference.
    """
    is_ci_environment = os.getenv("CI", "false").lower() in ("true", "1")
    if is_ci_environment:
        registry_path = "easy_nlp_translate.initialize.TRANSLATOR_REGISTRY" 
    
        module_path, object_name = registry_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[object_name])
        original_registry = getattr(module, object_name)

        if "ollama" in original_registry:
            monkeypatch.delitem(original_registry, "ollama")

# --- PromptStyle Enum
@pytest.fixture
def expected_codes():
    """
    Fixture that provides the expected prompt style codes.
    """
    return [
        "default",
        "formal",
        "translate_and_summarize",
        "formal_translate_and_summarize",
        "romantic",
        "poetic",
    ]


# --- TranslatorBase Class
class ConcreteTranslator(TranslatorBase):
    """
    A concrete implementation of TranslatorBase for testing purposes.
    """

    LANGUAGE_CODES = ["en", "de", "custom_test_lang"]

    def translate(self, text: str) -> str:
        """Dummy translate method for testing."""
        return f"translated_{self.target_lang}:{text}"


@pytest.fixture
def concrete_translator_class():
    """Pytest fixture that provides the ConcreteTranslator class."""
    return ConcreteTranslator


@pytest.fixture
def translator_base_class():
    """Pytest fixture that provides the TranslatorBase class."""
    return TranslatorBase


# --- HuggingFaceTranslator Class
class ConcreteHuggingFaceTranslator(HuggingFaceTranslator):
    """
    A concrete implementation of HuggingFaceTranslator for testing purposes.
    """

    def translate(self, text: str) -> str:
        """Dummy translate method for testing."""
        return f"translated_{self.target_lang}:{text}"


@pytest.fixture
def mock_tokenizer_instance():
    """
    Provides a MagicMock instance of a PreTrainedTokenizer.
    """
    return MagicMock(spec=PreTrainedTokenizer)


@pytest.fixture
def mock_model_instance():
    """
    Provides a MagicMock instance of a PreTrainedModel.
    """
    return MagicMock(spec=PreTrainedModel)


@pytest.fixture
def patched_huggingface_translator_class(
    mocker, mock_tokenizer_instance, mock_model_instance
):
    """
    Provides the HuggingFaceTranslator class with _init_tokenizer, _init_model
    """
    mocker.patch.object(
        ConcreteHuggingFaceTranslator,
        "_init_tokenizer",
        return_value=mock_tokenizer_instance,
    )
    mocker.patch.object(
        ConcreteHuggingFaceTranslator,
        "_init_model",
        return_value=mock_model_instance,
    )
    return ConcreteHuggingFaceTranslator


# --- LLM Translator Class
class ConcreteLLMTranslator(LLMTranslator):
    """
    A concrete implementation of LLMTranslator for testing purposes.
    """

    AVAILABLE_MODELS = ["model_a", "model_b"]
    LANGUAGE_CODES = ["en", "de", "custom_test_lang"]

    def _init_model(self):
        """Dummy method to initialize the model."""
        return MagicMock()

    def _get_credentials(self):
        """Dummy method to get credentials."""
        return {"api_key": "dummy_key"}

    def _generate(self, input: str) -> Iterable:
        return ["my mocked translation"]

    def _post_process(self, raw_response: Iterable) -> str:
        return next(iter(raw_response))


@pytest.fixture
def patched_llm_translator_class():
    """
    Provides the LLMTranslator class with _init_tokenizer, _init_model
    """
    return ConcreteLLMTranslator


# --- MBartTranslator Class
@pytest.fixture
def get_mbart():
    """
    Provides the MBartTranslator class with _init_tokenizer, _init_model
    """
    return MBARTTranslator
