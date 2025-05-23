from typing import Iterable
import pytest

from unittest.mock import MagicMock
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.translator_base import TranslatorBase
from src.llm_translator_base import LLMTranslator
from src.huggingface_translator_base import HuggingFaceTranslator
from src.huggingface_models import MBARTTranslator


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

    def _init_model(self, model_name: str):
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
