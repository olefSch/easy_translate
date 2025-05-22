import pytest

from unittest.mock import MagicMock
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.translator_base import TranslatorBase
from src.huggingface_translator_base import HuggingFaceTranslator


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
