import pytest

from evaluation.models.base_translator import TranslationError
from evaluation.models.m2m100_model_translator import M2M100Translator
from evaluation.models.marianmt_model_translator import MarianTranslator
from evaluation.models.mBART_50_model_translator import MBartTranslator
from evaluation.models.nllb_model_translator import NllbTranslator

# Parameterized list of translator classes with appropriate language configurations
TRANSLATORS = [
    (
        MarianTranslator,
        {"source_lang": "en", "target_lang": "de", "device": "cpu"},
    ),
    (
        M2M100Translator,
        {"source_lang": "en", "target_lang": "de", "device": "cpu"},
    ),
    (
        MBartTranslator,
        {"source_lang": "en_XX", "target_lang": "de_DE", "device": "cpu"},
    ),
    (
        NllbTranslator,
        {
            "source_lang": "eng_Latn",
            "target_lang": "deu_Latn",
            "device": "cpu",
        },
    ),
]


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_identity_short_circuits(cls, kwargs):
    # If source and target languages are the same, translation should be a no-op
    t = cls(**kwargs)
    sample = "Fernweh"
    t.source_lang = t.target_lang  # Simulate identity translation scenario
    assert t.translate(sample) == sample


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_empty_input_raises(cls, kwargs):
    # Ensure empty or whitespace-only inputs raise a TranslationError
    t = cls(**kwargs)
    with pytest.raises(TranslationError):
        t.translate("")
    with pytest.raises(TranslationError):
        t.translate("   ")


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_unsupported_language_code_raises(cls, kwargs):
    # Try initializing the translator with unsupported language codes
    bogus = dict(kwargs, source_lang="xx", target_lang="yy")
    with pytest.raises(Exception):
        cls(**bogus).translate("hello")


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_basic_translation_works(cls, kwargs):
    # Ensure that a valid translation returns a non-empty, transformed string
    t = cls(**kwargs)
    source = "Hello, world!"
    result = t.translate(source)

    # The result should be a non-empty string and not just echo the input
    assert isinstance(result, str)
    assert result.strip() != "" and result.strip().lower() != source.lower()
