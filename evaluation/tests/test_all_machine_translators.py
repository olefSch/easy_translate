import pytest

from evaluation.models.base_translator import TranslationError
from evaluation.models.m2m100_model_translator import M2M100Translator
from evaluation.models.marianmt_model_translator import MarianTranslator
from evaluation.models.mBART_50_model_translator import MBartTranslator
from evaluation.models.nllb_model_translator import NllbTranslator

TRANSLATORS = [
    (MarianTranslator, {"source_lang": "en", "target_lang": "de", "device": "cpu"}),
    (M2M100Translator, {"source_lang": "en", "target_lang": "de", "device": "cpu"}),
    (
        MBartTranslator,
        {"source_lang": "en_XX", "target_lang": "de_DE", "device": "cpu"},
    ),
    (
        NllbTranslator,
        {"source_lang": "eng_Latn", "target_lang": "deu_Latn", "device": "cpu"},
    ),
]


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_identity_short_circuits(cls, kwargs):
    t = cls(**kwargs)
    sample = "Fernweh"
    t.source_lang = t.target_lang
    assert t.translate(sample) == sample


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_empty_input_raises(cls, kwargs):
    t = cls(**kwargs)
    with pytest.raises(TranslationError):
        t.translate("")
    with pytest.raises(TranslationError):
        t.translate("   ")


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_unsupported_language_code_raises(cls, kwargs):
    bogus = dict(kwargs, source_lang="xx", target_lang="yy")
    with pytest.raises(Exception):
        cls(**bogus).translate("hello")


@pytest.mark.parametrize("cls,kwargs", TRANSLATORS)
def test_basic_translation_works(cls, kwargs):
    t = cls(**kwargs)
    source = "Hello, world!"
    result = t.translate(source)
    assert isinstance(result, str)
    assert result.strip() != "" and result.strip().lower() != source.lower()
