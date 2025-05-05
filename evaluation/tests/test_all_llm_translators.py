from functools import partial

import pytest

from evaluation.models.base_translator import TranslationError
from evaluation.models.llm_model_translator import LLMTranslator

LLM_TRANSLATORS = [
    ("llama3.2", partial(LLMTranslator, model_name="llama3.2:3b")),
    ("llama3.1", partial(LLMTranslator, model_name="llama3.1:8b")),
    ("gemma", partial(LLMTranslator, model_name="gemma3:4b")),
    ("phi3", partial(LLMTranslator, model_name="phi3:3.8b")),
    ("mistral", partial(LLMTranslator, model_name="mistral:7b")),
]


@pytest.mark.parametrize("name,factory", LLM_TRANSLATORS)
def test_prompt_formatting(name, factory, monkeypatch):
    captured = {}

    def fake_generate(model, prompt, options):
        captured["prompt"] = prompt
        return {"response": "Hallo, Welt!"}

    t = factory(source_lang="English", target_lang="German")
    monkeypatch.setattr(t.client, "generate", fake_generate)

    src = "Hello, world!"
    t.translate(src)

    assert "English" in captured["prompt"]
    assert "German" in captured["prompt"]
    assert src in captured["prompt"]


@pytest.mark.parametrize("name,factory", LLM_TRANSLATORS)
def test_unsupported_language_raises(name, factory):
    with pytest.raises(ValueError):
        factory(source_lang="", target_lang="German")
    with pytest.raises(ValueError):
        factory(source_lang="English", target_lang="")
    with pytest.raises(ValueError):
        factory(source_lang="", target_lang="")


@pytest.mark.parametrize("name,factory", LLM_TRANSLATORS)
def test_basic_translation_works(name, factory):
    t = factory(source_lang="English", target_lang="German")
    source = "Hello, world!"
    result = t.translate(source)
    assert isinstance(result, str)
    assert result.strip() != "" and result.strip().lower() != source.lower()


@pytest.mark.parametrize("name,factory", LLM_TRANSLATORS)
def test_result_is_stripped(name, factory, monkeypatch):
    def fake_generate(model, prompt, options):
        return {"response": "  Hallo, Welt!  \n"}

    t = factory()
    monkeypatch.setattr(t.client, "generate", fake_generate)

    out = t.translate("Hello")
    assert out == "Hallo, Welt!"


@pytest.mark.parametrize("name,factory", LLM_TRANSLATORS)
def test_client_exception_becomes_translation_error(name, factory, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    t = factory()
    monkeypatch.setattr(t.client, "generate", boom)
    with pytest.raises(TranslationError):
        t.translate("Hello")
