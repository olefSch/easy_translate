import pytest
from typing import Iterable
from easy_nlp_translate.prompt_config import PromptStyle


# --- init testing
def test_base_init(patched_llm_translator_class):
    """
    Test the basic init functionality of the LLMTranslator class.
    Without input vars to use base values.
    """
    # Create an instance of the LLMTranslator class
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
    )

    assert translator.model_name == "model_a"
    assert translator.target_lang == "en"
    assert translator.source_lang is None
    assert translator.prompt_style == PromptStyle.DEFAULT


def test_init_with_costum_params(patched_llm_translator_class):
    """
    Test the init functionality of the LLMTranslator class.
    With input vars to use costum values.
    """
    # Create an instance of the LLMTranslator class
    translator = patched_llm_translator_class(
        model_name="model_b",
        target_lang="de",
        source_lang="en",
        prompt_type="formal",
    )

    assert translator.model_name == "model_b"
    assert translator.target_lang == "de"
    assert translator.source_lang == "en"
    assert translator.prompt_style == PromptStyle.FORMAL


def test_init_with_invalid_model_name(patched_llm_translator_class):
    """
    Test the init functionality of the LLMTranslator class.
    With invalid model name.
    """
    with pytest.raises(ValueError) as excinfo:
        patched_llm_translator_class(
            model_name="non_existent_model",
            target_lang="en",
        )
    assert "Model 'non_existent_model' is not available." in str(excinfo.value)
    assert "Available models are: ['model_a', 'model_b']" in str(excinfo.value)


def test_init_with_invalid_prompt_style(patched_llm_translator_class):
    """
    Test the init functionality of the LLMTranslator class.
    With invalid prompt style.
    """
    with pytest.raises(ValueError) as excinfo:
        patched_llm_translator_class(
            model_name="model_a",
            target_lang="en",
            prompt_type="invalid_prompt_style",
        )
    assert (
        "Prompt type 'invalid_prompt_style' is not available. Avaliable types are: "
        in str(excinfo.value)
    )


def test_init_wrong_temperature(patched_llm_translator_class):
    """
    Test the init functionality of the LLMTranslator class.
    With wrong temperature value.
    """
    with pytest.raises(ValueError) as excinfo:
        patched_llm_translator_class(
            model_name="model_a",
            target_lang="en",
            temperature=-0.5,
        )
    assert "Temperature must be between 0 and 1." in str(excinfo.value)


def test_init_wrong_max_tokens(patched_llm_translator_class):
    """
    Test the init functionality of the LLMTranslator class.
    With wrong max_tokens value.
    """
    with pytest.raises(ValueError) as excinfo:
        patched_llm_translator_class(
            model_name="model_a",
            target_lang="en",
            max_tokens=0,
        )
    assert "max_tokens must be greater than 0." in str(excinfo.value)


# --- render prompt
def test_render_prompt(patched_llm_translator_class):
    """
    Test the render_prompt method of the LLMTranslator class.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    text_to_translate = "Hello World"
    rendered_prompt = translator._render_prompt(text_to_translate)

    assert text_to_translate in rendered_prompt
    assert "German" in rendered_prompt
    assert "English" in rendered_prompt


def test_render_prompt_without_source_lang(patched_llm_translator_class):
    """
    Test the render_prompt method of the LLMTranslator class.
    Without source lang.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        prompt_type="default",
    )

    text_to_translate = "Hallo Welt"
    rendered_prompt = translator._render_prompt(text_to_translate)

    assert text_to_translate in rendered_prompt
    assert "English" in rendered_prompt
    assert "German" in rendered_prompt


def test_reder_prompt_with_custom_prompt(patched_llm_translator_class):
    """
    Test the render_prompt method of the LLMTranslator class.
    With custom prompt.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="custom",
        costum_prompt="translate like Lothar Matthäus",
    )

    text_to_translate = "Hallo Welt"
    rendered_prompt = translator._render_prompt(text_to_translate)

    assert "translate like Lothar Matthäus" in rendered_prompt


# --- translate
def test_translate(patched_llm_translator_class):
    """
    Test the translate method of the LLMTranslator class.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    text_to_translate = "Hallo Welt"
    translated_text = translator.translate(text_to_translate)

    assert translated_text == "my mocked translation"


def test_translate_with_wrong_input_text(patched_llm_translator_class):
    """
    Test the translate method of the LLMTranslator class.
    With wrong input text.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    text_to_translate = ""
    with pytest.raises(ValueError) as excinfo:
        translator.translate(text_to_translate)

    assert "Text to translate must be a non-empty string." in str(
        excinfo.value
    )


# --- test raw mocked stuff
def test_get_credentials(patched_llm_translator_class):
    """
    Test the get_credentials method of the LLMTranslator class.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    credentials = translator._get_credentials()
    assert credentials == {"api_key": "dummy_key"}


def test_model_init(patched_llm_translator_class):
    """
    Test the model initialization of the LLMTranslator class.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    assert translator.model is not None


def test_gnerate(patched_llm_translator_class):
    """
    Test the generate method of the LLMTranslator class.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    input_text = "Hallo Welt"
    generated_text = translator._generate(input_text)
    assert generated_text == ["my mocked translation"]
    assert isinstance(generated_text, Iterable)


def test_post_process(patched_llm_translator_class):
    """
    Test the post_process method of the LLMTranslator class.
    """
    translator = patched_llm_translator_class(
        model_name="model_a",
        target_lang="en",
        source_lang="de",
        prompt_type="default",
    )

    raw_response = ["my mocked translation"]
    post_processed_text = translator._post_process(raw_response)
    assert post_processed_text == "my mocked translation"
