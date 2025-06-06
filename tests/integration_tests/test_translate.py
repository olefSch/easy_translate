import pytest
import os
from easy_nlp_translate import initialize_translator


# --- detector TESTS ---
def test_detect_langauge():
    """
    Test the language detector with a sample text and check if the detected language is correct.
    """
    translator = initialize_translator(
        translator_name="mbart", source_lang="en", target_lang="de"
    )

    text = "This is a dog."
    detected_language = translator.detect_language(text)

    assert detected_language == "en"


# --- MBART TESTS ---
def test_mbart_translator_with_default_model():
    """
    Test the MBART translator with default model and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="mbart", source_lang="en", target_lang="de"
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert translated_text == "Dies ist ein Hund."


def test_mbart_translator_with_no_input_language():
    """
    Test the MBART translator with no input language and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="mbart", target_lang="de"
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert translated_text == "Dies ist ein Hund."


def test_mbart_translator_with_custom_input_values():
    """
    Test the MBART translator with custom input values and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="mbart",
        source_lang="en",
        target_lang="de",
        max_length=50,
        num_beams=5,
        device="cpu",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert translated_text == "Dies ist ein Hund."


# --- LLM TESTS ---
# --- Gemini TESTS ---


def test_gemini_translator_with_default_model():
    """
    Test the Gemini translator with default model and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="gemini",
        model_name="gemini-1.5-flash",
        target_lang="de",
        source_lang="en",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_gemini_translator_with_no_input_language():
    """
    Test the Gemini translator with no input language and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="gemini",
        model_name="gemini-1.5-flash",
        target_lang="de",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_gemini_translator_with_custom_prompt():
    """
    Test the Gemini translator with custom input values and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="gemini",
        model_name="gemini-1.5-flash",
        source_lang="en",
        target_lang="de",
        prompt_type="romantic",
        temperature=0.7,
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_gemini_translator_wrong_model_name():
    """
    Test the Gemini translator with an invalid model name and check if it raises a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        initialize_translator(
            translator_name="gemini",
            model_name="invalid-model-name",
            target_lang="de",
            source_lang="en",
        )

    assert "Model 'invalid-model-name' is not available." in str(excinfo.value)


# --- OPENAI TESTS ---
def test_gpt_translator_with_default_model():
    """
    Test the GPT translator with default model and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="gpt",
        model_name="gpt-4.1-mini",
        target_lang="de",
        source_lang="en",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_gpt_translator_with_no_input_language():
    """
    Test the GPT translator with no input language and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="gpt",
        model_name="gpt-4.1-mini",
        target_lang="de",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_gpt_translator_with_custom_prompt():
    """
    Test the GPT translator with custom input values and check if the translation is not empty.
    """
    translator = initialize_translator(
        translator_name="gpt",
        model_name="gpt-4.1-mini",
        source_lang="en",
        target_lang="de",
        prompt_type="romantic",
        temperature=0.7,
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_gpt_translator_wrong_model_name():
    """
    Test the GPT translator with an invalid model name and check if it raises a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        initialize_translator(
            translator_name="gpt",
            model_name="invalid-model-name",
            target_lang="de",
            source_lang="en",
        )

    assert "Model 'invalid-model-name' is not available." in str(excinfo.value)


# --- CLAUDE TESTS ---
def test_claude_translator_with_default_model():
    translator = initialize_translator(
        translator_name="claude",
        model_name="claude-3-5-haiku-latest",
        target_lang="de",
        source_lang="en",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_claude_translator_with_no_input_language():
    translator = initialize_translator(
        translator_name="claude",
        model_name="claude-3-5-haiku-latest",
        target_lang="de",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_claude_translator_with_custom_prompt():
    translator = initialize_translator(
        translator_name="claude",
        model_name="claude-3-5-haiku-latest",
        source_lang="en",
        target_lang="de",
        prompt_type="romantic",
        temperature=0.7,
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


def test_claude_translator_wrong_model_name():
    with pytest.raises(ValueError) as excinfo:
        initialize_translator(
            translator_name="claude",
            model_name="invalid-model-name",
            target_lang="de",
            source_lang="en",
        )

    assert "Model 'invalid-model-name' is not available." in str(excinfo.value)


# --- OLLAMA TESTS ---

skip_ollama_tests_in_ci = pytest.mark.skipif(
    os.getenv("CI", "false").lower() in ("true", "1"),
    reason="Ollama tests are skipped in CI environment",
)


@skip_ollama_tests_in_ci
def test_ollama_translator_with_default_model():
    translator = initialize_translator(
        translator_name="ollama",
        model_name="llama3.2:latest",
        target_lang="de",
        source_lang="en",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


@skip_ollama_tests_in_ci
def test_ollama_translator_with_no_input_language():
    translator = initialize_translator(
        translator_name="ollama",
        model_name="llama3.2:latest",
        target_lang="de",
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


@skip_ollama_tests_in_ci
def test_ollama_translator_with_custom_prompt():
    translator = initialize_translator(
        translator_name="ollama",
        model_name="llama3.2:latest",
        source_lang="en",
        target_lang="de",
        prompt_type="romantic",
        temperature=0.7,
    )

    text = "This is a dog."
    translated_text = translator.translate(text)

    assert isinstance(translated_text, str)
    assert translated_text != ""


@skip_ollama_tests_in_ci
def test_ollama_translator_wrong_model_name():
    with pytest.raises(ValueError) as excinfo:
        initialize_translator(
            translator_name="ollama",
            model_name="invalid-model-name",
            target_lang="de",
            source_lang="en",
        )

    assert "Model 'invalid-model-name' is not available." in str(excinfo.value)
    assert "Available models are: " in str(excinfo.value)
    assert (
        "If you haven't installed the model yet, please run `ollama pull model`, but ensure you have Ollama installed and running."
        in str(excinfo.value)
    )
