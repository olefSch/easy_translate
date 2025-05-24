from src import initialize_translator


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
