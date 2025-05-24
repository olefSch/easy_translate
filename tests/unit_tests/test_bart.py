import pytest


def test_convert_language_code(get_mbart):
    """
    Test the convert_language_code function.
    """

    translator = get_mbart(
        target_lang="en",
        source_lang="de",
    )

    assert translator._convert_lang_code("en") == "en_XX"


def test_convert_language_code_invalid(get_mbart):
    """
    Test the convert_language_code function with an invalid language code.
    """

    translator = get_mbart(
        target_lang="en",
        source_lang="de",
    )

    with pytest.raises(ValueError) as excinfo:
        translator._convert_lang_code("invalid_lang")

    assert "Language code 'invalid_lang' is not supported" in str(
        excinfo.value
    )


def test_init_tokenizer(get_mbart):
    """
    Test the _init_tokenizer method.
    """

    translator = get_mbart(
        target_lang="en",
        source_lang="de",
    )
    tokenizer = translator._init_tokenizer()
    assert tokenizer.src_lang == "de_DE"


def test_init_tokenizer_no_source_lang(get_mbart):
    """
    Test the _init_tokenizer method without a source language.
    """

    translator = get_mbart(
        target_lang="en",
    )
    tokenizer = translator._init_tokenizer()
    assert tokenizer.src_lang is not None
    assert tokenizer.src_lang != "de_DE"


def test_translate(get_mbart):
    """
    Test the translate method.
    """

    translator = get_mbart(
        target_lang="en",
        source_lang="de",
    )

    text_to_translate = "Das ist ein Hund."
    translated_text = translator.translate(text_to_translate)

    assert translated_text == "This is a dog."


def test_translate_no_source_lang(get_mbart):
    """
    Test the translate method without a source language.
    """

    translator = get_mbart(
        target_lang="en",
    )

    text_to_translate = "Das ist ein Hund."
    translated_text = translator.translate(text_to_translate)

    assert translated_text == "This is a dog."
