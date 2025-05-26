import pytest
from easy_nlp_translate.initialize import initialize_translator


# --- mbart translator
def test_initialize_translator():
    """
    Test the initialize_translator function.
    """

    # Test with a valid translator type
    translator = initialize_translator(
        "mbart", target_lang="en", source_lang="de"
    )
    assert translator is not None
    assert translator.target_lang == "en"
    assert translator.source_lang == "de"


def test_initialize_translator_invalid_type():
    """
    Test the initialize_translator function with an invalid translator type.
    """
    with pytest.raises(ValueError):
        initialize_translator(
            "invalid_type", target_lang="en", source_lang="de"
        )


def test_initialize_translator_missing_source_lang():
    """
    Test the initialize_translator function with a missing source language.
    """
    translator = initialize_translator("mbart", target_lang="en")
    assert translator is not None
    assert translator.target_lang == "en"
    assert translator.source_lang is None
