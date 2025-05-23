import pytest

from src.prompt_config import PromptStyle


def test_from_code_valid_codes():
    """
    Test that from_code returns the correct enum member for valid codes (case-insensitive).
    """
    assert PromptStyle.from_code("default") == PromptStyle.DEFAULT
    assert PromptStyle.from_code("DEFAULT") == PromptStyle.DEFAULT
    assert PromptStyle.from_code("ForMal") == PromptStyle.FORMAL
    assert (
        PromptStyle.from_code("translate_and_summarize")
        == PromptStyle.SUMMARIZE
    )
    assert (
        PromptStyle.from_code("formal_translate_and_summarize")
        == PromptStyle.FORMAL_SUMMARIZE
    )
    assert PromptStyle.from_code("romantic") == PromptStyle.ROMANTIC
    assert PromptStyle.from_code("POETIC") == PromptStyle.POETIC


def test_from_code_invalid_code():
    """
    Test that from_code raises a ValueError for an invalid code.
    """
    invalid_code = "non_existent_style"
    with pytest.raises(ValueError) as excinfo:
        PromptStyle.from_code(invalid_code)

    expected_error_message_part = (
        f"Unallowed prompt style code '{invalid_code}'."
    )
    assert expected_error_message_part in str(excinfo.value)


def test_get_available_codes(expected_codes):
    """
    Test that get_available_codes returns a list of all defined codes.
    """
    available_codes = PromptStyle.get_available_codes()

    assert isinstance(available_codes, list)
    assert len(available_codes) == len(expected_codes)
    assert set(available_codes) == set(expected_codes)


def test_enum_member_attributes():
    """
    Test that enum members have the correct custom attributes.
    """
    default_style = PromptStyle.DEFAULT
    assert default_style.value == "default"
    assert default_style.description == "Basic Translation"
    assert default_style.template_filename == "default_translation.jinja"

    poetic_style = PromptStyle.POETIC
    assert poetic_style.value == "poetic"
    assert poetic_style.description == "Poetic Translation"
    assert poetic_style.template_filename == "poetic_translation.jinja"
