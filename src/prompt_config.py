from enum import Enum as PyEnum
from pathlib import Path
from typing import List

_PROMPT_CONFIG_MODULE_DIR = Path(__file__).resolve().parent
SHARED_LLM_PROMPT_TEMPLATES_DIR = _PROMPT_CONFIG_MODULE_DIR / "prompts"


class PromptStyle(PyEnum):
    DEFAULT = ("default", "Basic Translation", "default_translation.jinja")
    FORMAL = ("formal", "Formal Translation", "formal_translation.jinja")
    SUMMARIZE = (
        "translate_and_summarize",
        "Summarize and Translate",
        "translate_and_summarize.jinja",
    )
    FORMAL_SUMMARIZE = (
        "formal_translate_and_summarize",
        "Formal Summarize and Translate",
        "formal_translate_and_summarize.jinja",
    )
    ROMANTIC = (
        "romantic",
        "Romantic Translation",
        "romantic_translation.jinja",
    )
    POETIC = ("poetic", "Poetic Translation", "poetic_translation.jinja")

    def __init__(self, code: str, description: str, template_filename: str):
        """
        Initialize the PromptStyle enum with a code, description, and template filename.

        Args:
            code (str): The code representing the prompt style.
            description (str): A description of the prompt style.
            template_filename (str): The filename of the template associated with the prompt style.
        """
        self._value_ = code
        self.description: str = description
        self.template_filename: str = template_filename

    def __new__(cls, code: str, description: str, template_filename: str):
        obj = object.__new__(cls)
        obj._value_ = code
        obj.description = description
        obj.template_filename = template_filename
        return obj

    @classmethod
    def from_code(cls, code: str) -> "PromptStyle":
        """
        Returns the PromptStyle enum member corresponding to the given code."
        Args:
            code (str): The code representing the prompt style.
        Returns:
            PromptStyle: The corresponding PromptStyle enum member.
        Raises:
            ValueError: If the code does not match any PromptStyle member.
        """
        code_lower = code.lower()
        for member in cls:
            if member.value.lower() == code_lower:
                return member
        raise ValueError(
            f" Unallowed prompt style code '{code}'. Allowed codes are: {PromptStyle.get_available_codes()}"
        )

    @classmethod
    def get_available_codes(cls) -> List[str]:
        """
        Returns a list of all available prompt style codes.
        """
        return [member.value for member in cls]
