import logging
import os
from typing import Iterable, Optional
import anthropic
from anthropic.types import Message
from dotenv import load_dotenv

from ..llm_translator_base import LLMTranslator
from ..config import available_models_claude

logger = logging.getLogger(__name__)

load_dotenv()


class ClaudeTranslator(LLMTranslator):
    AVAILABLE_MODELS: list[str] = available_models_claude

    """
    A class for LLM-based translations using Anthropic's Claude models,
    inheriting from LLMTranslator.
    """

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        prompt_type: str = "default",
        custom_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initializes the ClaudeTranslator with a model name, target language,
        optional source language, and prompt type.

        Args:
            model_name (str): The name of the Claude model to use for translation.
            target_lang (str): The target language code for translation (e.g., 'fr' for French).
            source_lang (Optional[str]): The source language code for translation (e.g., 'en' for English).
            prompt_type (str): The type of prompt to use for the translation. Defaults to "default".
            custom_prompt (Optional[str]): A custom prompt to use if the prompt type is "custom". Defaults to None.
            temperature (float): The temperature for the model's responses. Defaults to 0.7.
            max_tokens (int): The maximum number of tokens to generate in the response. Defaults to 1000.
        """
        super().__init__(
            model_name,
            target_lang,
            source_lang,
            prompt_type,
            custom_prompt,
            temperature,
            max_tokens,
        )

    def _get_credentials(self) -> str:
        """
        Retrieves the API Key for the Anthropic Claude model.

        Returns:
            str: The API key for the Claude model.

        Raises:
            ValueError: If the ANTHROPIC_API_KEY environment variable is not set.
        """
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        return key

    def _init_model(self):
        """
        Initializes the Anthropic Claude client using the API key stored in self.credentials.
        Assumes self.credentials is already populated by the base class calling _get_credentials.

        Returns:
            anthropic.Anthropic: The initialized Anthropic client.
        """
        client = anthropic.Anthropic(api_key=self.credentials)
        return client

    def _generate(self, input: str) -> Iterable[Message]:
        """
        Generates a translation using the Claude model.
        Assumes self.model (the client) is initialized by the base class.

        Args:
            input_prompt (str): The fully formatted input prompt to be sent to the model.

        Returns:
            anthropic.types.Message: The raw Message object from the Claude model.
        """
        try:
            response = self.model.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": input}],
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate content with Claude model '{self.model_name}': {e}"
            )
        return response

    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-processes the raw response from the Claude model to extract the translated text.
        Directly accesses expected attributes; will raise AttributeError/IndexError if response
        structure is unexpected.

        Args:
            raw_response (anthropic.types.Message): The raw Message object from the Claude model.

        Returns:
            str: The translated text extracted from the raw response.
        """
        return raw_response.content[0].text.strip()
