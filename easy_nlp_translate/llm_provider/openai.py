import logging
import os
from typing import Optional, Iterable
from openai import OpenAI
from dotenv import load_dotenv

from ..llm_translator_base import LLMTranslator
from ..config import available_models_openai

logger = logging.getLogger(__name__)

load_dotenv()


class GPTTranslator(LLMTranslator):
    AVAILABLE_MODELS: list[str] = available_models_openai

    """
    A base class for LLM-based translators, inheriting from TranslatorBase.
    """

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        prompt_type: str = "default",
        costum_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initializes the LLMTranslator with a model name, target language, optional source language, and prompt type.

        Args:
            model_name (str): The name of the LLM model to use for translation.
            target_lang (str): The target language code for translation (e.g., 'fr' for French).
            source_lang (Optional[str]): The source language code for translation (e.g., 'en' for English).
                Defaults to None, implying auto-detection will be attempted.
            prompt_type (str): The type of prompt to use for the translation. Defaults to "default".
            costum_prompt (Optional[str]): A custom prompt to use if the prompt type is "custom". Defaults to None.
            temperature (float): The temperature for the model's responses. Defaults to 0.7.
            max_tokens (int): The maximum number of tokens to generate in the response. Defaults to 1000.
        """
        super().__init__(
            model_name,
            target_lang,
            source_lang,
            prompt_type,
            costum_prompt,
            temperature,
            max_tokens,
        )

    def _get_credentials(self) -> str:
        """
        Retries the API Key for the GPT model.
        Returns:
            str: The API key for the GPT model.
        """
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        return key

    def _init_model(self):
        """
        Initializes the GPT model using the provided API key.
        Returns:
            Any: The initialized GPT model.
        """
        client = OpenAI(api_key=self.credentials)
        return client

    def _generate(self, input: str) -> Iterable:
        """
        Generates a translation using the GPT model.
        Args:
            input (str): The input text to be translated.
        Returns:
            Iterable: An iterable containing the generated translation.
        """
        return self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful translator that translates text to {self.target_lang}.",
                },
                {"role": "user", "content": input},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-processes the raw response from the GPT model to extract the translated text.

        Args:
            raw_response (Iterable): The raw response from the GPT model.
        Returns:
            str: The translated text extracted from the raw response.
        """
        return raw_response.choices[0].message.content.strip()
