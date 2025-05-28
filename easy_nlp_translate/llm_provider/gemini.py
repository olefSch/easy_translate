import logging
import os
from typing import Optional, Iterable
from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..llm_translator_base import LLMTranslator
from ..config import available_models_gemini

logger = logging.getLogger(__name__)

load_dotenv()


class GeminiTranslator(LLMTranslator):
    AVAILABLE_MODELS: list[str] = available_models_gemini

    """
    A base class for LLM-based translators, inheriting from TranslatorBase.
    """

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        prompt_type: str = "default",
        temperature: float = 0.7,
    ):
        """
        Initializes the LLMTranslator with a model name, target language, optional source language, and prompt type.

        Args:
            model_name (str): The name of the LLM model to use for translation.
            target_lang (str): The target language code for translation (e.g., 'fr' for French).
            source_lang (Optional[str]): The source language code for translation (e.g., 'en' for English).
                Defaults to None, implying auto-detection will be attempted.
            prompt_type (str): The type of prompt to use for the translation. Defaults to "default".
            temperature (float): The temperature for the model's responses. Defaults to 0.7.
        """
        super().__init__(model_name, target_lang, source_lang, prompt_type, temperature)

    def _get_credentials(self) -> str:
        """
        Retries the API Key for the Gemini model.
        Returns:
            str: The API key for the Gemini model.
        """
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        return key

    def _init_model(self):
        """
        Initializes the Gemini model using the provided API key.
        Returns:
            Any: The initialized Gemini model.
        """
        client = genai.Client(api_key=self.credentials)
        return client

    def _generate(self, input: str) -> Iterable:
        """
        Generates a translation using the Gemini model.
        Args:
            input (str): The input text to be translated.
        Returns:
            Iterable: An iterable containing the generated translation.
        """
        response = self.model.models.generate_content(
            model=self.model_name,
            contents=input,
            config=types.GenerateContentConfig(
                temperature=self.temperature
            ),
        )
        return response

    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-processes the raw response from the Gemini model to extract the translated text.

        Args:
            raw_response (Iterable): The raw response from the Gemini model.
        Returns:
            str: The translated text extracted from the raw response.
        """
        return raw_response.text.strip()
