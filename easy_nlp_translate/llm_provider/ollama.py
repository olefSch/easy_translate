import logging
from typing import Optional, Iterable
import ollama
from dotenv import load_dotenv

from ..llm_translator_base import LLMTranslator

logger = logging.getLogger(__name__)

load_dotenv()


class OllamaTranslator(LLMTranslator):
    AVAILABLE_MODELS: list[str] = [
        model_obj.model for model_obj in ollama.list()["models"]
    ]

    """
    A class for LLM-based translations using local Ollama models,
    inheriting from LLMTranslator.
    """

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        prompt_type: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initializes the OllamaTranslator.

        Args:
            model_name (str): The name of the Ollama model to use (e.g., 'llama3', 'mistral').
            target_lang (str): The target language code (e.g., 'fr' for French).
            source_lang (Optional[str]): The source language code (e.g., 'en' for English).
            prompt_type (str): The type of prompt to use. Defaults to "default".
            temperature (float): The temperature for model responses. Defaults to 0.7.
            max_tokens (int): Max tokens for the response (maps to 'num_predict'). Defaults to 1000.
        """
        super().__init__(
            model_name,
            target_lang,
            source_lang,
            prompt_type,
            temperature,
            max_tokens,
        )

    def _validate_model_name(self, model_name: str) -> None:
        """
        Validates the model name against the available models.

        Args:
            model_name (str): The name of the model to validate.

        Raises:
            ValueError: If the model name is not in the list of available models.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not available. Available models are: {self.AVAILABLE_MODELS}. If you haven't installed the model yet, please run `ollama pull model`, but ensure you have Ollama installed and running."
            )

    def _get_credentials(self) -> None:
        """
        No explicit credentials are needed for Ollama models.
        Ollama models are typically run locally and do not require an API key.

        Returns:
            None
        """
        return None

    def _init_model(self):
        """
        No explicit initialization is needed for Ollama models.
        The model is used directly via the ollama library.

        Returns:
            None
        """
        return None

    def _generate(self, input: str) -> Iterable:
        """
        Generates a translation using the Ollama model.

        Args:
            input_prompt (str): The fully formatted input prompt.

        Returns:
            dict: The raw response dictionary from the Ollama model.
        """
        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        response = ollama.generate(
            model=self.model_name,
            prompt=input,
            options=options,
        )
        return response

    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-processes the raw response from Ollama to extract the translated text.

        Args:
            raw_response (Iterable): The raw response dictionary from Ollama.

        Returns:
            str: The translated text.
        """
        return raw_response.get("response", "").strip()
