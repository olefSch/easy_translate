import logging

from abc import abstractmethod
from .translator_base import TranslatorBase

from typing import Optional, Iterable
from jinja2 import Template


logger = logging.getLogger(__name__)


class LLMTranslator(TranslatorBase):

    AVAILABLE_MODELS: list[str] = NotImplemented

    """
    A base class for LLM-based translators, inheriting from TranslatorBase.
    """

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        prompt_type: Optional[str] = "default",
    ):
        """
        Initialize the LLM translator with optional source and required target languages.

        Args:
            model_name (str): The name of the LLM model to be used.
            target_lang (str): The target language code (e.g., 'fr' for French).
            source_lang (Optional[str], optional): The source language code (e.g., 'en' for English). 
                                                   Defaults to None (implying auto-detection).
        """
        super().__init__(target_lang, source_lang)
 
        self._validate_model_name(model_name)
        self.model_name = model_name
        self.model = self._init_model(model_name)
        self.prompt: Template = self._init_prompt(prompt_type)

    def _get_prompt_template(self, prompt_path: str) -> Template:
        """
        Get the prompt template from the specified path.

        Args:
            prompt_path (str): The path to the prompt template file.

        Returns:
            str: The prompt template.
        """
        with open(prompt_path, "r") as file:
            prompt_template = file.read()
        return Template(prompt_template)


    def _init_prompt(self, prompt_type: str) -> Template:
        """
        Initialize the prompt for the LLM model baseed on the prompt type enum.

        Args:
            prompt_type (str): The type of prompt to be used.

        Returns:
            str: The initialized prompt.
        """
        return self._get_prompt_template(prompt_type)

    def _validate_model_name(self, model_name: str):
        """
        Validate the model name.

        Args:
            model_name (str): The name of the LLM model to be used.

        Raises:
            ValueError: If the model name is not in the list of available models.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' is not available. Available models are: {self.AVAILABLE_MODELS}")
    
    @abstractmethod
    def _get_credentials(self):
        """
        Get the credentials for the LLM model.

        Returns:
            dict: The credentials for the LLM model.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _init_model(self, model_name: str):
        """
        Initialize the LLM model.

        Args:
            model_name (str): The name of the LLM model to be used.

        Returns:
            Any: The initialized LLM model.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        Args:
            text (str): The text whose language is to be detected.

        Returns:
            str: The detected language code.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _generate(self, input: str) -> Iterable:
        """
        Generate a translation for the given input.

        Args:
            input (str): The input text to be translated.

        Returns:
            str: The generated model output
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-process the generated output.

        Args:
            output (str): The generated output to be post-processed.

        Returns:
            str: The post-processed output.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
