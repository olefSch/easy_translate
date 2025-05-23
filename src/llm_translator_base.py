import logging

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Iterable
from jinja2 import Template

from .translator_base import TranslatorBase
from .prompt_config import PromptStyle, SHARED_LLM_PROMPT_TEMPLATES_DIR
from .config import language_code_to_name_map


logger = logging.getLogger(__name__)


class LLMTranslator(TranslatorBase):
    AVAILABLE_MODELS: list[str] = NotImplemented
    TEMPLATES_DIR: Path = SHARED_LLM_PROMPT_TEMPLATES_DIR
    LANGUAGE_MAPPER: dict[str, str] = language_code_to_name_map

    """
    A base class for LLM-based translators, inheriting from TranslatorBase.
    """

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        prompt_type: str = "default",
    ):
        """
        Initializes the LLMTranslator with a model name, target language, optional source language, and prompt type.
        Args:
            model_name (str): The name of the LLM model to use for translation.
            target_lang (str): The target language code for translation (e.g., 'fr' for French).
            source_lang (Optional[str]): The source language code for translation (e.g., 'en' for English).
                Defaults to None, implying auto-detection will be attempted.
        """
        super().__init__(target_lang, source_lang)

        self._validate_model_name(model_name)
        self.model_name = model_name
        self.credentials = self._get_credentials()
        self.model = self._init_model(model_name)

        try:
            self.prompt_style: PromptStyle = PromptStyle.from_code(prompt_type)
        except ValueError:
            raise ValueError(
                f"Prompt type '{prompt_type}' is not available. Avaliable types are: {PromptStyle.get_available_codes()}"
            )

        self.prompt: Template = self._init_prompt()

    def _get_prompt_template(self, prompt_path: Path) -> Template:
        """
        Loads the prompt template from the specified file path.
        Args:
            prompt_path (Path): The path to the prompt template file.
        Returns:
            Template: A Jinja2 Template object initialized with the content of the prompt file.
        """
        with open(prompt_path, "r") as file:
            prompt_template = file.read()
        return Template(prompt_template)

    def _init_prompt(self) -> Template:
        """
        Initializes the prompt template based on the selected prompt style.
        Returns:
            Template: A Jinja2 Template object initialized with the content of the prompt file.
        """
        prompt_path: Path = (
            self.TEMPLATES_DIR / self.prompt_style.template_filename
        )

        logger.info(f"Loading prompt template from {prompt_path}")

        return self._get_prompt_template(prompt_path)

    def _validate_model_name(self, model_name: str):
        """
        Validates if the given model name is available.
        Args:
            model_name (str): The name of the model to validate.
        Raises:
            ValueError: If the model name is not found in `self.AVAILABLE_MODELS`.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not available. Available models are: {self.AVAILABLE_MODELS}"
            )

    def _render_prompt(self, text_to_translate: str) -> str:
        """
        Renders the prompt template with the provided text and language information.
        Args:
            text_to_translate (str): The text to be translated.
        Returns:
            str: The rendered prompt string.
        Raises:
            ValueError: If the source language cannot be detected and is not provided.
        """
        if self.source_lang is None:
            source_lang = self.detect_language(text_to_translate)
            if source_lang is None:
                raise ValueError(
                    "Source language could not be detected. Please provide a valid source language."
                )
        else:
            source_lang = self.source_lang

        long_source_lang = self.LANGUAGE_MAPPER.get(source_lang)
        long_target_lang = self.LANGUAGE_MAPPER.get(self.target_lang)

        logger.info(
            f"Detected source language: {long_source_lang}, target language: {long_target_lang}"
        )

        return self.prompt.render(
            source_language=long_source_lang,
            target_language=long_target_lang,
            text_to_translate=text_to_translate,
        )

    @abstractmethod
    def _get_credentials(self):
        """
        Retrieves the credentials needed to access the model.
        This method should be implemented in subclasses to provide the necessary credentials.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )

    @abstractmethod
    def _init_model(self, model_name: str):
        """
        Initializes the model with the given name.
        This method should be implemented in subclasses to provide the necessary model initialization.
        Args:
            model_name (str): The name of the model to initialize.
        Returns:
            The initialized model object.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )

    @abstractmethod
    def _generate(self, input: str) -> Iterable:
        """
        Generates a response from the model based on the input prompt.
        This method should be implemented in subclasses to provide the necessary generation logic.
        Args:
            input (str): The input prompt to generate a response for.
        Returns:
            Iterable: An iterable containing the generated response.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )

    @abstractmethod
    def _post_process(self, raw_response: Iterable) -> str:
        """
        Post-processes the raw response from the model to extract the final translated text.
        This method should be implemented in subclasses to provide the necessary post-processing logic.
        Args:
            raw_response (Iterable): The raw response from the model.
        Returns:
            str: The final translated text.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )

    def translate(self, text: str) -> str:
        """
        Translates the given text using the configured LLM.

        Orchestrates the translation process by:
        1. Validating the input text.
        2. Rendering the appropriate prompt (including language detection if needed).
        3. Sending the prompt to the LLM via `_generate`.
        4. Post-processing the LLM's response via `_post_process`.

        Args:
            text: The text to be translated.

        Returns:
            The translated text.

        Raises:
            ValueError: If input text validation fails, language detection fails
                when required, or prompt rendering fails.
            NotImplementedError: If any of the abstract methods are not implemented.
            Any exceptions from underlying LLM API calls.
        """
        LLMTranslator._validate_basic_text_to_translate(text)

        rendered_prompt = self._render_prompt(text_to_translate=text)

        logger.debug(
            f"LLM '{self.model_name}' ({self.__class__.__name__}) - "
            f"Style '{self.prompt_style.name}' - Final Prompt: {rendered_prompt}"
        )

        raw_llm_output = self._generate(rendered_prompt)
        translated_text = self._post_process(raw_llm_output)
        logger.debug(
            f"LLM '{self.model_name}' - Post-processed translation: {translated_text}"
        )

        return translated_text
