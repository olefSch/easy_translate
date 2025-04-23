from ollama import Client
from models.base_translator import BaseTranslator

class LLMTranslator(BaseTranslator):
    """
    Translator that drives any Ollama‑hosted model via the Ollama Python client.
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        num_predict: int = 256,
        source_lang: str = "English",
        target_lang: str = "German",
        stop: list[str] | None = None
    ):
        """
        Args:
            model_name: the Ollama model ID (e.g. "llama3-1", "gemma3-4b", etc.)
            num_predict: maximum number of tokens to predict per call
            stop: optional list of stop sequences
        """
        self.client      = Client()
        self.model_name  = model_name
        self.num_predict = num_predict
        self.stop        = stop or ["—"]
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(
        self,
        text: str,
    ) -> str:
        """
        Translate the given text using a local Ollama LLM model.
        """
        prompt = (
            f"Translate the following sentence from {self.source_lang} to {self.target_lang}:\n\n"
            f"{text}\n\n"
            "Return ONLY the translated sentence—no quotes, labels, explanations or extra whitespace."
        )
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": self.num_predict,
                "stop": self.stop,
            }
        )
        return response["response"].strip()
