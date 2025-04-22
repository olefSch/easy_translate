from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.base_translator import BaseTranslator
import torch

class T5Translator(BaseTranslator):
    """
    Example translator class for a local T5 model.
    """

    def __init__(
        self,
        model_name_or_path: str = "t5-small",
        source_lang: str = "English",
        target_lang: str = "German",
        device: str = "cpu"
    ):
        """
        Load a local T5 model from disk or Hugging Face.
        
        Args:
            model_name_or_path (str): Path or name of the model (e.g., "t5-small").
            source_lang (str): Source language name (used in the prompt).
            target_lang (str): Target language name (used in the prompt).
            device (str): "cpu" or "cuda".
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(device)

    def translate(self, text: str) -> str:
        """
        Translate the given text using the T5 model.
        """
        # T5 often expects a task-specific prompt:
        prompt = f"translate {self.source_lang} to {self.target_lang}: {text}"

        # Encode input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate translation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_length=128,        # adjust as needed
                num_beams=4,          # or configure for your needs
                early_stopping=True
            )

        # Decode the output
        translated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return translated_text
