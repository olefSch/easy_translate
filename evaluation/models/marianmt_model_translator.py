from transformers import MarianMTModel, MarianTokenizer
from models.base_translator import BaseTranslator
import torch

class MarianTranslator(BaseTranslator):
    """
    Example translator class for a local MarianMT model.
    """

    def __init__(
        self,
        model_name_or_path: str = "Helsinki-NLP/opus-mt-en-de",
        source_lang: str = "English",
        target_lang: str = "German",
        device: str = "cpu"
    ):
        """
        Load a local MarianMT model from disk or Hugging Face.
        
        Args:
            model_name_or_path (str): Path or name of the MarianMT model 
                                      (e.g., "Helsinki-NLP/opus-mt-en-de").
            source_lang (str): Source language name (not strictly required by Marian,
                               but kept for consistency or referencing).
            target_lang (str): Target language name (same note as above).
            device (str): "cpu" or "cuda".
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device

        # Load tokenizer and model
        self.tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)
        self.model = MarianMTModel.from_pretrained(model_name_or_path)
        self.model.to(device)

    def translate(self, text: str) -> str:
        """
        Translate the given text using the MarianMT model.
        """
        # MarianMT doesn't require a prompt like T5;
        # you simply pass the source text directly.
        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.model.device)

        # Generate translation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=128,       # adjust as needed
                num_beams=4,         # tune as needed
                early_stopping=True
            )

        # Decode the output
        translated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return translated_text
