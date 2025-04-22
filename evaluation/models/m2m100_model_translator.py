from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from models.base_translator import BaseTranslator
import torch

class M2M100Translator(BaseTranslator):
    """
    Example translator class for a local M2M100 model.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/m2m100_418M",
        source_lang: str = "en",
        target_lang: str = "de",
        device: str = "cpu"
    ):
        """
        Load a local M2M100 model from disk or Hugging Face.
        
        Args:
            model_name_or_path (str): Path or name of the M2M100 model 
                                      (e.g., "facebook/m2m100_418M").
            source_lang (str): Source language code (e.g., "en" for English).
            target_lang (str): Target language code (e.g., "de" for German).
            device (str): "cpu" or "cuda".
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device

        # Load tokenizer and model
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name_or_path)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(device)

        # Set the source language for tokenization
        self.tokenizer.src_lang = self.source_lang

    def translate(self, text: str) -> str:
        """
        Translate the given text using the M2M100 model.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        # Force the target language at the beginning of the output
        forced_bos_token_id = self.tokenizer.get_lang_id(self.target_lang)

        # Generate translation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=128,          # adjust as needed
                num_beams=4,            # tune as needed
                early_stopping=True,
                forced_bos_token_id=forced_bos_token_id
            )

        # Decode the output
        translated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return translated_text
