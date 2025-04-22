from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.base_translator import BaseTranslator
import torch

class NllbTranslator(BaseTranslator):
    """
    Example translator class for a local NLLB (No Language Left Behind) model.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/nllb-200-distilled-600M",
        source_lang: str = "eng_Latn",
        target_lang: str = "deu_Latn",
        device: str = "cpu"
    ):
        """
        Load a local NLLB model from disk or Hugging Face.
        
        Args:
            model_name_or_path (str): Path or name of the NLLB model 
                                      (e.g., "facebook/nllb-200-distilled-600M").
            source_lang (str): Source language code (e.g. "eng_Latn").
            target_lang (str): Target language code (e.g. "deu_Latn").
            device (str): "cpu" or "cuda".
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(device)

        # Set the source language so the tokenizer knows how to handle the input
        self.tokenizer.src_lang = self.source_lang

    def translate(self, text: str) -> str:
        """
        Translate the given text using the NLLB model.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        # Force the model to produce the target language at the beginning
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        # Generate translation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=128,             # adjust as needed
                num_beams=4,               # tune as needed
                early_stopping=True,
                forced_bos_token_id=forced_bos_token_id
            )

        # Decode the output
        translated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return translated_text
