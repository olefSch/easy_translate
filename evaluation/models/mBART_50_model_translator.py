from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from models.base_translator import BaseTranslator
import torch

class MBartTranslator(BaseTranslator):
    """
    Example translator class for a local mBART-50 model.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/mbart-large-50-many-to-many-mmt",
        source_lang: str = "en_XX",
        target_lang: str = "de_DE",
        device: str = "cpu"
    ):
        """
        Load a local mBART-50 model from disk or Hugging Face.
        
        Args:
            model_name_or_path (str): Path or name of the mBART-50 model
                                      (e.g., "facebook/mbart-large-50-many-to-many-mmt").
            source_lang (str): Source language code (e.g. "en_XX").
            target_lang (str): Target language code (e.g. "de_DE").
            device (str): "cpu" or "cuda".
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device

        # Load tokenizer and model
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(device)

        # Set the source language so the tokenizer knows how to handle the input
        self.tokenizer.src_lang = self.source_lang

    def translate(self, text: str) -> str:
        """
        Translate the given text using the mBART-50 model.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        # Force the model to produce text in the target language
        forced_bos_token_id = self.tokenizer.lang_code_to_id[self.target_lang]

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
