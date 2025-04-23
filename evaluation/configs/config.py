from models import * 
from functools import partial

MODEL_REGISTRY = {
    "nllb":    NllbTranslator,
    "m2m100":  M2M100Translator,
    "mbart50": MBartTranslator,
    "marian":  MarianTranslator,
    "t5":      T5Translator,
    "llama3.2" : partial(LLMTranslator, model_name="llama3.2:3b"),
    "llama3.1" : partial(LLMTranslator, model_name="llama3.1:8b"),
    "gemma"  : partial(LLMTranslator, model_name="gemma3:4b"),
    "phi3"   : partial(LLMTranslator, model_name="phi3:3.8b"),
    "mistral"   : partial(LLMTranslator, model_name="mistral:7b"),
}
