from models import *
from translation_evaluator import TranslationEvaluator
from configs.config import MODEL_REGISTRY
import yaml
from datasets import load_dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    models_to_evaluate = ["mistral", "m2m100", "marian", "t5", "nllb", "mbart50", "llama3.1", "llama3.2", "gemma", "phi3"]

    # Load language mappings
    with open("configs/language_mappings.yaml", 'r') as f:
        language_mappings = yaml.safe_load(f)['language_mappings']

    evaluator = TranslationEvaluator()

    for model_name in models_to_evaluate:
        translator_class = MODEL_REGISTRY.get(model_name)
        if not translator_class:
            logger.warning(f"⚠️ Skipping unknown model: {model_name}")
            continue

        for lang_pair, model_configs in language_mappings.items():
            if model_name not in model_configs:
                continue

            logger.info(f"Evaluating {model_name.upper()} on {lang_pair}")

            source_lang, target_lang = lang_pair.split("-")
            config = model_configs[model_name]
            source_code = config.get("source")
            target_code = config.get("target")

            # Load dataset
            try:
                dataset = load_dataset("wmt19", lang_pair, split="train[:1000]")
                source_sentences = [ex['translation'][source_lang] for ex in dataset]
                target_sentences = [ex['translation'][target_lang] for ex in dataset]
            except Exception as e:
                logger.warning(f"⚠️ Failed to load dataset for {lang_pair}: {e}")
                continue

            # Initialize translator
            try:
                if model_name == "marian":
                    model_args = {
                        "model_name_or_path": f"Helsinki-NLP/opus-mt-{source_code}-{target_code}",
                        "source_lang": source_lang.capitalize(),
                        "target_lang": target_lang.capitalize()
                    }
                else:
                    model_args = {
                        "source_lang": source_code,
                        "target_lang": target_code
                    }

                translator = translator_class(**model_args)
                model_id = f"{model_name}_{lang_pair}"
                evaluator.register_model(model_id, translator)
                evaluator.evaluate(
                    source_sentences, 
                    target_sentences, 
                    model_list=[model_id]
                )

            except Exception as e:
                logger.warning(f"❌ Failed to evaluate {model_name} on {lang_pair}: {e}")
                continue

    # Final report
    evaluator.generate_report()


if __name__ == "__main__":
    main()
