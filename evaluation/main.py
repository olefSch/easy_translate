import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import load_dataset

from configs.config import (DATASET_NAME, DATASET_SPLIT, LANGUAGE_MAPPING_PATH,
                            MODEL_REGISTRY, MODELS_TO_EVALUATE, OUTPUT_DIR)
from translation_evaluator import TranslationEvaluator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def load_language_mappings(path: Path) -> Dict[str, Any]:
    """Load the `language_mappings` section from a YAML file.

    Args:
        path (Path): Path to the YAML file containing the
            top-level `language_mappings` key.

    Returns:
        Dict[str, Any]: The dictionary under `language_mappings`.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        KeyError: If the loaded YAML does not contain `language_mappings`.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mappings not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "language_mappings" not in cfg:
        raise KeyError(f"'language_mappings' not in {path}")
    return cfg["language_mappings"]


def evaluate_models(
    evaluator: TranslationEvaluator,
    models: List[str],
    mappings: Dict[str, Any],
    dataset_name: str,
    split: str,
    output_dir: Path,
) -> None:
    """Evaluate translation models on specified language pairs and save reports.

    Args:
        evaluator (TranslationEvaluator): The evaluation orchestrator instance.
        models (List[str]): Keys of translators to evaluate (must exist in MODEL_REGISTRY).
        mappings (Dict[str, Any]): A mapping from language‐pair strings
            (e.g. "en-de") to per‐model configuration dicts.
        dataset_name (str): Name of the HuggingFace dataset (e.g. "wmt19").
        split (str): Dataset split specifier (e.g. "train[:1]").
        output_dir (Path): Directory in which to write `<model>_<lang_pair>_report.csv`.

    Returns:
        None.

    Raises:
        FileNotFoundError: If a language mapping file is missing (handled upstream).
        KeyError: If a model key is not found in the registry (logged and skipped).
        Exception: Any errors during dataset loading, translator instantiation,
            evaluation, or report generation are caught and logged per‐case.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        translator_cls = MODEL_REGISTRY.get(model_name)
        if translator_cls is None:
            logger.warning("Skipping unknown model '%s'.", model_name)
            continue

        for lang_pair, cfgs in mappings.items():
            if model_name not in cfgs:
                continue

            logger.info("Evaluating %s on %s", model_name, lang_pair)
            src_lang, tgt_lang = lang_pair.split("-", 1)
            src_code, tgt_code = (
                cfgs[model_name].get("source"),
                cfgs[model_name].get("target"),
            )

            if not (src_code and tgt_code):
                logger.error(
                    "Missing source/target in mapping for %s:%s—skipping.",
                    model_name,
                    lang_pair,
                )
                continue

            # Load dataset
            try:
                ds = load_dataset(dataset_name, lang_pair, split=split)
                inputs = [ex["translation"][src_lang] for ex in ds]
                refs = [ex["translation"][tgt_lang] for ex in ds]
            except Exception:
                logger.exception(
                    "Failed to load %s[%s]; skipping.", dataset_name, split
                )
                continue

            # Instantiate translator
            try:
                init_args = {"source_lang": src_code, "target_lang": tgt_code}
                if model_name == "marian":
                    init_args["model_name_or_path"] = (
                        f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
                    )
                    init_args["source_lang"] = src_lang.capitalize()
                    init_args["target_lang"] = tgt_lang.capitalize()

                translator = translator_cls(**init_args)
            except Exception:
                logger.exception("Failed to init translator %s; skipping.", model_name)
                continue

            # Register, evaluate, report
            model_id = f"{model_name}_{lang_pair}"
            evaluator.register_model(model_id, translator)

            try:
                evaluator.evaluate(inputs, refs, model_names=[model_id])
            except Exception:
                logger.exception("Evaluation failed for %s; continuing.", model_id)
                continue

            report_file = output_dir / f"{model_id}_report.csv"
            try:
                evaluator.generate_report(report_file, models=model_id)
            except Exception:
                logger.exception("Could not write report for %s.", model_id)


def main() -> None:
    """Orchestrate loading configs, running evaluations, and saving reports."""
    try:
        mappings = load_language_mappings(LANGUAGE_MAPPING_PATH)
    except Exception as e:
        logger.error("Aborting: %s", e)
        return

    evaluator = TranslationEvaluator()
    evaluate_models(
        evaluator=evaluator,
        models=MODELS_TO_EVALUATE,
        mappings=mappings,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
