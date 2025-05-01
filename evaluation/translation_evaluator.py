import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import evaluate
import pandas as pd

from evaluation.models.base_translator import BaseTranslator

# Configure module‐level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)


class TranslationEvaluator:
    """Evaluate translation models against reference translations.

    This class orchestrates evaluation of multiple translation models
    computing metrics like BLEU and METEOR on provided datasets.
    Results can be retrieved programmatically or saved as reports.

    Typical usage:
        evaluator = TranslationEvaluator()
        evaluator.register_model('model1', translator1)
        results = evaluator.evaluate(
            inputs, references, model_names=['model1'])
        evaluator.generate_report('results.csv', models=['model1'])
    """

    def __init__(self) -> None:
        """Initialize the evaluator with default metrics."""

        self._registered_models: Dict[str, BaseTranslator] = {}
        self._results: Dict[str, Dict[str, float]] = {}
        self._bleu = evaluate.load("bleu")
        self._meteor = evaluate.load("meteor")

    def register_model(self, name: str, model: BaseTranslator) -> None:
        """Register a translation model for evaluation.

        Args:
            name (str): Unique identifier for the model.
            model (BaseTranslator): Instance implementing BaseTranslator interface.

        Raises:
            ValueError: If name is empty or model is None.
        """
        if not name:
            raise ValueError("Model name must not be empty.")
        if model is None:
            raise ValueError("Model instance must not be None.")
        if name in self._registered_models:
            logger.warning("Overwriting existing model '%s'.", name)
        self._registered_models[name] = model
        logger.info("Registered model '%s'.", name)

    def evaluate(
        self,
        inputs: List[str],
        references: List[str],
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate registered models on inputs against reference translations.

        Args:
            inputs (List[str]): List of source texts to translate.
            references (List[str]): List of corresponding reference translations.
            model_names (Optional[List[str]]): Subset of registered model names to evaluate.
                If None, evaluates all registered models.

        Returns:
            Dict[str, Dict[str, float]]: Mapping from model name to a dict of metric scores.

        Raises:
            ValueError: If inputs and references lengths differ.
            KeyError: If a specified model or metric is not registered.
        """

        if len(inputs) != len(references):
            raise ValueError(
                f"Inputs length {len(inputs)} does not match "
                f"references length {len(references)}."
            )

        model_names = model_names or list(self._registered_models)
        for name in model_names:
            if name not in self._registered_models:
                raise KeyError(f"Model '{name}' not registered.")

        logger.info(
            "Starting evaluation of %d models on %d samples",
            len(model_names),
            len(inputs),
        )
        start_time = time.time()

        formatted_refs = [[r] for r in references]

        for name in model_names:
            model = self._registered_models[name]
            translations = self._batch_translate(model, inputs)

            bleu_res = self._bleu.compute(
                predictions=translations, references=formatted_refs
            )["bleu"]
            meteor_res = self._meteor.compute(
                predictions=translations, references=formatted_refs
            )["meteor"]

            self._results[name] = {
                "bleu": bleu_res,
                "meteor": meteor_res,
            }

        elapsed = time.time() - start_time
        logger.info("Completed evaluation in %.2f seconds", elapsed)
        return self._results

    def generate_report(
        self,
        file_path: Union[str, Path],
        models: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Save and print evaluation report for BLEU & METEOR.

        Args:
            file_path (Union[str, Path]): Destination CSV file path.
            models (Optional[Union[str, List[str]]]): Single model name,
                list of names, or None for all.

        Returns:
            None.

        Raises:
            ValueError: If no evaluation results are available.
        """

        if not self._results:
            raise ValueError("No evaluation results to report. Run evaluate() first.")

        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(self._results, orient="index")
        df.index.name = "model"
        df = df.round(4)

        # filter if requested
        if models is not None:
            if isinstance(models, str):
                models = [models]
            missing = [m for m in models if m not in df.index]
            if missing:
                logger.warning(f"⚠️ Requested model(s) not found: {missing}")
            df = df.loc[[m for m in models if m in df.index]]

        # Print to console
        logger.info("===== Evaluation Report =====\n%s", df.to_string())

        # Save to CSV
        try:
            path = Path(file_path)
            df.to_csv(path)
            logger.info(f"✅ Results saved to: {path}")
        except Exception as e:
            logger.error(f"❌ Failed to write CSV report: {e}")

    @staticmethod
    def _batch_translate(model: BaseTranslator, texts: List[str]) -> List[str]:
        """Translate a batch of texts using the given model.

        Args:
            model (BaseTranslator): Translator instance.
            texts (List[str]): List of source texts.

        Returns:
            List[str]: A list of translated text strings.
        """
        return [model.translate(t) for t in texts]
