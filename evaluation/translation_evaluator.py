import logging
from typing import List, Dict
import time
import evaluate
from models.base_translator import BaseTranslator
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class TranslationEvaluator:
    """
    TranslationEvaluator orchestrates the evaluation of different
    translation models on a given dataset. It calculates standard
    translation metrics, compares them, and generates summary reports.

    Typical usage:
        1. Instantiate TranslationEvaluator.
        2. Register one or more translation models (local or remote).
        3. Run `evaluate` with input data and reference translations.
        4. Retrieve metrics and generate a comparative report.
    """

    def __init__(self) -> None:
        """
        Initialize the TranslationEvaluator with optional configuration.

        Args:
            None
        """
        self.registered_models = {}
        self.evaluation_results = {}
        self.bleu_metric = evaluate.load("bleu")
        self.meteor_metric = evaluate.load("meteor")

    def register_model(self, model_name: str, model_instance: BaseTranslator) -> None:
        """
        Register a translation model with the evaluator.

        Args:
            model_name (str): Identifier for the model (e.g., "LocalLLM_v1").
            model_instance (BaseTranslator): Instance of a class implementing
                the BaseTranslator interface.
        Returns:
            None.
        """
        if model_name in self.registered_models:
            logger.warning(f"Model '{model_name}' is already registered. Overwriting.")
        self.registered_models[model_name] = model_instance
        logger.info(f"Model '{model_name}' registered successfully.")

    def evaluate(
        self, 
        input_texts: List[str], 
        reference_texts: List[str],
        model_list: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all registered models on the given input and reference texts.

        Args:
            input_texts (List[str]): List of source texts to translate.
            reference_texts (List[str]): List of reference translations 
                corresponding to the input_texts.
            metrics (List[str], optional): List of metric names to use. 
                Defaults to ["bleu", "meteor"].

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with model names 
            as keys and metric results as values, e.g.:
                {
                    "LocalLLM_v1": {"bleu": 0.42, "meteor": 0.60},
                    "LocalLLM_v2": {"bleu": 0.44, "meteor": 0.61}
                }
        """

        if model_list is None:
            model_list = list(self.registered_models.keys())

        logger.info("Starting evaluation...")
        start_time = time.time()

        for model_name in model_list:
            model = self.registered_models[model_name]
            logger.info(f"Evaluating model '{model_name}'...")
            translations = self._generate_translations(model, input_texts)
            results = {}

            formatted_references = [[ref] for ref in reference_texts]

            bleu_result = self.bleu_metric.compute(
                predictions=translations, 
                references=formatted_references
            )
            results["bleu"] = bleu_result["bleu"]

            meteor_result = self.meteor_metric.compute(
                predictions=translations,
                references=formatted_references
            )
            results["meteor"] = meteor_result["meteor"]

            self.evaluation_results[model_name] = results

        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

        return self.evaluation_results
    
    def _generate_translations(self, model: BaseTranslator, input_texts: List[str]) -> List[str]:
        """
        Helper method to generate translations from a model.

        Args:
            model (BaseTranslator): The translation model to use.
            input_texts (List[str]): List of source texts to translate.

        Returns:
            List[str]: Translated texts from the model.
        """
        translations = []
        for text in input_texts:
            translation = model.translate(text)
            translations.append(translation)

        return translations

    def generate_report(self) -> None:
        """
        Print a summary and save the evaluation results to a CSV using pandas.
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results found. Run `evaluate` first.")
            return

        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(self.evaluation_results, orient='index')
        df.index.name = "model"
        df = df.round(4)

        # Print to console
        logger.info("===== Evaluation Report =====")
        for model_name, row in df.iterrows():
            logger.info(f"Model: {model_name}")
            for metric_name, score in row.items():
                logger.info(f"  {metric_name.upper()}: {score:.4f}")
        logger.info("=============================")

        # Save to CSV
        report_path = "translation_evaluation_report.csv"
        try:
            df.to_csv(report_path)
            logger.info(f"✅ Results saved to: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write CSV report: {e}")

