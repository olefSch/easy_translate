import pytest

from evaluation.translation_evaluator import TranslationEvaluator

@pytest.fixture
def evaluator() -> TranslationEvaluator:
    # Provide a fresh evaluator instance for each test
    return TranslationEvaluator()

