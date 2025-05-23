import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from evaluation.translation_evaluator import TranslationEvaluator


@pytest.fixture
def evaluator() -> TranslationEvaluator:
    # Provide a fresh evaluator instance for each test
    return TranslationEvaluator()


