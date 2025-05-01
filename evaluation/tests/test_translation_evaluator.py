import logging
from pathlib import Path

import pandas as pd
import pytest

import evaluation.translation_evaluator as te_mod
from evaluation.models.base_translator import BaseTranslator
from evaluation.translation_evaluator import TranslationEvaluator

te_mod.logger.propagate = True


class DummyTranslator(BaseTranslator):
    """A translator that returns the source text unchanged."""

    def translate(self, text: str) -> str:
        return text


@pytest.fixture
def evaluator() -> TranslationEvaluator:
    return TranslationEvaluator()


def test_scores_and_report(tmp_path: Path, evaluator: TranslationEvaluator):
    # Arrange
    inputs = ["hello world", "goodbye"]
    references = ["hello world", "goodbye"]
    evaluator.register_model("dummy", DummyTranslator())

    # Act
    results = evaluator.evaluate(inputs, references)

    # Assert evaluate() results
    assert "dummy" in results
    bleu_score = results["dummy"]["bleu"]
    meteor_score = results["dummy"]["meteor"]
    for score in (bleu_score, meteor_score):
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # Write and verify report
    report = tmp_path / "dummy_report.csv"
    evaluator.generate_report(report, models="dummy")
    assert report.exists()
    df = pd.read_csv(report, index_col="model")
    expected_bleu = round(bleu_score, 4)
    expected_meteor = round(meteor_score, 4)
    assert pytest.approx(expected_bleu, rel=1e-6) == df.loc["dummy", "bleu"]
    assert pytest.approx(expected_meteor, rel=1e-6) == df.loc["dummy", "meteor"]


def test_evaluate_length_mismatch(evaluator: TranslationEvaluator):
    evaluator.register_model("dummy", DummyTranslator())
    with pytest.raises(ValueError) as excinfo:
        evaluator.evaluate(["one"], ["one", "two"])
    assert "Inputs length" in str(excinfo.value)


def test_evaluate_unknown_model(evaluator: TranslationEvaluator):
    with pytest.raises(KeyError):
        evaluator.evaluate(["x"], ["x"], model_names=["not_registered"])


def test_generate_report_without_evaluate(
    tmp_path: Path, evaluator: TranslationEvaluator
):
    with pytest.raises(ValueError):
        evaluator.generate_report(tmp_path / "no_results.csv")


def test_register_model_validation(evaluator: TranslationEvaluator):
    with pytest.raises(ValueError):
        evaluator.register_model("", DummyTranslator())
    with pytest.raises(ValueError):
        evaluator.register_model("dummy", None)


def test_duplicate_registration_logs_warning(caplog, evaluator: TranslationEvaluator):
    caplog.set_level(logging.WARNING)
    evaluator.register_model("dup", DummyTranslator())
    evaluator.register_model("dup", DummyTranslator())
    assert "Overwriting existing model 'dup'." in caplog.text


def test_multiple_models_evaluation_and_report(
    tmp_path: Path, evaluator: TranslationEvaluator
):
    evaluator.register_model("m1", DummyTranslator())
    evaluator.register_model("m2", DummyTranslator())
    inputs = ["a", "b"]
    references = ["a", "b"]

    results = evaluator.evaluate(inputs, references)
    assert set(results.keys()) == {"m1", "m2"}

    report = tmp_path / "all_models_report.csv"
    evaluator.generate_report(report)
    df = pd.read_csv(report, index_col="model")
    assert set(df.index) == {"m1", "m2"}


def test_filter_report_by_model(tmp_path: Path, evaluator: TranslationEvaluator):
    evaluator.register_model("m1", DummyTranslator())
    evaluator.register_model("m2", DummyTranslator())
    evaluator.evaluate(["x"], ["x"])

    report = tmp_path / "filtered_report.csv"
    evaluator.generate_report(report, models=["m2"])
    df = pd.read_csv(report, index_col="model")
    assert list(df.index) == ["m2"]


def test_report_write_failure(
    tmp_path: Path, evaluator: TranslationEvaluator, monkeypatch, caplog
):
    evaluator.register_model("dummy", DummyTranslator())
    evaluator.evaluate(["a"], ["a"])

    # Simulate disk failure by having to_csv raise
    def fake_to_csv(self, path, *args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    caplog.set_level(logging.ERROR)
    evaluator.generate_report(tmp_path / "fail.csv")
    assert "Failed to write CSV report" in caplog.text


def test_re_evaluation_updates_results(tmp_path: Path, evaluator: TranslationEvaluator):
    evaluator.register_model("dummy", DummyTranslator())

    # First run with perfect match
    evaluator.evaluate(["a"], ["a"])
    first_meteor = evaluator._results["dummy"]["meteor"]

    # Second run with mismatch
    evaluator.evaluate(["x"], ["y"])
    second_meteor = evaluator._results["dummy"]["meteor"]

    assert first_meteor != second_meteor

    report = tmp_path / "re_eval_report.csv"
    evaluator.generate_report(report, models="dummy")
    df = pd.read_csv(report, index_col="model")
    assert pytest.approx(round(second_meteor, 4), rel=1e-6) == df.loc["dummy", "meteor"]
