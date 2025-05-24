# Configuration

This section describes the global configuration used in the translation evaluation framework. The configuration defines available models, evaluation targets, dataset settings, and language-specific code mappings for each model type.

---

## Overview

All configuration parameters are stored in `config.py` and `language_mappings.yaml`. These define:

- The available translation models
- The models to be evaluated in the current run
- Dataset split settings
- Output directory paths
- Language code mappings by model and language pair

---

## Model Registry

Translation models are registered in the `MODEL_REGISTRY` dictionary. Each key corresponds to a model name used in evaluation scripts, and the value is either a translator class or a pre-configured partial constructor.

```python
MODEL_REGISTRY = {
    "nllb":     NllbTranslator,
    "m2m100":   M2M100Translator,
    "mbart50":  MBartTranslator,
    "marian":   MarianTranslator,
    "llama3.2": partial(LLMTranslator, model_name="llama3.2:3b"),
    "llama3.1": partial(LLMTranslator, model_name="llama3.1:8b"),
    "gemma":    partial(LLMTranslator, model_name="gemma3:4b"),
    "phi3":     partial(LLMTranslator, model_name="phi3:3.8b"),
    "mistral":  partial(LLMTranslator, model_name="mistral:7b"),
}
```

## Models to Evaluate

The `MODELS_TO_EVALUATE` list determines which models are run during evaluation:

```python
MODELS_TO_EVALUATE = [
    "mistral",
    "m2m100",
    "marian",
    "nllb",
    "mbart50",
    "llama3.1",
    "llama3.2",
    "gemma",
    "phi3",
]
```

## Dataset Configuration

Evaluation is based on the WMT-19 dataset using a controlled subset:

```python
DATASET_NAME = "wmt19"
DATASET_SPLIT = "train[:1000]"
```

- `DATASET_NAME` defines the source dataset.
- `DATASET_SPLIT` specifies a sample of 1,000 sentence pairs per language pair for evaluation.

## Output Directory

Generated evaluation reports are saved to:

```python
OUTPUT_DIR = Path("reports")
```

## Language Code Mappings

Each translation model requires specific source/target language code formats. These mappings are defined in the external YAML file:

```python
LANGUAGE_MAPPING_PATH = Path("configs/language_mappings.yaml")
```

Sample structure from language_mappings.yaml:

```python
de-en:
  nllb:    {source: deu_Latn,  target: eng_Latn}
  mbart50: {source: de_DE,     target: en_XX}
  marian:  {source: de,        target: en}
  mistral: {source: German,    target: English}
```
This format is repeated for each language pair (fi-en, gu-en, etc.), and each entry is model-specific. This allows exact alignment with the required format for each tokenizer and model.

## Summary

| Config Parameter        | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| `MODEL_REGISTRY`        | Maps model names to translator classes   |
| `MODELS_TO_EVALUATE`    | List of models to run                    |
| `DATASET_NAME`          | Defines which dataset to use             |
| `DATASET_SPLIT`         | Limits dataset to manageable size        |
| `OUTPUT_DIR`            | Destination for reports and results      |
| `LANGUAGE_MAPPING_PATH` | Path to language code definitions (YAML) |