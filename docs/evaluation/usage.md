# Usage Guide

This section explains how to run the evaluation script, configure model selection, and generate reports for translation model performance.

The main entry point for the framework is `main.py`. It handles configuration loading, dataset preparation, translation execution, and report generation.

---

## Running the Evaluation

To start the evaluation, run the following command from the project root:

```bash
python main.py
```

This will:

1. Load the language mappings from `configs/language_mappings.yaml`
2. Instantiate a `TranslationEvaluator` object
3. Evaluate each model listed in `MODELS_TO_EVALUATE` across supported language pairs
4. Generate a `.csv` report for each model-language pair in the `report/` directory

## Configuration Overview

All paths and evaluation parameters are defined in `config.py`

```python
MODELS_TO_EVALUATE = ["mbart50", "nllb", "mistral"]
MODEL_REGISTRY = {
    "nllb": NllbTranslator,
    "mbart50": MBartTranslator,
    "mistral": partial(LLMTranslator, model_name="mistral:7b")
}
DATASET_NAME = "wmt19"
DATASET_SPLIT = "train[:1000]"
OUTPUT_DIR = Path("reports")
LANGUAGE_MAPPING_PATH = Path("configs/language_mappings.yaml")
```

To evaluate different models or use a smaller dataset split, update these values before running `main.py`.

## What Happens Internally

**Load Language Mappings**

```python
mappings = load_language_mappings(LANGUAGE_MAPPING_PATH)
```

Loads a YAML file that maps each language pair to the model-specific source and target language codes.

**Loop Through Models and Language Pairs**

Each model in `MODELS_TO_EVALUATE` is evaluated on each language pair for which it has a mapping:
```python
for model_name in models:
    for lang_pair in mappings:
        if model_name in mappings[lang_pair]:
            ...
```

**Load Dataset**

Translations and reference texts are pulled from Hugging Face datasets:
```python
ds = load_dataset(dataset_name, lang_pair, split=split)
inputs = [ex["translation"][src_lang] for ex in ds]
refs = [ex["translation"][tgt_lang] for ex in ds]
```

**Initialize Translator**

Each translator class is dynamically loaded from `MODEL_REGISTRY`:
```python
translator_cls = MODEL_REGISTRY[model_name]
translator = translator_cls(source_lang=src_code, target_lang=tgt_code)
```

**Evaluate Translations**

The `TranslationEvaluator` handles translation and scoring:
```python
evaluator.register_model(model_id, translator)
evaluator.evaluate(inputs, refs, model_names=[model_id])
```

**Generate Report**

Results are saved as CSV files in the output directory:
```python
report_file = output_dir / f"{model_id}_report.csv"
evaluator.generate_report(report_file, models=model_id)
```
Each report includes scores for BLEU and METEOR.

## Output Files
After a successful run, the following will be available:

```text
reports/
├── mbart50_de-en_report.csv
├── nllb_fi-en_report.csv
├── mistral_ru-en_report.csv
...
```
Each file contains the metric results for a specific model and language pair.

## Common Issues

| Issue                           | Explanation                                                                                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeyError: 'language_mappings'` | Check the YAML file formatting and top-level key.                                                                                                           |
| `FileNotFoundError`             | Make sure the paths in `config.py` are correct.                                                                                                             |
| `Unknown model 'xyz'`           | Ensure all listed models exist in `MODEL_REGISTRY`.                                                                                                         |
| Hugging Face dataset error      | The dataset or language pair may not be supported. Try a different one or check availability on [huggingface.co/datasets](https://huggingface.co/datasets). |