# Evaluation Overview

The Easy Translate Evaluation framework is designed to benchmark the performance of machine translation models and large language models across multiple languages, using consistent metrics and datasets.

---

## Goal

To provide a reproducible, flexible, and extensible pipeline for:

- Evaluating translation quality using standard metrics
- Comparing traditional NMT models with modern local LLM-based translators
- Identify an effective model for local, offline translation
---

## Architecture

The evaluation pipeline consists of:

1. **Dataset Preparation**  
   Loads subsets of the [WMT-19 dataset](https://huggingface.co/datasets/wmt/wmt19) for selected language pairs

2. **Model Translation**  
   Translates sentences using various models (Hugging Face models, local LLMs via Ollama)

3. **Scoring**  
   Calculates quality metrics (e.g., BLEU, METEOR) against gold-standard references

4. **Reporting**  
   Outputs aggregate scores and reports
---

## Key Features

- **Multilingual support** for 6 language pairs (to English)
- **Modular model interface** for both transformer-based and LLM-based models
- **Metric flexibility** with support for BLEU, METEOR
- **Reproducibility** via configuration files and version-pinned models
- **Custom language mappings** per model type (NLLB, MBART, Marian, etc.)

---

## Next Steps

- [Metrics](metrics.md) - Details on BLEU, METEOR, and how translation quality is measured
- [Evaluator](evaluator.md) - Core evaluation logic: how models are registered, run, and scored
- [Models](model.md) - Overview of model implementations, including MBart, Marian, and LLMs
- [Config](config.md) - Centralized setup for model selection, dataset use, and language mappings
- [Usage](usage.md) - How to run evaluations and generate reports step by step
