# Evaluation Overview

The Easy Translate Evaluation framework is designed to benchmark the performance of machine translation models and large language models across multiple languages, using consistent metrics and datasets.

---

## Goal

To provide a reproducible, flexible, and extensible pipeline for:

- Evaluating translation quality using standard metrics
- Comparing traditional NMT models with modern local LLM-based translators
- Identify an effective model for local, offline translation
---

## Dataset: WMT-19

The [WMT-19](https://huggingface.co/datasets/wmt/wmt19) (Workshop on Machine Translation 2019) dataset serves as the primary benchmark for this evaluation. It includes professionally curated parallel corpora across many language pairs and is widely adopted in machine translation research.

Key attributes:

- High-quality human references for English translations
- Standardized formatting, compatible with MT toolkits
- A sample of 1,000 sentence pairs per language was selected to manage computational requirements

---

## Language Pairs

Translations were evaluated **from six source languages into English**, using subsets of the WMT-19 dataset:

- 🇩🇪 German → English (`de-en`)
- 🇫🇮 Finnish → English (`fi-en`)
- 🇮🇳 Gujarati → English (`gu-en`)
- 🇱🇹 Lithuanian → English (`lt-en`)
- 🇷🇺 Russian → English (`ru-en`)
- 🇨🇳 Chinese → English (`zh-en`)

---

## Tools & Frameworks

Two model types were included in the evaluation:

- Transformer-based machine translation models, accessed through the [Hugging Face Transformers](https://huggingface.co/docs/transformers) library
- Locally-run LLMs such as LLaMA, Mistral, Gemma, and Phi, deployed via [Ollama](https://ollama.com/)

This setup enables comparison between domain-specific MT systems and general-purpose LLMs in translation tasks.

---

## Models Evaluated

| Model      | Type           | Size (Parameters) | Description |
|------------|----------------|-------------------|-------------|
| `mbart50`  | MT-specific    | ~610M             | Facebook’s multilingual encoder-decoder model for 50+ languages. |
| `marian`   | MT-specific    | ~280M             | Fast, language-pair–specific model from Helsinki-NLP. |
| `nllb`     | MT-specific    | 600M (distilled)  | Facebook’s "No Language Left Behind" (distilled version). |
| `m2m100`   | MT-specific    | ~418M             | Facebook’s many-to-many multilingual model supporting 100+ languages. |
| `mistral`  | General LLM    | 7B                | Decoder-only open LLM (Mistral AI); not trained for translation. |
| `llama3.1` | General LLM    | 8B                | Meta’s LLaMA 3 model (8B), with instruction tuning and multilingual potential. |
| `llama3.2` | General LLM    | 3B                | Smaller LLaMA 3 variant (3B); weaker performance on translation tasks. |
| `gemma`    | General LLM    | 4B                | Google’s compact open-source LLM, multilingual-capable. |
| `phi3`     | General LLM    | 3.8B              | Microsoft’s small LLM designed for efficiency; weak on translation. |

---

## Language Mappings

Each model architecture requires different language code formats. The following table illustrates the mappings used for the `de-en` (German to English) direction:

| Model Type                   | Source Lang | Target Lang |
|-----------------------------|-------------|-------------|
| `nllb`                      | `deu_Latn`  | `eng_Latn`  |
| `m2m100`                    | `de`        | `en`        |
| `mbart50`                   | `de_DE`     | `en_XX`     |
| `marian`                    | `de`        | `en`        |
| `llama/gemma/phi3/mistral` | `German`    | `English`   |

Equivalent mappings were defined for all language pairs during evaluation in the configurations.

## Evaluation Results

### BLEU Scores

| base\_model | de-en  | fi-en  | gu-en  | lt-en  | ru-en  | zh-en  |
| ----------- | ------ | ------ | ------ | ------ | ------ | ------ |
| gemma       | 0.2173 | 0.1929 | 0.0980 | 0.1756 | 0.2208 | 0.2253 |
| llama3.1    | 0.2054 | 0.1730 | 0.0951 | 0.1000 | 0.1953 | 0.1957 |
| llama3.2    | 0.1757 | 0.1314 | 0.0215 | 0.0768 | 0.1741 | 0.1806 |
| m2m100      | 0.2223 | 0.2163 | 0.0193 | 0.2484 | 0.2072 | 0.2026 |
| marian      | 0.2899 | 0.2946 | NaN    | NaN    | 0.1930 | 0.2573 |
| mbart50     | 0.2859 | 0.2928 | 0.8241 | 0.5610 | 0.2307 | 0.2721 |
| mistral     | 0.1812 | 0.1144 | 0.0103 | 0.0388 | 0.1934 | 0.1803 |
| nllb        | 0.2726 | 0.2570 | 0.1026 | 0.2570 | 0.2141 | 0.2476 |
| phi3        | 0.1520 | 0.0318 | 0.0000 | 0.0099 | 0.0951 | 0.1204 |

### METEOR Scores

| base\_model | de-en  | fi-en  | gu-en  | lt-en  | ru-en  | zh-en  |
| ----------- | ------ | ------ | ------ | ------ | ------ | ------ |
| gemma       | 0.5098 | 0.4902 | 0.3822 | 0.4187 | 0.4517 | 0.5118 |
| llama3.1    | 0.5060 | 0.4724 | 0.3897 | 0.3820 | 0.4291 | 0.4921 |
| llama3.2    | 0.4708 | 0.4113 | 0.3160 | 0.3071 | 0.3952 | 0.4691 |
| m2m100      | 0.5106 | 0.5107 | 0.0724 | 0.5055 | 0.4219 | 0.4725 |
| marian      | 0.5754 | 0.5048 | NaN    | NaN    | 0.4955 | 0.5662 |
| mbart50     | 0.5609 | 0.5384 | 0.6526 | 0.8063 | 0.4937 | 0.5777 |
| mistral     | 0.4772 | 0.3782 | 0.0862 | 0.2632 | 0.4135 | 0.4740 |
| nllb        | 0.5455 | 0.4375 | 0.3934 | 0.5194 | 0.4784 | 0.5166 |
| phi3        | 0.3825 | 0.2192 | 0.0584 | 0.1779 | 0.3214 | 0.3948 |

### Averaged Scores

|base\_model | BLEU   | METEOR |
| -------------- | ------ | ------ |
| mbart50        | 0.4122 | 0.6139 |
| marian         | 0.2588 | 0.5465 |
| nllb           | 0.2252 | 0.5106 |
| gemma          | 0.1883 | 0.4607 |
| m2m100         | 0.1860 | 0.4405 |
| llama3.1       | 0.1608 | 0.4452 |
| llama3.2       | 0.1267 | 0.3949 |
| mistral        | 0.1182 | 0.3493 |
| phi3           | 0.0682 | 0.2693 |

---

