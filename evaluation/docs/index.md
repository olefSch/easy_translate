# Easy Translate Evaluation

Welcome to the documentation for the Easy Translate Evaluation module — a tool for benchmarking translation quality using both traditional machine translation (MT) models and modern large language models (LLMs).

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

---

Explore the **Evaluation** section for detailed documentation on:

- [Overview](overview.md) - High-level architecture and evaluation process
- [Evaluation metrics](metrics.md)
- [Evaluator class](evaluator.md)
- [Model implementations](model.md)
- [Configuration setup](config.md)
- [Usage examples and workflows](usage.md)
