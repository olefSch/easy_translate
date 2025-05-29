---
description: Easy NLP Translate is a python library designed to simplify text translation. It offers an easy-to-use interface for high-quality results, leveraging both carefully evaluated local transformer models and powerful Large Language Models (LLMs) from popular providers. Customize your translations with flexible prompt options. Requires Python 3.11 or 3.12.---
---
# Easy NLP Translate - A Simple and Efficient Translation Library

This library provides an easy-to-use interface for translating text using both local transformer models and large language models (LLMs) from popular providers. It is designed to simplify the translation process while ensuring high-quality results through careful model evaluation also shown in the documentation.

## 🚀 Features

- Translation with a local transformer model selected based on our evaluation
- LLM-Translation using common provider with a custom prompt or based on our prompt library

## 📦 Installation & Instructions

Install the package from PyPi. It requires Python 3.11 or 3.12:

```bash
pip install easy-nlp-translate
```

## 🏗️ Example Project where easy-nlp-translate gets used

This [Repo](https://github.com/philipp-mey/nlp_frontend) shows an example project where `easy-nlp-translate` is used to translate video subtitles. It demonstrates how to integrate the library into a project and utilize its features effectively. The project includes a Streamlit frontend which enables users to upload videos in order to generate subtitles. Those subtitles then get translated using `easy-nlp-translate` and then saved as SRT files.

## 🧾 License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/olefSch/easy_translate/blob/main/LICENSE) file for details.
