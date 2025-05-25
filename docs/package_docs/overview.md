# General Instructions on How to Use the Package

This package allows you to create translator objects for translating text according to your needs.

## Available Translators

The following translators are available, each identified by a unique ID:

| Translator | ID (`translator_name`) | Description                                                                                                |
|------------|------------------------|------------------------------------------------------------------------------------------------------------|
| MBart50    | `mbart`                | A multilingual translation model supporting 50 languages. It can also run locally without external dependencies. This is also the best machine translator out of the evaluation we did.|
| Gemini     | `gemini`               | *Coming soon...* |
| GPT        | `gpt`                  | *Coming soon...* |
| Claude     | `claude`               | *Coming soon...* |

## Basic Usage

To use a translator, you first need to initialize it using its `translator_name` ID.

```python title="Initialize the Translator"
from easy_nlp_translate import initialize_translator

# Initialize the translator using its ID
translator = initialize_translator(
    translator_name="mbart", 
    source_lang="en", 
    target_lang="de"
)
```

!!! note "Initialization Parameters"
    For more details on `source_lang`, `target_lang`, and any other parameters specific to each translator (like API keys or model paths), please refer to the detailed documentation for the respective translator classes.

Once the translator object is created, you can use its `translate` method to translate your text.

```python title="Translating Text"
translated_text = translator.translate("My dog is beautiful.")

print(translated_text)
# Expected output (will vary based on the model and target language):
# Mein Hund ist wunderschön.
```

You can also translate multiple texts in a batch if the translator supports it (e.g., the `translate_batch` method). Please check the specific translator's documentation for availability and usage of batch translation.

## LLM Translator Initialization

comming ...
