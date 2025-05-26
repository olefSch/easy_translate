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

There is also the option to initialize the translator using LLMs like Gemini, GPT, or Claude. This is useful for more complex translation tasks that require understanding context or nuances in the text and to use prompts based on the promt library we offer.

```python title="Initialize LLM Translator"
from easy_nlp_translate import initialize_translator

# Initialize the LLM translator using its ID
translator = initialize_translator(
    translator_name="gemini", 
    model_name="gemini-1.5-pro",
    source_lang="en", 
    target_lang="de",
    prompt_type="formal",
)
```

!!! note "LLM Translator Parameters"
    The parameters for LLM translators may include `model_name`, `prompt_type`, and others specific to the LLM being used. Refer to the documentation for each LLM translator for more details.


These are the available llm translators:

| Translator | ID (`translator_name`) | Models | API_KEY env var |
|------------|------------------------|--------|-------------------|
| Gemini     | `gemini`               | `gemini-2.5-flash-preview-05-20`, `gemini-2.5-pro-preview-05-06`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`,`gemini-1.5-pro`| `GEMINI_API_KEY` |
| GPT        | `gpt`                  | coming soon... | `OPENAI_API_KEY`|
| Claude     | `claude`               | coming soon... | `ANTHROPIC_API_KEY`|

These are the available prompt types:

| Prompt Type                      | Description                      |
|----------------------------------|----------------------------------|
| default                          | Basic Translation                |
| formal                           | Formal Translation               |
| translate_and_summarize          | Summarize and Translate          |
| formal_translate_and_summarize | Formal Summarize and Translate   |
| romantic                         | Romantic Translation             |
| poetic                           | Poetic Translation               |

!!! note "We are working on costum prompts, so stay tuned for updates!"
    It will probably a paramter in the `initialize_translator` function to pass a custom prompt.
