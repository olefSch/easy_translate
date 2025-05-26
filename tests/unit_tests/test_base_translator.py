import pytest
from langdetect import DetectorFactory
from easy_nlp_translate.exceptions import DetectionError

DetectorFactory.seed = 0


class TestTranslatorBase:
    def test_init_successful(self, concrete_translator_class):
        """
        Test successful initialization with valid languages.
        """
        translator_de_en = concrete_translator_class(
            target_lang="de", source_lang="en"
        )
        assert translator_de_en.target_lang == "de"
        assert translator_de_en.source_lang == "en"

        translator_custom_de = concrete_translator_class(
            target_lang="custom_test_lang", source_lang="de"
        )
        assert translator_custom_de.target_lang == "custom_test_lang"
        assert translator_custom_de.source_lang == "de"

    def test_init_auto_detect_source(self, concrete_translator_class):
        """
        Test successful initialization when source_lang is None (auto-detect).
        """
        translator = concrete_translator_class(target_lang="de")
        assert translator.target_lang == "de"
        assert translator.source_lang is None

    def test_init_invalid_target_lang(self, concrete_translator_class):
        """
        Test initialization fails with an invalid target_lang.
        """
        with pytest.raises(ValueError) as excinfo:
            concrete_translator_class(target_lang="zz", source_lang="en")
        assert "Language 'zz' is not supported" in str(excinfo.value)
        assert str(["en", "de", "custom_test_lang"]) in str(excinfo.value)

    def test_init_invalid_source_lang(self, concrete_translator_class):
        """
        Test initialization fails with an invalid source_lang.
        """
        with pytest.raises(ValueError) as excinfo:
            concrete_translator_class(target_lang="en", source_lang="zz")
        assert "Language 'zz' is not supported" in str(excinfo.value)
        assert str(["en", "de", "custom_test_lang"]) in str(excinfo.value)

    def test_init_same_source_target_lang(self, concrete_translator_class):
        """
        Test initialization fails if source and target languages are the same.
        """
        with pytest.raises(ValueError) as excinfo:
            concrete_translator_class(target_lang="en", source_lang="en")
        assert "Source and target languages cannot be the same" in str(
            excinfo.value
        )

    def test_validate_langauge_valid(self, concrete_translator_class):
        """
        Test _validate_langauge with a valid language based on ConcreteTranslator's LANGUAGE_CODES.
        """
        translator = concrete_translator_class(
            target_lang="de", source_lang="en"
        )
        translator._validate_langauge("en")
        translator._validate_langauge("de")
        translator._validate_langauge("custom_test_lang")

    def test_validate_langauge_invalid(self, concrete_translator_class):
        """
        Test _validate_langauge with an invalid language based on ConcreteTranslator's LANGUAGE_CODES.
        """
        translator = concrete_translator_class(
            target_lang="de", source_lang="en"
        )
        with pytest.raises(ValueError) as excinfo:
            translator._validate_langauge("zz")
        assert "Language 'zz' is not supported" in str(excinfo.value)
        assert str(["en", "de", "custom_test_lang"]) in str(excinfo.value)

    # --- Tests for detect_language ---
    def test_detect_language_english(self, concrete_translator_class):
        """
        Test detect_language for English text (which is in ConcreteTranslator.LANGUAGE_CODES).
        """
        translator = concrete_translator_class(target_lang="de")
        assert (
            translator.detect_language("The green cat eats the mouse.") == "en"
        )

    def test_detect_language_german(self, concrete_translator_class):
        """
        Test detect_language for German text (which is in ConcreteTranslator.LANGUAGE_CODES).
        """
        translator = concrete_translator_class(target_lang="en")
        assert translator.detect_language("Der Hund jagt die Katze.") == "de"

    def test_detect_language_unsupported_by_concrete_translator_but_detectable(
        self, concrete_translator_class
    ):
        """
        Test detect_language when langdetect finds a language NOT in ConcreteTranslator.LANGUAGE_CODES.
        """
        translator = concrete_translator_class(target_lang="en")
        with pytest.raises(ValueError) as excinfo:
            translator.detect_language("Ceci est une phrase en français.")
        assert "Language 'fr' is not supported" in str(excinfo.value)
        assert str(["en", "de", "custom_test_lang"]) in str(excinfo.value)

    def test_detect_language_failure_short_text(
        self, concrete_translator_class
    ):
        """
        Test detect_language failure for very short or ambiguous text.
        """
        translator = concrete_translator_class(target_lang="en")
        with pytest.raises(DetectionError):
            translator.detect_language("?!")

    def test_detect_language_invalid_input_text(
        self, concrete_translator_class
    ):
        """
        Test detect_language with invalid input text (e.g., empty string).
        """
        translator = concrete_translator_class(target_lang="en")
        with pytest.raises(ValueError) as excinfo:
            translator.detect_language("   ")
        assert "Text to translate must be a non-empty string" in str(
            excinfo.value
        )

    # --- Tests for _validate_basic_text_to_translate (staticmethod on TranslatorBase) ---
    def test_validate_basic_text_to_translate_valid(
        self, translator_base_class
    ):
        """
        Test _validate_basic_text_to_translate with valid text.
        """
        translator_base_class._validate_basic_text_to_translate("Hello world")

    def test_validate_basic_text_to_translate_empty(
        self, translator_base_class
    ):
        """
        Test _validate_basic_text_to_translate with empty string.
        """
        with pytest.raises(ValueError) as excinfo:
            translator_base_class._validate_basic_text_to_translate("")
        assert "Text to translate must be a non-empty string" in str(
            excinfo.value
        )

    def test_validate_basic_text_to_translate_whitespace_only(
        self, translator_base_class
    ):
        """
        Test _validate_basic_text_to_translate with whitespace-only string.
        """
        with pytest.raises(ValueError) as excinfo:
            translator_base_class._validate_basic_text_to_translate(
                "   \t\n  "
            )
        assert "Text to translate must be a non-empty string" in str(
            excinfo.value
        )

    # --- Tests for translate_batch ---
    def test_translate_batch_successful(self, concrete_translator_class):
        """
        Test translate_batch with a list of valid texts.
        """
        translator = concrete_translator_class(
            target_lang="de", source_lang="en"
        )
        texts = ["Hello", "World"]
        expected_translations = ["translated_de:Hello", "translated_de:World"]
        assert translator.translate_batch(texts) == expected_translations

    def test_translate_batch_empty_list(self, concrete_translator_class):
        """
        Test translate_batch with an empty list.
        """
        translator = concrete_translator_class(
            target_lang="de", source_lang="en"
        )
        assert translator.translate_batch([]) == []

    def test_translate_batch_with_invalid_text_in_batch(
        self, concrete_translator_class
    ):
        """
        Test translate_batch fails if any text in the batch is invalid.
        """
        translator = concrete_translator_class(
            target_lang="de", source_lang="en"
        )
        texts = ["Hello", "   ", "World"]
        with pytest.raises(ValueError) as excinfo:
            translator.translate_batch(texts)
        assert "Text to translate must be a non-empty string" in str(
            excinfo.value
        )
