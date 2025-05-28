import pytest
import os


# --- initilize_translator function
@pytest.fixture(autouse=True)
def unregister_ollama_translator_for_test(monkeypatch):
    """
    Unregister the 'ollama' translator from the registry for testing purposes.
    This fixture ensures that the 'ollama' translator is not available during tests,
    allowing for testing of other translators without interference.
    """
    is_ci_environment = os.getenv("CI", "false").lower() in ("true", "1")
    if is_ci_environment:
        registry_path = "easy_nlp_translate.initialize.TRANSLATOR_REGISTRY"

        module_path, object_name = registry_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[object_name])
        original_registry = getattr(module, object_name)

        if "ollama" in original_registry:
            monkeypatch.delitem(original_registry, "ollama")
