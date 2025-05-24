import pytest


def test_init_defaults_and_device_check(
    patched_huggingface_translator_class,
    mock_tokenizer_instance,
    mock_model_instance,
):
    """
    Tests initialization with default values and checks if the device is either 'cpu' or 'cuda'.
    The actual device ('cpu' or 'cuda') depends on torch.cuda.is_available() at class definition time
    or how it's mocked here.
    """
    translator = patched_huggingface_translator_class(
        source_lang="en", target_lang="fr"
    )

    assert translator.device in ("cpu", "cuda")

    assert translator.max_length == 512
    assert translator.num_beams == 4

    assert translator.source_lang == "en"
    assert translator.target_lang == "fr"

    assert translator.tokenizer is mock_tokenizer_instance
    assert translator.model is mock_model_instance


@pytest.mark.parametrize(
    "max_l_param, num_b_param, expected_error_msg_part",
    [
        (0, 4, "max_length must be a positive integer."),
        (-1, 4, "max_length must be a positive integer."),
        ("abc", 4, "max_length must be a positive integer."),
        (512, 0, "num_beams must be a positive integer."),
        (512, -1, "num_beams must be a positive integer."),
        (512, "xyz", "num_beams must be a positive integer."),
        (0, 0, "max_length must be a positive integer."),
    ],
)
def test_init_validation_errors(
    patched_huggingface_translator_class,
    max_l_param,
    num_b_param,
    expected_error_msg_part,
):
    with pytest.raises(ValueError, match=expected_error_msg_part):
        patched_huggingface_translator_class(
            target_lang="fr", max_length=max_l_param, num_beams=num_b_param
        )
