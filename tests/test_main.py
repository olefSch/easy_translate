from hello import say_hello


def test_say_hello():
    expecetd_value = "hello world!"
    actual_value = say_hello()
    assert actual_value == expecetd_value


def test_main():
    assert True
