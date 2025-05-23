import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .initialize import initialize_translator

__all__ = ["initialize_translator"]
