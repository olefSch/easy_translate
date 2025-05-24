from .initialize import initialize_translator
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["initialize_translator"]
