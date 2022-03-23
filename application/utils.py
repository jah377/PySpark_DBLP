from googletrans import Translator
from typing import Union


def clean_ints(x: Union[str, None]) -> Union[int, None]:
    """Correctly formats the given string to be an integer.

    Args:
        x (Union[str, None]): String to format

    Returns:
        Union[int, None]: Integer formated or None if unformattable
    """
    return int(x) if x is not None and x.isdigit() else None


def translate(x: str) -> str:
    """Translates the given string with the provided translator to english.

    Args:
        translator (googletrans.Translator): Translator object
        x (str): string to translate

    Returns:
        str: translated string to english
    """
    translator = Translator()
    return translator.translate(x, dest="en").text
