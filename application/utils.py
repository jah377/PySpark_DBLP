from googletrans import Translator
from pyspark.sql import SparkSession
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


def start_spark() -> SparkSession:
    """Creates a spark session on the local machine.

    Returns:
        SparkSession: spark session object
    """
    return (SparkSession.builder
            .master("local")
            .config("spark.driver.bindAddress", "localhost")
            .config("spark.driver.port", "8080")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.host", "localhost")
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.default.parallelism", "2")
            .config("spark.shuffle.io.retryWait", "2000ms")
            .config("spark.shuffle.io.maxRetries", "2")
            .getOrCreate())
