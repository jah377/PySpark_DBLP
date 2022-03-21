import duckdb
import googletrans
import pyspark.sql.functions as fn
from constants import DATA_PATH
from extra_info import (load_extra_information_from_jsons, upload_duckdb,
                        upload_extra_information)
from googletrans import Translator
from pyspark.sql import SparkSession


def translate(input: str) -> str:
    """Translates the given string with the provided translator to english.

    Args:
        translator (googletrans.Translator): Translator object
        input (str): string to translate

    Returns:
        str: translated string to english
    """
    translator = Translator()
    translator.raise_Exception = True
    return translator.translate(input, dest="en").text


if __name__ == "__main__":
    spark = (SparkSession.builder
             .master("local")
             .config("spark.driver.bindAddress", "localhost")
             .getOrCreate())

    to_eng = fn.udf(translate, fn.StringType())

    train = spark.read.csv(f"{DATA_PATH}/train.csv")

    for i in range(1, 5):
        df = (spark.read.option("header", True)
              .csv(f"{DATA_PATH}/dblp-{i}.csv"))
        (df.withColumn("clean_author", fn.when(
            df.pauthor.endswith(".") & df.ptitle.contains("|"),
            df.ptitle).otherwise(df.pauthor))
         .withColumn("clean_title", fn.when(
             df.pauthor.endswith(".") & df.ptitle.contains("|"),
             df.pauthor).otherwise(df.ptitle))
         .withColumn("pyear", fn.abs(df.pyear))
        #  .withColumn("translated_title", to_eng('clean_title'))
         .drop(df.pauthor)
         .drop(df.ptitle)
         .show())

    con = con = duckdb.connect(database=':memory:')
    table_names = ["BOOKTITLE", "BOOKTITLEFULL",
                   "JOURNAL", "JOURNALFULL", "TYPE"]
    upload_extra_information(load_extra_information_from_jsons,
                             upload_duckdb, table_names, con)

    # To query for the data run con.execute("<QUERY>").fetchdf() like
    # con.execute("SELECT * FROM BOOKTITLE WHERE id = 1").fetchdf()
