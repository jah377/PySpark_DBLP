import duckdb
import json
import pandas as pd
import pyspark.sql.functions as fn
import unidecode
from constants import DATA_PATH
from duckdb import DuckDBPyConnection
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import *
from typing import Any, Callable, Dict, List, Union
from utils import clean_ints


def get_json_dict(path: str) -> Dict:
    """Loads the json information at the given patha and returns
    the dictionary with the data.

    Args:
        path (str): path of the JSON to laod

    Returns:
        Dict: dictionary with JSON information
    """
    with open(path) as file:
        data = json.load(file)

    return data


def load_extra_information_from_jsons() -> List[Union[Dict, List[Dict]]]:
    """Loads the additional information for the book title, full book title,
    journal, full journal, and type.

    Returns:
        List[Union(Dict, List[Dict])]: List of extra data in dictionaries
    """
    jsons_data = []

    # Load book title info
    title_info = get_json_dict(f"{DATA_PATH}/pbooktitle.json")
    title_info["name"] = {int(k): unidecode.unidecode(v).strip()
                          if v is not None else v
                          for k, v in title_info["name"].items()}
    title_info["name"][0] = None
    jsons_data.append(title_info["name"])

    # Load book title full info
    title_full_info = get_json_dict(f"{DATA_PATH}/pbooktitlefull.json")
    title_full_info = [{**data,
                        "name": unidecode.unidecode(data["name"]).strip()}
                       if data["name"] is not None else data
                       for data in title_full_info]
    title_full_info[4]["name"] = None
    jsons_data.append(title_full_info)

    # Load journal info
    journal_info = get_json_dict(f"{DATA_PATH}/pjournal.json")
    journal_info = [{**data,
                     "name": unidecode.unidecode(data["name"]).strip()}
                    if data["name"] is not None else data
                    for data in journal_info]
    jsons_data.append(journal_info)

    # Load journal full info
    journal_full_info = get_json_dict(f"{DATA_PATH}/pjournalfull.json")
    journal_full_info = [{**data,
                          "name": unidecode.unidecode(data["name"]).strip()}
                         if data["name"] is not None else data
                         for data in journal_full_info]
    jsons_data.append(journal_full_info)

    # Load type info
    type_info = get_json_dict(f"{DATA_PATH}/ptype.json")
    type_info["name"] = {int(k): unidecode.unidecode(v).strip()
                         for k, v in type_info["name"].items()}
    jsons_data.append(type_info["name"])

    return jsons_data


def upload_duckdb(data_list: List[Union[Dict, List[Dict]]],
                  table_names: List[str],
                  con: duckdb.DuckDBPyConnection) -> None:
    """Uploads the list of data to DuckDB with the given table names and
    connection.

    Args:
        data_list (List[Union): List with data to upload
        table_names (List[str]): List of table names
        con (duckdb.DuckDBPyConnection): DuckDB connection
    """
    for name, data in zip(table_names, data_list):
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        else:
            df = pd.DataFrame(data)

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {name}(
                id int,
                name varchar,
                PRIMARY KEY(id))
        """)
        con.append(name, df)


def upload_extra_information(extra_data_function: Callable[[Any], Any],
                             upload_function: Callable[[Any], None], *args) -> None:
    """Uploads the returned extra data with the given upload function and
    any extra parameters.

    Args:
        extra_data_function (Callable[[Any], Any]): Function retrieving extra data
        upload_function (Callable[[Any], None]): Function uploading extra data
        arg
    """
    data = extra_data_function()
    upload_function(data, *args)


def create_train_table(con: DuckDBPyConnection) -> None:
    """Creates the main table to store the training data in DuckDB.

    Args:
        con (DuckDBPyConnection): DuckDB connection
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS TRAIN(
            pyear int,
            paddress varchar,
            ppublisher varchar,
            pseries varchar,
            pid varchar,
            pkey varchar,
            ptype_id int,
            pjournal_id int,
            pbooktitle_id int,
            pjournalfull_id varchar,
            pbooktitlefull_id int,
            clean_author varchar,
            clean_title varchar,
            PRIMARY KEY(pkey))
    """)


def clean_train_data(df: DataFrame) -> DataFrame:
    """Cleans the provided training spark DataFrame

    Args:
        df (DataFrame): Spark training DataFrame

    Returns:
        DataFrame: Clean Spark training DataFrame
    """
    clean_ints_udf = fn.udf(clean_ints, IntegerType())
    return (
        df.withColumn("clean_author", fn.when(
            df.pauthor.endswith(".") & df.ptitle.contains("|"),
            df.ptitle).otherwise(df.pauthor))
          .withColumn("clean_title", fn.when(
              df.pauthor.endswith(".") & df.ptitle.contains("|"),
              df.pauthor).otherwise(df.ptitle))
          .withColumn("pkey", fn.when(
              df.pbooktitle_id.contains("/"),
              df.pbooktitle_id).otherwise(df.pkey))
          .withColumn("pkey", fn.when(
              df.ptype_id.contains("/"),
              df.ptype_id).otherwise(fn.col("pkey")))
          .withColumn("pyear", fn.abs(df.pyear).cast(IntegerType()))
          .withColumn("ptype_id", clean_ints_udf(df.ptype_id))
          .withColumn("pjournal_id", clean_ints_udf(df.pjournal_id))
          .withColumn("pbooktitle_id", clean_ints_udf(df.pbooktitle_id))
          .withColumn("pjournalfull_id", clean_ints_udf(df.pjournalfull_id))
          .drop("pauthor", "ptitle", "partition", "_c0", "peditor")
    )


def get_and_upload_clean_train_data(
        con: DuckDBPyConnection,
        spark_session: SparkSession
) -> List[pd.DataFrame]:
    """Retrieves the training data in the different files, cleans it and
    uploads it to DuckDB.

    Args:
        con (DuckDBPyConnection): DudkDB connection
        spark_session (SparkSession): Spark session

    Returns:
        List[pd.DataFrame]: DataFrames with data from different files
    """
    data = []

    for i in range(1, 5):
        df = (spark_session.read.option("header", True)
                           .csv(f"{DATA_PATH}/dblp-{i}.csv"))
        df = clean_train_data(df).toPandas()
        con.register("train_data", df)
        con.execute("INSERT INTO TRAIN SELECT * FROM train_data")

    return data


def get_train_with_extra(con: DuckDBPyConnection) -> pd.DataFrame:
    """Retrieves the training data from the DuckDB database

    Args:
        con (DuckDBPyConnection): DuckDB connection

    Returns:
        pd.DataFrame: Merged training data
    """
    return (
        con.execute("""
            SELECT
                pyear as year, paddress as address, ppublisher as publisher,
                pseries as series, pid as id, pkey as key, clean_author, clean_title,
                TYPE.name as type, BOOKTITLE.name as book_title,
                BOOKTITLEFULL.name as full_book_title, JOURNAL.name as journal,
                JOURNALFULL.name as full_journal
            FROM TRAIN
            LEFT JOIN TYPE ON (TRAIN.ptype_id = TYPE.id)
            LEFT JOIN BOOKTITLE ON (TRAIN.pbooktitle_id = BOOKTITLE.id)
            LEFT JOIN BOOKTITLEFULL ON (TRAIN.pbooktitle_id = BOOKTITLEFULL.id)
            LEFT JOIN JOURNAL ON (TRAIN.pbooktitle_id = JOURNAL.id)
            LEFT JOIN JOURNALFULL ON (TRAIN.pbooktitle_id = JOURNALFULL.id)""")
        .fetch_df()
    )


def load_train_df_to_spark(spark_session: SparkSession, train_df: pd.DataFrame) -> DataFrame:
    """Loads the given pandas DataFrame to spark using the provided
    Spark session.

    Args:
        spark_session (SparkSession): spark session to load DataFrame
        train_df (pd.DataFrame): pandas DataFrame to load

    Returns:
        DataFrame: spark DataFrame
    """
    schema = StructType([
        StructField("year", StringType(), True),
        StructField("address", StringType(), True),
        StructField("publisher", StringType(), True),
        StructField("series", StringType(), True),
        StructField("id", StringType(), True),
        StructField("pkey", StringType(), True),
        StructField("clean_author", StringType(), True),
        StructField("clean_title", StringType(), True),
        StructField("type", StringType(), True),
        StructField("book_title", StringType(), True),
        StructField("full_book_title", StringType(), True),
        StructField("journal", StringType(), True),
        StructField("full_journal", StringType(), True)
    ])

    return spark_session.createDataFrame(train_df, schema=schema)
