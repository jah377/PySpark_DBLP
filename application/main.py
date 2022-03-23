import duckdb
import pandas as pd
import pyspark.sql.functions as fn
from constants import DATA_PATH
from extra_info import (load_extra_information_from_jsons, upload_duckdb,
                        upload_extra_information)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from utils import translate, clean_ints


if __name__ == "__main__":
    con = duckdb.connect(database=":memory:")
    table_names = ["BOOKTITLE", "BOOKTITLEFULL",
                   "JOURNAL", "JOURNALFULL", "TYPE"]
    upload_extra_information(load_extra_information_from_jsons,
                             upload_duckdb, table_names, con)

    # To query for the data run con.execute("<QUERY>").fetchdf() like
    print(con.execute("SELECT * FROM BOOKTITLE WHERE id = 1").fetchdf())

    spark = (SparkSession.builder
             .master("local")
             .config("spark.driver.bindAddress", "localhost")
             .getOrCreate())

    to_eng = fn.udf(translate, fn.StringType())
    clean_ints_udf = fn.udf(clean_ints, IntegerType())

    train = spark.read.csv(f"{DATA_PATH}/train.csv")

    data = []
    for i in range(1, 5):
        df = (spark.read.option("header", True)
              .csv(f"{DATA_PATH}/dblp-{i}.csv"))
        df = (df.withColumn("clean_author", fn.when(
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
            .toPandas())
        con.register("train_data", df)
        con.execute("INSERT INTO TRAIN SELECT * FROM train_data")
