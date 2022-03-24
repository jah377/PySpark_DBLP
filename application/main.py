import duckdb
import pandas as pd
import pyspark.sql.functions as fn
from constants import DATA_PATH
from extra_info import (load_extra_information_from_jsons, upload_duckdb,
                        upload_extra_information)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from utils import translate, clean_ints
import unidecode

if __name__ == "__main__":
    con = duckdb.connect(database=":memory:")
    table_names = ["BOOKTITLE", "BOOKTITLEFULL",
                   "JOURNAL", "JOURNALFULL", "TYPE"]
    upload_extra_information(load_extra_information_from_jsons,
                             upload_duckdb, table_names, con)
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

    spark = (SparkSession.builder
             .master("local")
             .config("spark.driver.bindAddress", "localhost")
             .getOrCreate())

    to_eng = fn.udf(translate, fn.StringType())
    clean_ints_udf = fn.udf(clean_ints, IntegerType())

    train = spark.read.csv(f"{DATA_PATH}/train.csv")

    data = []
    dblp = pd.DataFrame()
    for i in range(1, 5):
        df = (spark.read.option("header", True)
              .csv(f"{DATA_PATH}/dblp-{i}.csv"))
        df = (df.withColumn("author", fn.when(
            df.pauthor.endswith(".") & df.ptitle.contains("|"),
            df.ptitle).otherwise(df.pauthor))
            .withColumn("title", fn.when(
                df.pauthor.endswith(".") & df.ptitle.contains("|"),
                df.pauthor).otherwise(df.ptitle))
            .withColumn("pkey", fn.when(
                df.pbooktitle_id.contains("/"),
                df.pbooktitle_id).otherwise(df.pkey))
            .withColumn("pkey", fn.when(
                df.ptype_id.contains("/"),
                df.ptype_id).otherwise(fn.col("pkey")))
            .withColumn("pyear", fn.abs(df.pyear).cast(IntegerType()))
            .withColumn("clean_author", unidecode.unidecode(fn.translate(fn.col('author'), '|-', '  ')))
            .withColumn("clean_title", unidecode.unidecode(fn.translate(fn.col('title'), '|-', '  ')))
            .withColumn("ptype_id", clean_ints_udf(df.ptype_id))
            .withColumn("pjournal_id", clean_ints_udf(df.pjournal_id))
            .withColumn("pbooktitle_id", clean_ints_udf(df.pbooktitle_id))
            .withColumn("pjournalfull_id", clean_ints_udf(df.pjournalfull_id))
            .drop("pauthor", "ptitle", "partition", "_c0", "peditor", "author", "title")
            .toPandas())
        db2 = pd.concat([dblp, df]).reset_index(drop=True)
        print(db2)
        con.register("train_data", df)
        con.execute("INSERT INTO TRAIN SELECT * FROM train_data")
 
    # This query retrieves the extra data from the JSONs
    print(con.execute("""
        SELECT
            pyear as year, paddress as address, ppublisher as publisher,
            pseries as series, pid as id, pkey as key, clean_author, clean_title,
            TYPE.name as type, BOOKTITLE.name as book_title,
            BOOKTITLEFULL.name as full_book_title, JOURNAL.name as journal,
            JOURNALFULL.name as full_journal
        FROM TRAIN
        JOIN TYPE ON (TRAIN.ptype_id = TYPE.id)
        JOIN BOOKTITLE ON (TRAIN.pbooktitle_id = BOOKTITLE.id)
        JOIN BOOKTITLEFULL ON (TRAIN.pbooktitle_id = BOOKTITLEFULL.id)
        JOIN JOURNAL ON (TRAIN.pbooktitle_id = JOURNAL.id)
        JOIN JOURNALFULL ON (TRAIN.pbooktitle_id = JOURNALFULL.id)""").fetch_df())
