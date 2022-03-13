from pyspark.sql import SparkSession
from pyspark.sql import Row

import pyspark.sql.functions as fn
import pyspark.sql.types as tp


spark = SparkSession.builder \
    .master("local") \
    .config("spark.driver.bindAddress","localhost") \
    .getOrCreate()
