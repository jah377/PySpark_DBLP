from pyspark.sql import SparkSession
from pyspark.sql import Row
from googletrans import Translator
import pyspark.sql.functions as fn
from pyspark.sql.types import ArrayType, FloatType, StringType
import pyspark.sql.types as tp
translator = Translator()
def translate(c):
  return translator.translate(c, dest='en').text

spark = SparkSession.builder \
    .master("local") \
    .config("spark.driver.bindAddress","localhost") \
    .getOrCreate()


#to_eng = fn.udf(translate, StringType())
def upperCase(str):
    return str.upper()
f = fn.udf(lambda x: "", StringType())

train = spark.read.csv(f'application/data/train.csv')
for i in range(1,5):
    df = spark.read.option("header",True).csv(f'application/data/dblp-{i}.csv')
    df.withColumn('clean_author',fn.when(df.pauthor.endswith(".") & df.ptitle.contains('|'),df.ptitle).otherwise(df.pauthor))\
      .withColumn('clean_title',fn.when(df.pauthor.endswith(".") & df.ptitle.contains('|'),df.pauthor).otherwise(df.ptitle))\
      .withColumn('pyear', fn.abs(df.pyear))\
      .select(f(fn.col('ptitle')).alias('translated'))\
      .drop(df.pauthor)\
      .drop(df.ptitle)\
      .show()
