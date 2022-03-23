import duckdb
import os
import pyspark.sql.functions as fn
import time
from data import (create_train_table, get_and_upload_clean_train_data,
                  get_train_with_extra, load_extra_information_from_jsons,
                  load_train_df_to_spark, upload_duckdb,
                  upload_extra_information)
from pyspark.ml.classification import (LogisticRegression,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType
from typing import Iterable
from utils import start_spark


def jaccard_similarity(a: Iterable, b: Iterable) -> float:
    """Computes the Jaccard similarity between the two provided objects.

    Args:
        a (Iterable): Object to compute similarity
        b (Iterable): Object to compute similarity

    Returns:
        float: Jaccard similarity
    """
    a = set(a)
    b = set(b)
    # Calculate Jaccard similarity
    return len(a.intersection(b)) / len(a.union(b))


def getDfWithSuffixedColumns(df, suffix):
    return df.select([fn.col(c).alias(f"{c}{suffix}") for c in df.columns])


def mergeDatasetOnKey(suffix, df1, df2):
    df_new = getDfWithSuffixedColumns(df2, suffix)
    return df1.join(df_new, df1[f"key{suffix}"] == df_new[f"pkey{suffix}"], "left")


def calculateNumericFeatures(training_df):
    jaccard_udf = fn.udf(jaccard_similarity, FloatType())  # defining udf
    temp_df = training_df

    temp_df = temp_df.withColumn("jaccard_author", jaccard_udf(
        *["clean_author1", "clean_author2"]).cast("Double"))
    temp_df = temp_df.withColumn("jaccard_title", jaccard_udf(
        *["clean_title1", "clean_title2"]).cast("Double"))
    temp_df = temp_df.withColumn(
        "jaccard_key", jaccard_udf(*["key1", "key2"]).cast("Double"))
    temp_df = temp_df.withColumn("diff_year", temp_df.year2 - temp_df.year1)
    return temp_df


def prepareFeatures(features, data):
    """
    Brings the dataset into a format that can be used by the later on used algorithms
    """
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    transformed_data = assembler.transform(data)
    if "label" in transformed_data.columns:
        transformed_data = transformed_data.withColumn(
            "label", transformed_data.label.cast("boolean").cast("integer"))
    return transformed_data


def trainRandomForest(train):
    rf = RandomForestClassifier(labelCol="label",
                                featuresCol="features",
                                maxDepth=8)
    return "RandomForest", rf.fit(train)


def trainLogisticRegression(train):
    reg = LogisticRegression(labelCol="label")
    return "LogisticRegression", reg.fit(train)


def enrichIncomingDataset(data, cleaned_base_data):
    data = data.drop(*["_c0", "partition"])

    # merging training data with cleaned data and calculate new columns
    data = mergeDatasetOnKey("1", data, cleaned_base_data)
    data = mergeDatasetOnKey("2", data, cleaned_base_data)
    # if pkey1 is null after the initial merge, we have to fill the values with values from pkey2
    for c in ["year", "id", "key", "type", "journal", "clean_author", "clean_title"]:
        data = data.withColumn(f"{c}1",
                               fn.when(
                                   (fn.isnan(f"{c}1")
                                    | fn.col(f"{c}1").isNull()),
                                   fn.col(f"{c}2")).otherwise(fn.col(f"{c}1")))

    for c in ["year", "id", "key", "type", "journal", "clean_author", "clean_title"]:
        data = data.withColumn(f"{c}2",
                               fn.when(
                                   fn.isnan(f"{c}2") | fn.col(
                                       f"{c}2").isNull(),
                                   fn.col(f"{c}1")).otherwise(fn.col(f"{c}2")))

    data = calculateNumericFeatures(data)

    required_features = [
        "jaccard_author",
        "jaccard_title",
        "jaccard_key",
        "diff_year"
    ]

    return prepareFeatures(required_features, data)


if __name__ == "__main__":
    con = duckdb.connect(database=":memory:")
    table_names = ["BOOKTITLE", "BOOKTITLEFULL",
                   "JOURNAL", "JOURNALFULL", "TYPE"]
    upload_extra_information(load_extra_information_from_jsons,
                             upload_duckdb, table_names, con)
    create_train_table(con)
    spark = start_spark()

    full_data = get_and_upload_clean_train_data(con, spark)

    df = load_train_df_to_spark(spark, get_train_with_extra(con))

    # training
    train_df = spark.read.csv("data/train.csv", header=True)
    prepared_data = enrichIncomingDataset(train_df, df)

    print("Training set count:", train_df.count())
    print("Prepared Data Count", prepared_data.count())
    print()

    train, test_with_label = prepared_data.randomSplit([0.9, 0.1], seed=1000)

    name, model = trainRandomForest(train)

    rf_predicts = model.transform(test_with_label)

    test_df = spark.read.csv("data/test_hidden.csv", header=True)
    prepared_data = enrichIncomingDataset(test_df, df)

    timestamp = int(time.time())
    os.mkdir(f"submit/{timestamp}")

    predictions = model.transform(prepared_data)
    predictions = predictions.withColumn("prediction", fn.initcap(
        predictions.prediction.cast("Boolean").cast("String")))
    predictions.select("prediction").toPandas().to_csv(
        f"submit/{timestamp}/prediction.csv", index=False, header=False)

    print("Test set count:", test_df.count())
    print("Prepared Data Count", prepared_data.count())
    print("Predictions Count", predictions.count())
    print()

    # validation
    validation_df = spark.read.csv("data/validation_hidden.csv", header=True)
    prepared_data = enrichIncomingDataset(validation_df, df)

    validations = model.transform(prepared_data)
    validations = validations.withColumn("prediction", fn.initcap(
        validations.prediction.cast("Boolean").cast("String")))
    validations.select("prediction").toPandas().to_csv(
        f"submit/{timestamp}/validation.csv", index=False, header=False)

    print("Validation set count:", validation_df.count())
    print("Prepared Data Count", prepared_data.count())
    print("Predictions Count", validations.count())
    print()

    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                        metricName="accuracy")
    print(f"{name} Accuracy:", multi_evaluator.evaluate(rf_predicts))

    # TODO loading and saving, see comments below 👇

    # loading doesnt work that way:
    #   py4j.protocol.Py4JJavaError: An error occurred while calling o227.load.
    #   java.lang.NoSuchMethodException: org.apache.spark.ml.classification.RandomForestClassificationModel.<init>(java.lang.String)
    # name, model = "RF", RandomForestClassifier.read().load("test_rf")

    # only save model when loading actually works
    # model.write().overwrite().save("test_rf")

    spark.stop()
    print("Finished model training. Stopped SparkSession. Good Bye.")
