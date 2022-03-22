import pyspark.sql.functions as fn
import pyspark.sql.types as tp
from pyspark.sql import SparkSession, DataFrame

from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


def startSparkSession():
    return (SparkSession.builder
                .master("local")
                .config("spark.driver.bindAddress", "localhost")
                .config("spark.driver.port", "8080")
                .config("spark.driver.memory", "2g")
                .config("spark.driver.host", "localhost")
                .config("spark.dynamicAllocation.enabled", "true" )
                .config("spark.default.parallelism", "2" )
                .config("spark.shuffle.io.retryWait", "2000ms" )
                .config("spark.shuffle.io.maxRetries", "2" )
                .getOrCreate())

def jaccard_similarity(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    # calucate jaccard similarity
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j


def getDfWithSuffixedColumns(df, suffix):
    return df.select([fn.col(c).alias(f"{c}{suffix}") for c in df.columns])

def mergeDatasetOnKey(suffix, df1, df2): 
    df_new = getDfWithSuffixedColumns(df2, suffix=suffix)
    return df1.join(df_new, df1[f"key{suffix}"]==df_new[f"pkey{suffix}"], "inner")

def calculateNumericFeatures(training_df):
    jaccard_udf = fn.udf(jaccard_similarity, fn.StringType()) # defining udf
    temp_df = training_df

    temp_df = temp_df.withColumn("jaccard_author", jaccard_udf(*["clean_author1", "clean_author2"]).cast("Double"))
    temp_df = temp_df.withColumn("jaccard_title", jaccard_udf(*["clean_title1", "clean_title2"]).cast("Double"))
    temp_df = temp_df.withColumn("jaccard_key", jaccard_udf(*["key1", "key2"]).cast("Double"))
    temp_df = temp_df.withColumn("diff_year", temp_df.pyear2 - temp_df.pyear1)
    return temp_df

def prepareFeatures(features, data):
    '''
    Brings the dataset into a format that can be used by the later on used algorithms
    '''
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    transformed_data = assembler.transform(data)
    if "label" in transformed_data.columns:
        transformed_data = transformed_data.withColumn("label", transformed_data.label.cast('boolean').cast('integer'))
    return transformed_data


def trainRandomForest(train):
    rf = RandomForestClassifier(labelCol='label', 
                            featuresCol='features',
                            maxDepth=8)
    return "RandomForest", rf.fit(train)

def trainLogisticRegression(train):
    reg = LogisticRegression(labelCol='label')
    return "LogisticRegression", reg.fit(train)


def enrichIncomingDataset(data, cleaned_base_data):
    data = data.drop(*["_c0", "partition"])

    # merging training data with cleaned data and calculate new columns
    data = mergeDatasetOnKey("1", data, cleaned_base_data)
    data = mergeDatasetOnKey("2", data, cleaned_base_data)
    data = calculateNumericFeatures(data)

    required_features = [
                    'jaccard_author',
                    'jaccard_title',
                    'jaccard_key',
                    'diff_year'
                   ]

    return prepareFeatures(required_features, data)

if __name__ == "__main__":

    spark = startSparkSession()

    df = spark.read.csv("data/db/db.csv", sep="!", header=True)
    # technically not needed, since we are doing a feature selection in `def enrichIncomingDataset``
    df = df.drop(*[
        "_c0",
        "paddress",
        "ppublisher",
        "pseries",
        "pbooktitlefull_id",
        "pjournalfull_id",
        "peditor",
        "pbooktitle_id",
        "partition"
    ])

    # training
    train_df = spark.read.csv("data/train.csv", header=True)
    prepared_data = enrichIncomingDataset(train_df, df)
    [train, test_with_label] = prepared_data.randomSplit([0.9, 0.1], seed=1000)

    name, model = trainRandomForest(train)
    print("Finished Training")

    rf_predicts = model.transform(test_with_label)

    # prediction
    test_df = spark.read.csv("data/test_hidden.csv", header=True)
    prepared_data = enrichIncomingDataset(test_df, df)

    predictions = model.transform(prepared_data)
    predictions = predictions.withColumn("prediction", fn.initcap(predictions.prediction.cast('Boolean').cast('String')))
    predictions.select("prediction").write.csv(path='submit/prediction.csv', mode='overwrite')


    # validation
    validation_df = spark.read.csv("data/validation_hidden.csv", header=True)
    prepared_data = enrichIncomingDataset(validation_df, df)

    validations = model.transform(prepared_data)
    validations = validations.withColumn("prediction", fn.initcap(validations.prediction.cast('Boolean').cast('String')))
    validations.select("prediction").write.csv(path='submit/validation.csv', mode='overwrite')


    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'label', metricName = 'accuracy')
    print(f'{name} Accuracy:', multi_evaluator.evaluate(rf_predicts))

    # TODO loading and saving, see comments below 👇

    # loading doesnt work that way: 
    #   py4j.protocol.Py4JJavaError: An error occurred while calling o227.load.
    #   java.lang.NoSuchMethodException: org.apache.spark.ml.classification.RandomForestClassificationModel.<init>(java.lang.String)
    # name, model = "RF", RandomForestClassifier.read().load("test_rf")

    # only save model when loading actually works
    # model.write().overwrite().save("test_rf")

    spark.stop()
    print("Finished model training. Stopped SparkSession. Good Bye.")
