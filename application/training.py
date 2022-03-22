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


if __name__ == "__main__":

    spark = startSparkSession()

    df = spark.read.csv("data/db/db.csv", sep="!", header=True)
    df = df.drop(*["id", "partition"])

    train_df = spark.read.csv("data/train.csv", header=True)
    train_df = train_df.drop(*[
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

    # merging training data with cleaned data and calculate new columns
    train_df = mergeDatasetOnKey("1", train_df, df)
    train_df = mergeDatasetOnKey("2", train_df, df)
    train_df = calculateNumericFeatures(train_df)


    required_features = [
                    'jaccard_author',
                    'jaccard_title',
                    'jaccard_key',
                    'diff_year'
                   ]

    prepared_data = prepareFeatures(required_features, train_df)

    [train, test_with_label] = prepared_data.randomSplit([0.9, 0.1], seed=1000)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'label', metricName = 'accuracy')

    # loading doesnt work that way: 
    #   py4j.protocol.Py4JJavaError: An error occurred while calling o227.load.
    #   java.lang.NoSuchMethodException: org.apache.spark.ml.classification.RandomForestClassificationModel.<init>(java.lang.String)
    # name, model = "RF", RandomForestClassifier.read().load("test_rf")

    name, model = trainRandomForest(train)
    rf_predicts = model.transform(test_with_label)
    print(f'{name} Accuracy:', multi_evaluator.evaluate(rf_predicts))

    # only save model when loading actually works
    # model.write().overwrite().save("test_rf")

    spark.stop()
    print("Finished model training. Stopped SparkSession. Good Bye.")
