"""
This file is intended for use in the cloud, to test scalability.

N-Grams and ChiQSelector are not used to save time.

This file must be completed if you use GCloud's "Submit a Job" feature.
It must first be uploaded to a "Bucket," which is a storage location where you store your datasets and scripts.
"""

import findspark

findspark.init()

import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark1 = SparkSession.builder \
    .master("local[*]") \
    .appName("CloudJob") \
    .getOrCreate()

# The path used is a custom path for the Cloud environment
# It indicates the resource present in a GCloud 'Bucket' named 'spark-twitter-bd'

path = "gs://spark-twitter-bd/training_noemoticon.csv"

schema = StructType([
    StructField("target", IntegerType(), True),
    StructField("id", StringType(), True),
    StructField("date", StringType(), True),
    StructField("query", StringType(), True),
    StructField("author", StringType(), True),
    StructField("tweet", StringType(), True)])


# data recovery and removal of outliers

df = spark1.read.csv(path,
                     inferSchema=True,
                     header=False,
                     schema=schema)

df.dropna()

# train and test separation, 80%/20% ratio
(train_set, test_set) = df.randomSplit([0.80, 0.20])

# creating the pipeline with HashingTF and IDF
tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
hashtf = HashingTF(inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features")
label_stringIdx = StringIndexer(inputCol="target", outputCol="label")

# model and evaluator object
lr = LogisticRegression()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx, lr])

# start of training and timing
st = time.time()
pipelineFit = pipeline.fit(train_set)

# training time record
print('Training time:', time.time() - st)

# model evaluation
predictions = pipelineFit.transform(test_set)
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

# print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# total execution time statement
print("Complete exec time:", time.time() - st)
