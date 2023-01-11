"""
Ce fichier est destiné à etre utilisé sur le Cloud, pour tester le passage à l'échelle.
Les N-Grams et le ChiQSelector ne sont pas utilisés afin de gagner du temps.

Ce fichier devra être renseigné si on on utilise la fonctionnalité "Envoyer un job" de GCloud.
Il devra etre uploadé au préalable dans un 'Bucket', qui est un stockage où l'on dépose nos datasets et scripts
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

# le path utilisé est un chemin personalisé à l'environnement Cloud
# Il indique la ressource présente dans un 'Bucket' GCloud nommé 'spark-twitter-bd'

path = "gs://spark-twitter-bd/training_noemoticon.csv"

schema = StructType([
    StructField("target", IntegerType(), True),
    StructField("id", StringType(), True),
    StructField("date", StringType(), True),
    StructField("query", StringType(), True),
    StructField("author", StringType(), True),
    StructField("tweet", StringType(), True)])


# récupération de la donnée et suppression des valeurs aberrantes

df = spark1.read.csv(path,
                     inferSchema=True,
                     header=False,
                     schema=schema)

df.dropna()

# séparation train et test, ratio 80 %/20 %
(train_set, test_set) = df.randomSplit([0.80, 0.20])

# création de la pipeline avec HashingTF et IDF
tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
hashtf = HashingTF(inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features")
label_stringIdx = StringIndexer(inputCol="target", outputCol="label")

# model et objet évaluateur
lr = LogisticRegression()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx, lr])

# début de l'entrainement et du chronometrage
st = time.time()
pipelineFit = pipeline.fit(train_set)

# relevé du temps d'entrainement
print('Training time:', time.time() - st)

# evaluation du modele
predictions = pipelineFit.transform(test_set)
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

# print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# relevé du temps total d'execution
print("Complete exec time:", time.time() - st)
