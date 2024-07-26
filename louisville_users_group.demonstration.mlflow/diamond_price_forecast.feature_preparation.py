# Databricks notebook source
from pyspark.sql.functions import *

import seaborn as sns

from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# COMMAND ----------

## Load the Diamonds sample dataset from Seaborn
diamonds_df = sns.load_dataset('diamonds')
diamonds_sdf = spark.createDataFrame(diamonds_df)

display(diamonds_sdf.limit(5))

# COMMAND ----------

## Add an incrementing integer column; will be our table's primary key
diamonds_sdf = diamonds_sdf.withColumn("id", monotonically_increasing_id())

## Drop duplicate rows from the dataset
diamonds_sdf = diamonds_sdf.dropDuplicates()

display(diamonds_sdf.limit(5))

# COMMAND ----------

feature_table_name = 'fortuno_sandbox.sandbox.diamond_feature'

fs = FeatureStoreClient()

fs.create_table(
   name = feature_table_name, 
   primary_keys = 'id',
   schema = diamonds_sdf.schema,
   df = diamonds_sdf,
   description ='Diamond features'
)

