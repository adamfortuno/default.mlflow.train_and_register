# Databricks notebook source
import mlflow

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

## Load the model from the model registry
registered_model_path = 'dbfs:/databricks/mlflow-tracking/888983576038403/d6f7f6351b504f43889f008f599ad437/artifacts/diamond-price-forecast-a1'
model = mlflow.pyfunc.load_model(registered_model_path)

# COMMAND ----------

feature_table_name = 'fortuno_sandbox.sandbox.diamond_feature'

fs = FeatureStoreClient()

diamonds_sdf = fs.read_table(name=feature_table_name)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

sample_data = [(0.61, 'Ideal', 'D', 'SI1', 22.6, 56.0, 3811, 6.1, 6.08, 3.84, 1)]
sample_data_schema = StructType([
    StructField('carat', DoubleType(), True),
    StructField('cut', StringType(), True),
    StructField('color', StringType(), True),
    StructField('clarity', StringType(), True),
    StructField('depth', DoubleType(), True),
    StructField('table', DoubleType(), True),
    StructField('price', LongType(), True),
    StructField('x', DoubleType(), True),
    StructField('y', DoubleType(), True),
    StructField('z', DoubleType(), True),
    StructField('id', LongType(), False)
])
sample_data_sdf = spark.createDataFrame(sample_data, sample_data_schema)

model.predict(sample_data_sdf.toPandas()) 

