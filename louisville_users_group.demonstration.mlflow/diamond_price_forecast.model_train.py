# Databricks notebook source
import mlflow
import logging

from pyspark.sql.functions import *

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from databricks.feature_store import FeatureStoreClient

from mlflow.models.signature import infer_signature

# COMMAND ----------

logging.getLogger("mlflow").setLevel(logging.FATAL)

# COMMAND ----------

feature_table_name = 'fortuno_sandbox.sandbox.diamond_feature'

fs = FeatureStoreClient()

diamonds_sdf = fs.read_table(name=feature_table_name)

display(diamonds_sdf.limit(5))

# COMMAND ----------

categories_string = ['cut', 'clarity', 'color']
categories_string_idx = [ cat + "_idx" for cat in categories_string]
categories_string_ohe = [cat + "_ohe" for cat in categories_string]
categories_numeric = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

label = "price"

## Input: ['cut', 'clarity', 'color']
## Output: ['cut_idx', 'clarity_idx', 'color_idx']
## We'll skip any unknown catagory
string_indexer = StringIndexer(
   inputCols = categories_string,
   outputCols = categories_string_idx,
   handleInvalid = "skip"
)

## Input: ['cut_idx', 'clarity_idx', 'color_idx']
## Output: ['cut_ohe', 'clarity_ohe', 'color_ohe']
## We'll skip any unknown catagory
ohe = OneHotEncoder()
ohe.setInputCols(categories_string_idx)
ohe.setOutputCols(categories_string_ohe)

assembler_inputs = categories_string_ohe + categories_numeric

## ['cut_ohe', 'clarity_ohe', 'color_ohe', 'carat', 'depth', 'table', 'price', 'x', 'y', 'z']
print(assembler_inputs)

## inputCols: 'size_ohe', 'day_ohe', 'total_bill', 'sex', 'smoker', 'time']
## outputCols: 'features' <- this is the column that will have the merged vectorized feature set
vec_assembler = VectorAssembler(
   inputCols=assembler_inputs,
   outputCol="features"
)

## We're going to be creating a regression model using a GBT regressor algorithm 
gbt = GBTRegressor(
   featuresCol="features",
   labelCol=label,
   maxIter=5
)

## Finally, we'll test the model to ensure its quality
evaluator = RegressionEvaluator(
   labelCol=label,
   predictionCol="prediction",
   metricName="rmse"
)


# COMMAND ----------

## Create my training and testing datasets by splitting the diamonds dataframe
## into two dataframes.
train_sdf, test_sdf = diamonds_sdf.randomSplit([0.8, 0.2], seed=42)

print(f"Number of rows test set: {test_sdf.count()}")
print(f"Number of rows train set: {train_sdf.count()}")
print(f"Sum of count rows of train and test set: {train_sdf.count() + test_sdf.count()}")
print(f"Total number of rows of initial dataframe: {diamonds_sdf.count()}")

# COMMAND ----------

model_name = "diamond-price-forecast-a1"

## `run_name` is the name associated with this run
## You can also include the ID for the associated experiment via
## the `experiment_id` parameter.
with mlflow.start_run(run_name="train-diamond-price-forecast-a1") as run:
    # define pipeline stages according to model
    stages = [string_indexer, ohe, vec_assembler, gbt]
    
    # set pipeline
    pipeline = Pipeline(stages=stages)
    
    # fit pipeline to train set
    model = pipeline.fit(train_sdf)
    
    # manually log parameter to mlflow
    mlflow.log_param("maxIter", 5)
    
    # predict test set
    pred_sdf = model.transform(test_sdf)
    
    # evaluate prediction
    rmse = evaluator.evaluate(pred_sdf)
    
    ## manually log metric to mlflow
    mlflow.log_metric("rmse", rmse)

    input_example = train_sdf.head(3)
    #signature = infer_signature(train_sdf, pred_sdf.select("prediction").limit(10000).toPandas())
    signature = infer_signature(train_sdf, pred_sdf.select("prediction"))
    mlflow.spark.log_model(
        model,
        model_name,
        signature=signature,
        input_example=input_example
    )

    ## Print out the identifier associated with this run
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id


# COMMAND ----------

display(train_sdf.schema)
display(pred_sdf.schema)

# COMMAND ----------

print(f"Experiment: {experiment_id}")
print(f"Run: {run_id}")
