# Databricks notebook source
import mlflow

from mlflow.models.signature import infer_signature

# COMMAND ----------

## Lookup my experiment - just for giggles
experiments = \
  mlflow.search_experiments(filter_string = "name ilike '%/diamond_price_forecast.model_train'")

# Print details of each experiment
for experiment in experiments:
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print("------")

# COMMAND ----------

runs = \
  mlflow.search_runs(experiment_ids=[888983576038403]).sort_values(by='end_time', ascending=False)

display(runs)

# COMMAND ----------

best_run_id = mlflow.search_runs(experiment_ids=[888983576038403]) \
  .sort_values(by=['metrics.rmse'], ascending=True) \
  .head(1)["run_id"].values[0]

print(best_run_id)



# COMMAND ----------



target_run_id = best_run_id

model_name = "diamond-price-forecast-a1"
model_uri = f"runs:/{target_run_id}/{model_name}"

print(model_uri)

registered_model_name = 'diamond-price-forecast-a1-gamma'
registered_model = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

# COMMAND ----------

print(registered_model)
