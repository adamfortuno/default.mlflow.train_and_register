# default.mlflow.train_and_register

The notebooks in this repository are meant to be to be run on Databricks' Data Intelligence Platform. They will create a model that predicts the price of a dimond based on its characteristics. We train the model on a diamond dataset provided by Seaborn. We commit the model to the MLFlow Tracking Server. We register one of the logged models to the MLFlow Model Registry. We load the registered model and use it to generate a prediction.

## Intructions

1. Create a Git Folder for the repository [1].
2. If one does not exist, create a single node (Spark) cluster using the latest generally available (non-beta) LTS ML runtime. The latest ML runtime as of the writing of this doc is 14.3 LTS ML. You do not need the GPU capable runtimes.
3. Ensure the cluster is running (started).
4. Open a notebook, select your cluster and run the cells in the notebook [2]. You will process the notebooks in the following sequence:

diamond_price_forecast.feature_preparation.py
diamond_price_forecast.model_train.py
diamond_price_forecast.model_registration.py
diamond_price_forecast.model_inference.py

Reference the following documents for help executing the preceeding steps:

1. [Set up Databricks Git Folders](https://learn.microsoft.com/en-us/azure/databricks/repos/repos-setup)
2. [Run Databricks notebooks](https://learn.microsoft.com/en-us/azure/databricks/notebooks/run-notebook)

The purpose of this exercise is to understand the code in the notebook. Review each cell. Understand what the statements in the cell do. Try to manipulate the project based on your understand to add some new feature.

## Documentation

The purpose of this project is to demonstrate the following:

* Create a feature table in Databricks feature store
* Train an ML model and log it with an MLFlow Tracking Server
* Register an ML model with MLFlow Model Registry
* Create predictions from a registered model

You'll be working with the diamonds dataset from Seaborn.

This project consists of four notebooks:

* **diamond_price_forecast.feature_preparation.py**, download the diamond dataset from Seaborn, process it, and save it as a feature table in Databricks [Feature Store](https://www.databricks.com/product/feature-store).
* **diamond_price_forecast.model_train.py**, train a [Gradient Boost Tree](https://www.geeksforgeeks.org/ml-gradient-boosting/) mode using Spark ML's `GBTRegressor` class. The code in this notebook loads the diamond feature table, creates training/testing datasets, trains a model, and logs the model.
* **diamond_price_forecast.model_registration.py**, this notebook registers the GBT model we trained and logged in MLFlow Model Registry.
* **diamond_price_forecast.model_inference.py**, this notebook loads the registered model from the MLFlow model registry and creates a prediction.
