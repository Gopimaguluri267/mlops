
# MLOps Repository

This repository contains the necessary components for implementing MLOps (Machine Learning Operations) practices as part of an MLOps (MSDS-603) course at University of San Francisco. It provides a structured environment for model training, experiment tracking, and model registry using MLFlow.

## Repository Structure

```/mlops```
* This is the main directory

```/mlops/data```
* This directory contains all the data used for training, validation and testing purposes. It contains both the raw and processed data.

```/mlops/labs```
* Contains lab exercises and practical implementations.

```/mlops/labs/mlruns```
* Directory used by MLflow to store experiment tracking data, including metrics, parameters, and artifacts from various runs.

```/mlops/models```
* This directory has all the trained models, model versions.

```/mlops/notebooks```
* This directory contains the collection of Jupyter notebooks for exploratory data analysis, model prototyping, and result visualization.

```/environment.yaml```
* Conda environment file defining all the dependencies required to run the code in this repository.

```/requirements.txt```
* Text file listing all Python package dependencies needed. This is useful for pip-based environment setup.


## Getting Started

1. Clone this repository

2. Set up the environments.
    - Using conda: conda env create -f environment.yaml
    - Using pip: pip install -r requirements.txt


## MLflow Experiment Tracking

* This repository uses MLflow for experiment tracking. To view the MLflow UI locally:

    - Navigate to the repository root

    - Run ```mlflow ui --backend-store-uri sqlite:///mlops/labs/mlflow.db```
    
    - Open a browser and go to http://localhost:5000
