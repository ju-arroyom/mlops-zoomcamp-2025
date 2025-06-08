import os
import pickle
from pathlib import Path

import mlflow

from sklearn.linear_model import LinearRegression

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("nyc-taxi-hw3")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def train_model(X, y, dv):
    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X, y) 
        print(f"Model Intercept: {model.intercept_:.2f}")
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="LinealRegression")
    return run.info.run_id
