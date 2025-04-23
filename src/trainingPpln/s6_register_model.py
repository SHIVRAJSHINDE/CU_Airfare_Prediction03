import os
import json

import mlflow
import time
import dagshub
from mlflow.tracking import MlflowClient


class ModelManagerRegister:
    """Class to manage model saving, loading, and registration with MLflow."""

    def __init__(self, model_name: str, info_path: str):
        self.model_name = model_name
        self.info_path = info_path
        #self.client = mlflow.tracking.MlflowClient()
        self.client = MlflowClient()

        # dagshub_token = os.getenv("DAGSHUB_PAT")
        # if not dagshub_token:
        #    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/AirlineFare_EndToEnd.mlflow")        


        mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/CU_Airfare_Prediction02.mlflow")
        dagshub.init(repo_owner='SHIVRAJSHINDE', repo_name='CU_Airfare_Prediction02', mlflow=True)
        
        # self.tracking_uri = "http://localhost:5000"
        # mlflow.set_tracking_uri(self.tracking_uri)

    def save_model_info(self, run_id: str, model_path: str) -> None:
        """Save the model run ID and path to a JSON file."""
        os.makedirs(os.path.dirname(self.info_path), exist_ok=True)  # Ensure the directory exists
        model_info = {'run_id': run_id, 'model_path': model_path}
        
        with open(self.info_path, 'w') as file:
            json.dump(model_info, file, indent=4)

    def load_model_info(self) -> dict:
        """Load the model info from a JSON file."""
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"Model info file not found: {self.info_path}")

        with open(self.info_path, 'r') as file:
            return json.load(file)

    def register_model(self):
        """Register the model with the MLflow Model Registry."""
        model_info = self.load_model_info()

        print("----------------------------------------------------------------")
        print(model_info)
        print("----------------------------------------------------------------")
 
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        # model= "Lasso"    

        # model_uri = f"runs:/{model_info['run_id']}/{model}"

        print("----------------------------------------------------------------")
        print(model_uri)
        print("----------------------------------------------------------------")
        # Register the model
 
        model_version = mlflow.register_model(model_uri, self.model_name)
 
        print("----------------------------------------------------------------")
        print(model_version)
        print("----------------------------------------------------------------")
        print("model_name,model_version.version")
        print(self.model_name,model_version.version)
        # Transition the model to "Staging" stage
        

if __name__ == '__main__':
    model_name="Lasso"
    info_path='reports/experiment_info.json'

    model_manager = ModelManagerRegister(model_name, info_path)
    
    # Load model info and register
    model_manager.register_model()












