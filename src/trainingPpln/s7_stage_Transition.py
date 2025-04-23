import os
import json
import sys
import mlflow
import time
import dagshub
from src.Utils.exception import CustomException

class ModelManager:
    """Class to manage model saving, loading, and registration with MLflow."""

    def __init__(self, model_name: str, info_path: str):
        
        self.model_name = model_name
        self.info_path = info_path
        self.client = mlflow.tracking.MlflowClient()


        # dagshub_token = os.getenv("DAGSHUB_PAT")
        # if not dagshub_token:
        #     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/AirlineFare_EndToEnd.mlflow")        


        # mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/CU_Airfare_Prediction01.mlflow")
        # dagshub.init(repo_owner='SHIVRAJSHINDE', repo_name='CU_Airfare_Prediction01', mlflow=True)

    
        # self.tracking_uri = "http://localhost:5000"
        # mlflow.set_tracking_uri(self.tracking_uri)


    def load_model_info(self) -> dict:
        """Load the model info from a JSON file."""
        try:
            if not os.path.exists(self.info_path):
                raise FileNotFoundError(f"Model info file not found: {self.info_path}")

            with open(self.info_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(CustomException(e,sys))


    def register_model(self):
        try:
            model_info = self.load_model_info()
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            

            print('-------------------------------------------------------')
            print("Registering model with URI:", model_uri)
            print('-------------------------------------------------------')

            # Register the model and extract version number
            model_version = mlflow.register_model(model_uri, self.model_name).version

            print('-------------------------------------------------------')
            print(f"Registered model '{self.model_name}' as version {model_version}")
            print('-------------------------------------------------------')

            # Ensure registration is complete before transitioning
            time.sleep(5)  # Add delay to ensure visibility in MLflow

            # Transition the model version
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=True
            )

            print(f"Model '{self.model_name}' version {model_version} moved.")

        except Exception as e:
            print(CustomException(e, sys))



if __name__ == '__main__':
    mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/CU_Airfare_Prediction02.mlflow")
    dagshub.init(repo_owner='SHIVRAJSHINDE', repo_name='CU_Airfare_Prediction02', mlflow=True)

    model_name="Lasso"
    info_path='reports/experiment_info.json'

    model_manager = ModelManager(model_name, info_path)
    
    # Load model info and register
    model_manager.register_model()
