import yaml
import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import json
import os

import mlflow
import dagshub



class ModelTrainerClass:

    def __init__(self):
        """Initialize the trainer  with paths and load necessary data."""
        # self.X_train_path = X_train_path
        # self.y_train_path = y_train_path
        # self.params_path = params_path
        pass
    
    def load_X_train(self,X_train_Dir,X_train_File):
        """Load X_train from the provided file path."""
        file_path = os.path.join(X_train_Dir,X_train_File)
        
        X_train = pd.read_csv(file_path)

        X_train = np.array(X_train)
        
        return X_train

    def load_y_train(self,y_train_path,y_train_File):
        """Load y_train from the provided file path."""
        file_path = os.path.join(y_train_path,y_train_File)
        
        y_train = pd.read_csv(file_path)

        y_train = np.array(y_train).ravel()  # Ensure it's a flat array
        return y_train

    def load_params(self,params_path):
        """Load parameters from the YAML file."""
        file_path = os.path.join(params_path)
        with open(file_path, "r") as file:
            self.modelWithParams = yaml.safe_load(file)
            self.modelWithParams = self.modelWithParams['model']
 
        return self.modelWithParams
    
    def get_Model_class(self,model):
        model_class = globals()[model.split('.')[-1]]  # Get the class from the model name
        model = model_class()  # Instantiate the model
        return model

    def train_model(self, model, param_grid,X_train,y_train):
        """Train the model"""
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5)
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        best_model.fit(X_train, y_train)

        predicted_y = best_model.predict(X_train)

        return best_model, best_params, predicted_y

    def calculate_metrics(self, actual_X, actual_y, predicted_y):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(actual_y, predicted_y)
        mae = mean_absolute_error(actual_y, predicted_y)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_y, predicted_y)

        # Adjusted R²
        n = len(actual_y)
        p = actual_X.shape[1]  # Use actual_X to get the number of features
        aR2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return mse, mae, rmse, r2, aR2

    def get_Model_Name(self, model):
        """Get the model name."""
        model_name = str(model.__class__.__name__)
        print(model_name)
        return model_name




class MLflowLoggerClass:

    def __init__(self):
        """Initialize MLflowLogger with the tracking URI."""
        # dagshub_token = os.getenv("DAGSHUB_PAT")
        # if not dagshub_token:
        #    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/AirlineFare_EndToEnd.mlflow")        


        #mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/AirlineFare_EndToEnd.mlflow")

        mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/CU_Airfare_Prediction02.mlflow")
        dagshub.init(repo_owner='SHIVRAJSHINDE', repo_name='CU_Airfare_Prediction02', mlflow=True)

        # self.tracking_uri = "http://localhost:5000"
        # mlflow.set_tracking_uri(self.tracking_uri)

    def save_model_info(self, run_id: str, model_path: str, file_path: str) -> None:
        """Save the model run ID and path to a JSON file, ensuring the directory exists."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create the folder if it doesn't exist
        
        model_info = {'run_id': run_id, 'model_path': model_path}
        
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)

    def log_results(self, model_name, best_model, best_params, mse, mae, rmse, r2, aR2):
        """Log results to MLflow."""
        with mlflow.start_run(run_name=model_name) as run:
            # Log metrics to MLflow
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("Adjusted R2", aR2)

            # Log best parameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"Best {param_name}", param_value)

            # Log the model
            #mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            mlflow.sklearn.log_model(best_model, f"{model_name}")

            print("----------------------------------------------------------------")
            print(run.info.run_id)
            print("----------------------------------------------------------------")

            self.save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            # Print the metrics
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print(f"R2: {r2}")
            print(f"Adjusted R2: {aR2}")


# Main script execution
if __name__ == "__main__":

    X_train_Dir = "./Data/04_encoded_Data/"
    X_train_File = "X_train.csv"

    y_train_path = "./Data/04_encoded_Data/"
    y_train_File = "y_train.csv"

    params_path = "./modelsParams.yaml"

    # Initialize the ModelTrainerObj
    ModelTrainerObj = ModelTrainerClass()

    # Load data and params
    X_train = ModelTrainerObj.load_X_train(X_train_Dir,X_train_File)
    y_train = ModelTrainerObj.load_y_train(y_train_path,y_train_File)

    modelWithParams = ModelTrainerObj.load_params(params_path)
  
    #print("----------------------------------------------------")
    #print(modelParams['model'])
    for value in modelWithParams.values():
        model = ModelTrainerObj.get_Model_class(value['model'])
        params_Grid = value['param']

        print(model)
        print(params_Grid)

        # Train the model
        best_model, best_params, predicted_y = ModelTrainerObj.train_model(model, params_Grid,X_train,y_train)

        # Calculate metrics // performance of the model
        mse, mae, rmse, r2, aR2 = ModelTrainerObj.calculate_metrics(X_train, y_train, predicted_y)

        # Log the model name
        model_name = ModelTrainerObj.get_Model_Name(model)
        print(f"Model Name: {model_name}")

        # Initialize MLflowLogger and log results
        MLflowLoggerObj = MLflowLoggerClass()  # Replace with your URI
        MLflowLoggerObj.log_results(model_name, best_model, best_params, mse, mae, rmse, r2, aR2)
