import mlflow
import mlflow.pyfunc
import json
import pickle
import pandas as pd  # Importing pandas for DataFrame handling
from flask import Flask, request, render_template

from flask import Flask, request, render_template
from flask_cors import cross_origin
import os


import dagshub

app = Flask(__name__)

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
   raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/CU_Airfare_Prediction02.mlflow")

#mlflow.set_tracking_uri("https://dagshub.com/SHIVRAJSHINDE/CU_Airfare_Prediction03.mlflow")
#dagshub.init(repo_owner='SHIVRAJSHINDE', repo_name='CU_Airfare_Prediction03', mlflow=True)

# tracking_uri = "http://localhost:5000"
# mlflow.set_tracking_uri(tracking_uri)

def load_model_info() -> dict:
    """Load the model info from a JSON file."""
    if not os.path.exists('reports/experiment_info.json'):
        raise FileNotFoundError(f"Model info file not found: {'reports/experiment_info.json'}")

    with open('reports/experiment_info.json', 'r') as file:
        return json.load(file)

ModelName = "Lasso"
model_info = load_model_info()
model_uri = f"runs:/{model_info['run_id']}/{ModelName}"

# model_uri = 'runs:/b86d75382ed74105b6546b9c899fdc44/Lasso_model'

try:
    print(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    print(model)
except Exception as e:
    print("Error loading model:", e)

@app.route("/")
@cross_origin()

def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()

def predict():
    if request.method=="GET":
        return render_template(home.html)
    
    elif request.method=="POST":

        receiveData_Obj =  ReceiveData()
        
        df = receiveData_Obj.receive_data_from_ui_create_df(Airline = request.form.get('Airline'),
                                        Date_of_Journey = request.form.get('Date_of_Journey'),
                                        Source = request.form.get('Source'),
                                        Destination = request.form.get('Destination'),
                                        Dep_Time = request.form.get('Dep_Time'),
                                        Arrival_Time = request.form.get('Arrival_Time'),
                                        Duration = request.form.get('Duration'),
                                        Total_Stops = request.form.get('Total_Stops'))
        print(df)

        value = receiveData_Obj.execute_pipeline(df)
        prediction_value = model.predict(value)
        print("----------------------------------------------------")
        print(prediction_value)
        print("----------------------------------------------------")
        return render_template("prediction.html", prediction=prediction_value)

    return render_template("home.html")




class ReceiveData:
    def __init__(self):
        self.prediction_pipeline = PredictionPipeline()  # Instantiate PredictionPipeline class
        # self.data_cleaning = DataCleaningClass()  # Instantiate DataCleaningClass
        # self.encoding_scaling = EncodingAndScalingClass()  # Instantiate EncodingAndScalingClass

    def receive_data_from_ui_create_df(self, Airline: str, Date_of_Journey: pd.Timestamp, Source: str, 
                                       Destination: str, Dep_Time: str, Arrival_Time: pd.Timestamp, 
                                       Duration: str, Total_Stops: str) -> pd.DataFrame:
        """
        Converts received UI data into a DataFrame.
        """

        input_dict = {
            "Airline": [Airline],
            "Date_of_Journey": [Date_of_Journey],
            "Source": [Source],
            "Destination": [Destination],
            "Route": ["Route"],  # Placeholder for route info
            "Dep_Time": [Dep_Time],
            "Arrival_Time": [Arrival_Time],
            "Duration": [Duration],
            "Total_Stops": [Total_Stops],
            "Additional_Info": ["Additional_Info"],  # Placeholder for additional info
        }

        df = pd.DataFrame(input_dict)  # Creating DataFrame from input_dict
        return df

    def execute_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the prediction pipeline steps on the provided DataFrame.
        """
        print("Initial DataFrame:")

        # Apply prediction pipeline transformations
        df = self.prediction_pipeline.create_duration_column(df)
        print(df.T)
        print('After create_duration_column:')

        # Apply data cleaning process_Day_Month_Year
        df = self.prediction_pipeline.process_Day_Month_Year(df)
        print(df.T)
        print('After process_Day_Month_Year:')

        df = self.prediction_pipeline.Dept_Hours_Minutes(df)
        print(df.T)
        print('After Dept_Hours_Minutes:')

        df = self.prediction_pipeline.arrival_Hours_Minutes(df)
        print(df.T)
        print('After arrival_Hours_Minutes:')


        df = self.prediction_pipeline.process_duration(df)
        print(df.T)
        print('After duration_to_minutes:')


        df = self.prediction_pipeline.drop_unnecessary_columns(df)
        print(df.T)
        print('After drop_unnecessary_columns:')

        # Reorder columns as per final requirements
        df = self.prediction_pipeline.reorder_columns(df)
        print(df.T)
        print('After reorder_columns:')


        traformerObj = self.prediction_pipeline.loadtranformation()
        
        newData = traformerObj.transform(df)
        print(newData)
        return newData



class PredictionPipeline:

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary columns from the DataFrame."""
        df = df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info'], axis=1)
        print("Unnecessary columns dropped.")
        return df
        



    def create_duration_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the duration between 'Date_of_Journey' and 'Arrival_Time'.
        """
        
        df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
        df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])

        # Calculate the difference (duration) between the two datetime columns
        df['hoursMinutes'] = df['Arrival_Time']-df['Date_of_Journey'] 
        

        return df


    def process_Day_Month_Year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts and processes 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' columns."""
        print(df)

        df['Day'] = pd.to_datetime(df["Date_of_Journey"], format="%d-%m-%Y").dt.day
        df['Month'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.month
        df['Year'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.year
        
        print("Date and time columns processed.")
        return df

    def Dept_Hours_Minutes(self, df: pd.DataFrame) -> pd.DataFrame:
        def extract_hour(value):
            try:
                print('value')

                print(value)
                return pd.to_datetime(value, format="%d-%m-%Y %H:%M").hour
            except ValueError:
                return pd.to_datetime(value, format="%H:%M").hour
        
        df['Dept_Hour'] = df['Date_of_Journey'].apply(extract_hour)

        def extract_minute(value):
            try:
                return pd.to_datetime(value, format="%d-%m-%Y %H:%M").minute
            except ValueError:
                return pd.to_datetime(value, format="%H:%M").minute

        df['Dept_Minute'] = df['Date_of_Journey'].apply(extract_minute)

        return df

    def arrival_Hours_Minutes(self, df: pd.DataFrame) -> pd.DataFrame:
        def extract_hour(value):
            try:
                return pd.to_datetime(value, format="%d-%m-%Y %H:%M").hour
            except ValueError:
                return pd.to_datetime(value, format="%H:%M").hour
        
        df['Arr_Hour'] = df['Arrival_Time'].apply(extract_hour)

        def extract_minute(value):
            try:
                return pd.to_datetime(value, format="%d-%m-%Y %H:%M").minute
            except ValueError:
                return pd.to_datetime(value, format="%H:%M").minute

        df['Arr_Minute'] = df['Arrival_Time'].apply(extract_minute)

        return df


    def process_duration(self,df: pd.DataFrame) -> pd.DataFrame:
        """Converts 'Duration' column (format: '0 days HH:MM:SS') to total minutes."""
        
        # Function to convert '0 days HH:MM:SS' to total minutes
        def duration_to_minutes(duration_str: str) -> int:
            hours, minutes, _ = str(duration_str).split(' ')[-1].split(':')  # Extract hours and minutes
            return int(hours) * 60 + int(minutes)

        # Apply transformation directly to the 'Duration' column
        df['hoursMinutes'] = df['hoursMinutes'].apply(duration_to_minutes)

        return df



    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorders columns in the DataFrame for final output.
        """
        df = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day','Month','Year',
                 'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute', 'hoursMinutes']]
        return df

    def loadtranformation(self):
        with open('model/model_transform.pkl', 'rb') as file:
            transformer = pickle.load(file)
            return transformer












if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0", port=8080)
    app.run(debug=True, host="0.0.0.0")
