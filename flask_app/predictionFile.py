import pickle
import pandas as pd  # Importing pandas for DataFrame handling
from flask import Flask, request, render_template

# Import necessary classes from the pipeline module
# from src.Pipeline.s2_Data_Cleaning import DataCleaningClass
# from src.Pipeline.s4_Encoding import EncodingAndScalingClass 
# Class to handle data received from UI

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