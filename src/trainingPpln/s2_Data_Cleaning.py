import pandas as pd
import os
import sys
from src.Utils.exception import CustomException


class DataCleaningClass:
    '''def read_csv(self, raw_file_path: str) -> pd.DataFrame:
        try:
            print(raw_file_path)
            """Reads a CSV file and returns a DataFrame."""
            df = pd.read_csv(raw_file_path)
            return df
        except Exception as e:
            print(CustomException(e,sys))'''

    def read_csv_as_dataframe(self,raw_file_dir,raw_file) -> pd.DataFrame:

        # Combine the folder and file path into a full path
        # full_path = os.path.join(folder, file)
        
        file_path = os.path.join(raw_file_dir,raw_file)

        print(file_path)
        # Read the CSV file into a DataFrame
        try:

            df = pd.read_csv(file_path)
            return df
        
        except Exception as e:
            print(CustomException(e,sys))
            return None
        except Exception as e:
            print(CustomException(e,sys))
            return None



    def clean_total_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in 'Total_Stops' column with the mode."""
        try:

            mode_of_total_stops = df['Total_Stops'].mode()[0]
            df['Total_Stops'].fillna(mode_of_total_stops, inplace=True)
            print("Missing values in 'Total_Stops' filled with mode.")
            return df
        except Exception as e:
            print(CustomException(e,sys))

    def clean_airline_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the 'Airline' column by replacing specific values."""
        try:
            df['Airline'].replace("Multiple carriers Premium economy", "Multiple carriers", inplace=True)
            df['Airline'].replace("Jet Airways Business", "Jet Airways", inplace=True)
            df['Airline'].replace("Vistara Premium economy", "Vistara", inplace=True)
            print("Airline names cleaned.")
            return df
        except Exception as e:
            print(CustomException(e,sys))


    def clean_destination_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces 'New Delhi' with 'Delhi' in the 'Destination' column."""
        try:
            df['Destination'].replace(to_replace="New Delhi", value="Delhi", inplace=True)
            print("'New Delhi' replaced with 'Delhi' in 'Destination'.")
            return df
        except Exception as e:
            print(CustomException(e,sys))


    def create_duration_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a new column 'hoursMinutes' to represent flight duration in minutes."""
        try:
            df["hoursMinutes"] = 0
            for i in df.index:
                if " " in df.loc[i, 'Duration']:
                    column1 = df.loc[i, 'Duration'].split(" ")[0]
                    column2 = df.loc[i, 'Duration'].split(" ")[1]

                    if "h" in column1:
                        column1 = (int(column1.replace("h", "")) * 60)
                    elif "m" in column1:
                        column1 = (int(column1.replace("m", "")))

                    if "h" in column2:
                        column2 = (int(column2.replace("h", "")) * 60)
                    elif "m" in column2:
                        column2 = (int(column2.replace("m", "")))

                    df.loc[i, 'hoursMinutes'] = column1 + column2
                else:
                    column1 = df.loc[i, 'Duration']

                    if "h" in column1:
                        column1 = (int(column1.replace("h", "")) * 60)
                    elif "m" in column1:
                        column1 = (int(column1.replace("m", "")))

                    df.loc[i, 'hoursMinutes'] = column1

            print("'hoursMinutes' column created from 'Duration'.")
            return df
        except Exception as e:
            print(CustomException(e,sys))


    def process_date_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts and processes 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' columns."""
        try:
            # Extract Day, Month, Year from Date_of_Journey
            df['Day'] = pd.to_datetime(df["Date_of_Journey"], format="%d-%m-%Y").dt.day
            df['Month'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.month
            df['Year'] = pd.to_datetime(df['Date_of_Journey'], format="%d-%m-%Y").dt.year

            # Function to extract time safely
            def extract_time(value):
                try:
                    return pd.to_datetime(value, format="%H:%M").time()  # Only time
                except ValueError:
                    return pd.to_datetime(value).time()  # Full datetime case
            
            # Apply function to extract hours and minutes
            df['Dept_Hour'] = df['Dep_Time'].apply(lambda x: extract_time(x).hour)
            df['Dept_Minute'] = df['Dep_Time'].apply(lambda x: extract_time(x).minute)

            df['Arr_Hour'] = df['Arrival_Time'].apply(lambda x: extract_time(x).hour)
            df['Arr_Minute'] = df['Arrival_Time'].apply(lambda x: extract_time(x).minute)

            print("Date and time columns processed successfully.")
            return df

        except Exception as e:
            print(CustomException(e, sys))


    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary columns from the DataFrame."""
        try:
            df = df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info'], axis=1)
            print("Unnecessary columns dropped.")
            return df
        except Exception as e:
            print(CustomException(e,sys))


    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders columns for the final DataFrame."""
        try:
            df = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day', 'Month', 'Year', 
                    'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute', 'hoursMinutes', 'Price']]
            print("Columns reordered.")
            return df
        except Exception as e:
            print(CustomException(e,sys))


    def save_file(self, df: pd.DataFrame, directory: str, filename: str) -> None:
        """Saves the DataFrame to a CSV file in the specified directory."""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory '{directory}' was created.")
                file_path = os.path.join(directory, filename)
                df.to_csv(file_path, index=False)
                print(f"File has been saved to {file_path}")

            else:
                print(f"Directory '{directory}' already exists.")

                file_path = os.path.join(directory, filename)
                df.to_csv(file_path, index=False)
                print(f"File has been saved to {file_path}")

        except Exception as e:
            print(CustomException(e,sys))


# Example usage:
if __name__ == "__main__":
    raw_file_dir = "./Data/01_RawData/"
    raw_file = "Airline.csv"

    directory = "./Data/02_CleanedData/"
    filename = "CleanedData.csv"

    # Create an instance of DataCleaningClass
    data_cleaning_obj = DataCleaningClass()
    df = data_cleaning_obj.read_csv_as_dataframe(raw_file_dir,raw_file)
    # Apply the cleaning functions
    df = data_cleaning_obj.clean_total_stops(df)
    df = data_cleaning_obj.clean_airline_column(df)
    df = data_cleaning_obj.clean_destination_column(df)
    df = data_cleaning_obj.create_duration_column(df)
    df = data_cleaning_obj.process_date_time_columns(df)
    df = data_cleaning_obj.drop_unnecessary_columns(df)
    df = data_cleaning_obj.reorder_columns(df)
        # Save the cleaned data
    data_cleaning_obj.save_file(df,directory,filename)
