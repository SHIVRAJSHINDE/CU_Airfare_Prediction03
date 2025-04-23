import os
import pandas as pd
from pathlib import Path

class DataIngestionClass:

    def read_csv(OriginalDir,OriginalFile):
        # source_path = r"D:/Data/01_AirlineData/Airline.csv"  # Raw string

        # Read the CSV file and return the DataFrame
        try:

            file_path = os.path.join(OriginalDir,OriginalFile)
            print(file_path)
            df = pd.read_csv(file_path)
            
            return df

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

    def save_file(df, directory, filename):
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(directory):
            
            os.makedirs(directory)
            print(f"Directory '{directory}' was created.")

        else:

            print(f"Directory '{directory}' already exists.")

        # Construct the file path
        
        file_path = os.path.join(directory, filename)
        print(file_path)
        
        # Save the DataFrame to the file
        df.to_csv(file_path, index=False)  # index=False to avoid writing row indices
        print(df)
        print("----------------------------------------------------")
        print(f"File has been saved to {file_path}")
        print("----------------------------------------------------")
        
# This block will only execute if this script is run directly
if __name__ == "__main__":
    import os
    # source_path = os.path.join("D:", "\\Training", "04DataSets", "01_AirlineData", "Airline.csv")

    # print(source_path)
    OriginalDir = "./OriginalFolder/"
    OriginalFile = "Airline.csv"
    # source_path = "D:\\DataSets\\01_AirlineData\\Airline.csv"  # Use Pathlib to build the path
    # df = pd.read_csv(r"C:/Users/SHIVRAJ SHINDE/JupiterWorking/XL_ML/Z_DataSets/01_AirlineData/Airline.csv")
    # print(df)

    directory = "./Data/01_RawData/"
    filename = "Airline.csv"

    df = DataIngestionClass.read_csv(OriginalDir,OriginalFile)  # Read the CSV file
    
    DataIngestionClass.save_file(df, directory, filename)  # Save the DataFrame to the destination
