import os
import sys
import pandas as pd
import yaml
from src.Utils.exception import CustomException


class RemoveOutlier:
    def read_csv(self,cleaned_dir,filename):
        # source_path = r"D:/Data/01_AirlineData/Airline.csv"  # Raw string

        # Read the CSV file and return the DataFrame
        try:
            file_path = os.path.join(cleaned_dir,filename)
            print(file_path)

            df = pd.read_csv(file_path)
            
            return df
        
        except Exception as e:
            print(CustomException(e,sys))
            return None


    def load_yaml(self, yaml_path):
        yaml_path = os.path.join(yaml_path)
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            return data.get('airlineQuartile', {})

    def remove_outliers(self, df, airlineQuartile):
        cleaned_df = pd.DataFrame(columns=df.columns)
        
        for airline, quartiles in airlineQuartile.items():
            airDataSet = df[df['Airline'] == airline]
            q1 = airDataSet['Price'].quantile(quartiles[0])
            q3 = airDataSet['Price'].quantile(quartiles[1])
            IQR = q3 - q1
            lowerLimit = q1 - IQR * 1.5
            upperLimit = q3 + IQR * 1.5
            
            lowerLimitIndex = airDataSet[airDataSet['Price'] <= lowerLimit].index
            upperLimitIndex = airDataSet[airDataSet['Price'] >= upperLimit].index
            
            if airDataSet.shape[0] > 5:
                airDataSet = airDataSet.drop(lowerLimitIndex).drop(upperLimitIndex)
            
            cleaned_df = pd.concat([cleaned_df, airDataSet], axis=0)
        
        return cleaned_df

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


if __name__ == "__main__":

    cleaned_dir = "./Data/02_CleanedData/"
    filename = "./CleanedData.csv"

    yaml_path = "./constants.yaml"

    noOutlier_Dir = "./Data/03_noOutlierData/"
    noOutlier_File = "noOutlierDataFile.csv"


    removerObj = RemoveOutlier()

    df = removerObj.read_csv(cleaned_dir,filename)
    print(df)

    airlineQuartile = removerObj.load_yaml(yaml_path)
    print(airlineQuartile)

    cleaned_df = removerObj.remove_outliers(df, airlineQuartile)
    removerObj.save_file(df,noOutlier_Dir, noOutlier_File)
