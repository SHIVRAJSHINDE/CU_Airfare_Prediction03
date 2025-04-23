import os
import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import make_pipeline
from src.Utils.Utils import load_yaml,get_class,get_class_Scaler
from src.Utils.exception import CustomException


class EncodingAndScalingClass:
    def __init__(self):
        self.load_yaml = load_yaml
        self.get_class = get_class
        self.data = self.load_yaml(yaml_path="constants.yaml")
        
        # TrainTest Split
        self.testSize = self.data['trainTestSplit']['testSize']
        self.randomState = self.data['trainTestSplit']['randomState']

        # MinMax and Standard Scalaer Variable
        self.scaling = self.data['scaling']['scalingFeature']
        self.scalling = get_class_Scaler(self.scaling)


    def read_csv(self,noOutlier_Dir,noOutlier_File):
        # source_path = r"D:/Data/01_AirlineData/Airline.csv"  # Raw string

        # Read the CSV file and return the DataFrame
        try:
            file_path = os.path.join(noOutlier_Dir,noOutlier_File)
            print(file_path)
            df = pd.read_csv(file_path)
            
            return df
        except Exception as e:
            print(CustomException(e,sys))
            return None

    def split_df_to_X_y(self, df):
        try:
            X = df.drop(columns=['Price'])
            y = df['Price']
            return X, y
        except Exception as e:
            print(CustomException(e,sys))
            return None


    def train_test_split(self, X, y):
        try:
            return train_test_split(X, y, test_size=self.testSize, random_state=self.randomState)
        except Exception as e:
            print(CustomException(e,sys))
            return None


    def encoding_and_scaling(self):
        try:
            trf1 = ColumnTransformer([
                ('OneHot', OneHotEncoder(drop='first', handle_unknown='ignore'), [0, 1, 2])
            ], remainder='passthrough')

            trf2 = ColumnTransformer([
                ('Ordinal', OrdinalEncoder(categories=[['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']]), [16])
            ], remainder='passthrough')

            trf3 = ColumnTransformer([
                ('scale', self.scalling, slice(25))  # Scale first 25 columns
            ])
            
            return make_pipeline(trf1, trf2, trf3)
        except Exception as e:
            print(CustomException(e,sys))
            return None

    
    def fit_transform_X_train(self, pipe, X_train):
        try:
            return pd.DataFrame(pipe.fit_transform(X_train))
        except Exception as e:
            print(CustomException(e,sys))
            return None


    def transform_X_test(self, pipe, X_test):
        try:
            return pd.DataFrame(pipe.transform(X_test))
        except Exception as e:
            print(CustomException(e,sys))
            return None


    def makeTransformerFile(self, pipe):
        try:
            with open('model/model_transform.pkl', 'wb') as file:
                pickle.dump(pipe, file)
        except Exception as e:
            print(CustomException(e,sys))
            return None



    def save_dataframe(self, df, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            df.to_csv(file_path, index=False)

        except Exception as e:
            print(CustomException(e,sys))
            return None


if __name__ == "__main__":
    # file_path = "Data/03_noOutlierData/noOutlierDataFile.csv"
    noOutlier_Dir = "./Data/03_noOutlierData/"
    noOutlier_File = "noOutlierDataFile.csv"

    obj = EncodingAndScalingClass()
    
    df = obj.read_csv(noOutlier_Dir,noOutlier_File)
    X, y = obj.split_df_to_X_y(df)

    X_train, X_test, y_train, y_test = obj.train_test_split(X, y)
    
    pipeObj = obj.encoding_and_scaling()
    
    X_train_transformed = obj.fit_transform_X_train(pipeObj, X_train)
    X_test_transformed = obj.transform_X_test(pipeObj, X_test)
    
    obj.makeTransformerFile(pipeObj)


    obj.save_dataframe(X_train_transformed, "./Data/04_encoded_Data/X_train.csv")
    obj.save_dataframe(X_test_transformed, "./Data/04_encoded_Data/X_test.csv")
    obj.save_dataframe(y_train, "./Data/04_encoded_Data/y_train.csv")
    obj.save_dataframe(y_test, "./Data/04_encoded_Data/y_test.csv")
