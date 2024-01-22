import os
import sys
from pathlib import Path
import pandas as pd

from src.utils.exception import CustomException
from src.utils.ml_helper import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = Path('artifacts/preprocessing/preprocessor.pkl')
            model_path = Path("artifacts/models/model.pkl")

            print("Before Loading")

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            print("After Loading")
            
            data_transformed = preprocessor.transform(features)
            prediction = model.predict(data_transformed)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self, age: int, sex: str, bmi: float,
                children: int, smoker: str, region: str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age': [self.age], 
                'sex': [self.sex], 
                'bmi': [self.bmi], 
                'children': [self.children], 
                'smoker': [self.smoker], 
                'region': [self.region]
            }
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)