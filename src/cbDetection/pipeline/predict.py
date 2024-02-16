import os
import sys
from pathlib import Path
import pandas as pd

from cbDetection.utils.exception import CustomException
from cbDetection.utils.ml_helper import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load pre-trained models
            preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')
            preprocessor = load_object(file_path=preprocessor_path)

            model_path = Path("artifacts/model_trainer/model.pkl")
            model = load_object(file_path=model_path)

            # Data preprocessing
            transformed_data = preprocessor.transform(features)

            # Predictions
            prediction = model.predict(transformed_data)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, tweet_text: str):
        self.tweet_text = tweet_text

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'tweet_text': [self.tweet_text]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)