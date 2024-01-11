from src.constant import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import load_model
import pickle
from sklearn.pipeline import Pipeline  



PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV = "prediction.csv"
PREDICTION_FILE = "output.csv"
FEATURE_ENG_FOLDER = "feature_eng"



ROOT_DIR = os.getcwd()
BATCH_PREDICTION_CSV = os.path.join(ROOT_DIR, PREDICTION_FOLDER, PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR, PREDICTION_FOLDER, FEATURE_ENG_FOLDER)



class batch_prediction:
    def __init__(self, input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path):
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path


    def start_batch_prediction(self):
        try:
            # Load the feature engineering pipeline path
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            # Load the data transformation pipeline path
            with open(self.transformer_file_path, 'rb') as f:
                processor = pickle.load(f)

            # Load the model
            model = load_model(self.model_file_path)

            # Create a feature engineering pipeline
            feature_engineering_pipeline = Pipeline([("feature_engineering", feature_pipeline)])

            df = pd.read_csv(self.input_file_path)

            # Apply Feature engineering pipeline steps
            df = feature_engineering_pipeline.transform(df)

            FEATURE_ENGINEERING_PATH = FEATURE_ENG
            os.makedirs(FEATURE_ENGINEERING_PATH, exist_ok=True)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, 'batch_feature_eng.csv')

            df.to_csv(file_path, index=False)

            # Drop the 'Time_taken (min)' column
            df = df.drop('Time_taken (min)', axis=1)

            transformed_data = processor.transform(df)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, 'processor.csv')

            predictions = model.predict(transformed_data)

            df_prediction = pd.DataFrame(predictions, columns=['prediction'])

            BATCH_PREDICTION_PATH = PREDICTION_FOLDER
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH, PREDICTION_FILE)

            df_prediction.to_csv(csv_path, index=False)
            logging.info("Batch prediction done")


        except Exception as e:
            raise CustomException(e, sys.exc_info())

