import os
import sys
from Sales.Logger.logger import logging
from Sales.Exception.exception import CustomException
import pickle
from Sales.components.data_transformation import Data_Transformer, initiate_data_transform
from Sales.components.data_ingestion import read_train_data, read_test_data
from Sales.components.model_trainer import Model_Build
import numpy as np
import pandas as pd


def Prediction(model_path, processed_test_data_path):
    logging.info("Started prediction on test data")
    try:
        data = pd.read_csv(processed_test_data_path)
        model = pickle.load(open(model_path, 'rb'))
        prediction = model.predict(data)
        logging.info("Prediction on test data completed")
        return prediction
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
        