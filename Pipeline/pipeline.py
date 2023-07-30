import os
import sys
from Sales.Logger.logger import logging
from Sales.Exception.exception import CustomException
from Sales.components.data_transformation import Data_Transformer, initiate_data_transform
from  Sales.components.data_ingestion import read_train_data, read_test_data
from  Sales.components.model_trainer import Model_Build
from Pipeline.model_prediction import Prediction

def Model_Pipeline():

    logging.info("Pipeline started")

    try:
        ### Data Ingestion
        train_data_path, test_data_path = initiate_data_transform()

        ### Data Transformer
        processed_train_data_path, processed_test_data_path = Data_Transformer(train_data_path, test_data_path)

        ### Data Model
        _, _, _,xgboost_regression_path = Model_Build(processed_train_data_path)

        ### Predict Pipeline
        pred = Prediction(xgboost_regression_path, processed_test_data_path)

        logging.info("Pipeline completed")

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)

Model_Pipeline()