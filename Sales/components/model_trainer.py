from Sales.Logger.logger import logging
from Sales.Exception.exception import CustomException
import os, sys
from Sales.components.data_transformation import Data_Transformer, initiate_data_transform
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass
import pickle

@dataclass
class DataModelConfig:
    linear_regression_path: str = os.path.join('saved_models', 'linear_regression.pkl')
    extratree_regression_path: str = os.path.join('saved_models', 'extratree_regression.pkl')
    gradboost_regression_path: str = os.path.join('saved_models', 'gradboost_regression.pkl')
    xgboost_regression_path: str = os.path.join('saved_models', 'xgboost_regression.pkl')

def Model_Build(processed_train_data_path):

    data_model_config = DataModelConfig()

    train_data = pd.read_csv(processed_train_data_path)
    #train_data.dropna(inplace=True)

    logging.info("Started Feature selection")
    try:
        X = train_data.iloc[:,:10]
        y = train_data.iloc[:, -1]
        logging.info("Feature selection completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Started splitting the data into Train and Test set")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
        logging.info("Splitting the data into Train and Test set completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    '''

    Fitting the model using Linear Regression

    '''
    logging.info("Started fitting Model Using Linear Regression")
    try:
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        y_pred_linear_reg = regr.predict(X_test)
        linear_reg_r2_score = r2_score(y_test, y_pred_linear_reg)
        logging.info("Fitting Model Using Linear Regression completed")
        logging.info(f"Accuracy for Linear Regression is {linear_reg_r2_score}")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Saving the Linear Regression model")
    try:
        path = os.makedirs(os.path.dirname(data_model_config.linear_regression_path), exist_ok=True)
        file = open(data_model_config.linear_regression_path, 'wb')
        pickle.dump(regr, file)
        file.close()
        logging.info("Saving the Linear Regression model completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    '''

    Fitting the model using ExtraTree Regression

    '''
    logging.info("Started fitting Model Using ExtraTree Regression")
    try:
        etr = ExtraTreesRegressor()
        etr.fit(X_train, y_train)
        y_pred_extratree_reg = etr.predict(X_test)
        extratree_reg_r2_score = r2_score(y_test, y_pred_extratree_reg)
        logging.info("Fitting Model Using ExtraTree Regression completed")
        logging.info(f"Accuracy for ExtraTree Regression is {extratree_reg_r2_score}")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Saving the ExtraTree Regression model")
    try:
        path = os.makedirs(os.path.dirname(data_model_config.extratree_regression_path), exist_ok=True)
        file = open(data_model_config.extratree_regression_path, 'wb')
        pickle.dump(etr, file)
        file.close()
        logging.info("Saving the ExtraTree Regression model completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    '''

    Fitting the model using Gradient Boosting Regression

    '''
    logging.info("Started fitting Model Using Gradient Boosting Regression")
    try:
        gbr = GradientBoostingRegressor()
        gbr.fit(X_train, y_train)
        y_pred_gradboost_reg = gbr.predict(X_test)
        gradboost_reg_r2_score = r2_score(y_test, y_pred_gradboost_reg)
        logging.info("Fitting Model Using Gradient Boosting Regression completed")
        logging.info(f"Accuracy for Gradient Boosting Regression is {gradboost_reg_r2_score}")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Saving the Gradient Boosting Regression model")
    try:
        path = os.makedirs(os.path.dirname(data_model_config.gradboost_regression_path), exist_ok=True)
        file = open(data_model_config.gradboost_regression_path, 'wb')
        pickle.dump(gbr, file)
        file.close()
        logging.info("Saving the Gradient Boosting Regression model completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)

    '''

    Fitting the model using XGBoost Regression

    '''
    logging.info("Started fitting Model Using XGBoost Regression")
    try:
        xgb = XGBRFRegressor()
        xgb.fit(X_train, y_train)
        y_pred_xgboost_reg = xgb.predict(X_test)
        xgboost_reg_r2_score = r2_score(y_test, y_pred_xgboost_reg)
        logging.info("Fitting Model Using XGBoost Regression completed")
        logging.info(f"Accuracy for XGBoost Regression is {xgboost_reg_r2_score}")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    logging.info("Saving the XGBoost Regression model")
    try:
        path = os.makedirs(os.path.dirname(data_model_config.xgboost_regression_path), exist_ok=True)
        file = open(data_model_config.xgboost_regression_path, 'wb')
        pickle.dump(xgb, file)
        file.close()
        logging.info("Saving the XGBoost Regression model completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
    
    #print([linear_reg_r2_score,extratree_reg_r2_score,gradboost_reg_r2_score])
    
    return (data_model_config.linear_regression_path, data_model_config.extratree_regression_path, data_model_config.gradboost_regression_path, data_model_config.xgboost_regression_path)
