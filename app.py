from Sales.Logger.logger import logging
from Sales.Exception.exception import CustomException
import sys, os
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from Sales.components.data_transformation import Data_Transformer, initiate_data_transform
from Sales.components.data_ingestion import read_train_data, read_test_data
from Sales.components.model_trainer import Model_Build, DataModelConfig
from Pipeline.model_prediction import Prediction
from Pipeline.pipeline import Model_Pipeline

DIR = 'saved_models'
MODEL_DIR = 'gradboost_regression.pkl' 
MODEL_PATH = os.path.join(DIR, MODEL_DIR)
app = Flask(__name__)

try:
    logging.info("Model is loading")
    model = pickle.load(open(MODEL_PATH, "rb"))
    logging.info("Model loading is completed")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Item_Weight
        Item_Weight = float(request.form["Item_Weight"])

        # Item_Fat_Content
        Item_Fat_Content = int(request.form["Item_Fat_Content"])

        # Item_Visibility
        Item_Visibility = float(request.form["Item_Visibility"])

        # Item_Type
        Item_Type = int(request.form["Item_Type"])

        # Item_MRP
        Item_MRP = float(request.form["Item_MRP"])

        # Outlet_Identifier
        Outlet_Identifier = int(request.form["Outlet_Identifier"])

        # Outlet_Establishment_Year
        Outlet_Establishment_Year = int(request.form["Outlet_Establishment_Year"])

        # Outlet_Size
        Outlet_Size = int(request.form["Outlet_Size"])

        # Outlet_Location_Type
        Outlet_Location_Type = int(request.form["Outlet_Location_Type"])

        # Outlet_Type
        Outlet_Type = int(request.form["Outlet_Type"])

        prediction = model.predict([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type,
                                     Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size,
                                     Outlet_Location_Type, Outlet_Type]])

        output = round(prediction[0],4)

        return render_template('home.html',prediction_text="Item Outlet Sales is. {}".format(output))

    return render_template("home.html")

if __name__ == "__main__":
    app.run()



        
