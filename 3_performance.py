#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.preprocessing import StandardScaler
import random
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
import numpy as np


# 📌 Inhibit randomness
random.seed(42)
def model_fn(model_dir):
    """Loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)
    model.persist()
    return model
########################################################################################################################
# 📌 Load test set
test_data = pd.read_csv("test_data.csv",index_col=0)
# 📌 select the label
label="π*" # "α","β","π*"
# 📌 Remove NaN labels
test_data = test_data.dropna(subset=[label])
# 📌 Save ground truth values
y_test=test_data[label]

# 📌 Drop labels and SMILES before training
test_data = test_data.drop(columns=["α","β","π*","SMILES"])
########################################################################################################################
# 📌 5-Fold Ensemble feature selection lists
five_fold_alpha=['RNCG', 'AATSC1c', 'LogP', 'SIC3', 'IC2', 'MID_h', 'TPSA', 'NsOH', 'ETA_dEpsilon_D', 'AATSC1s', 'PEOE_VSA1', 'ATSC1are']

five_fold_beta=['MID_N', 'ATSC2c', 'BCUTdv-1l', 'RNCG', 'GATS1se', 'IC4', 'BCUTc-1h', 'GATS2pe', 'GATS1c', 'ATSC2dv', 'ATSC1s', 'SMR_VSA3', 'MATS4c', 'GATS1i', 'LogP', 'SMR_VSA6', 'AATS4Z', 'HBA', 'ATSC0c', 'MATS3s', 'AATSC2d', 'AATS4dv', 'TopoPSA', 'GATS3pe', 'GATS1dv', 'Mv', 'MZ', 'TPSA', 'Mare', 'SlogP_VSA2', 'BCUTc-1l', 'PEOE_VSA8', 'GATS2se', 'GATS2c', 'VSA_EState7', 'BCUTi-1h', 'AMID_N', 'AATS3v', 'MATS5c', 'MID_h', 'PEOE_VSA7', 'AATS3i']

five_fold_pi=['GATS2s', 'AATSC3d', 'VSA_EState8', 'SIC1', 'GATS4c', 'ATSC2dv', 'SIC0', 'AATSC2s', 'RNCG', 'MID_O', 'ATS0s', 'MINssCH2', 'MATS1se', 'GATS1s', 'AATS3i', 'ATSC3se', 'MAXsCH3', 'LogP', 'GATS4i', 'BCUTs-1l', 'BCUTd-1l', 'GATS1pe', 'GATS3se', 'MIC1', 'IC0', 'GATS1se', 'AATSC0i', 'SpMAD_Dzp', 'IC3', 'BertzCT', 'RPCG', 'MATS3d', 'GATS2d', 'AXp-2d', 'GATS1c', 'AATSC0dv', 'BCUTc-1l', 'AATS1s', 'MIC2', 'BCUTc-1h', 'AATS3dv']

merged_list=['MATS3s', 'LogP', 'ETA_dEpsilon_D', 'BCUTdv-1l', 'MATS4c', 'ATSC3se', 'BCUTd-1l', 'GATS2c', 'PEOE_VSA8', 'GATS4c', 'Mv', 'BCUTs-1l', 'ATSC2c', 'AATSC2d', 'TopoPSA', 'IC2', 'ATSC0c', 'SMR_VSA6', 'GATS3pe', 'BCUTc-1h', 'Mare', 'ATS0s', 'SIC1', 'AATS3v', 'AATS1s', 'GATS2pe', 'MATS5c', 'GATS1pe', 'GATS1dv', 'GATS2se', 'AATSC1c', 'PEOE_VSA1', 'ATSC1s', 'GATS3se', 'VSA_EState7', 'MAXsCH3', 'MZ', 'MID_N', 'AATS3i', 'AATS3dv', 'ATSC1are', 'SMR_VSA3', 'RNCG', 'MID_h', 'IC3', 'RPCG', 'MATS3d', 'AATSC1s', 'GATS2d', 'AATSC3d', 'SIC0', 'AMID_N', 'AATSC2s', 'GATS1i', 'GATS1se', 'VSA_EState8', 'MATS1se', 'IC4', 'AATS4dv', 'GATS1c', 'IC0', 'SlogP_VSA2', 'PEOE_VSA7', 'SpMAD_Dzp', 'MID_O', 'GATS2s', 'MIC1', 'BertzCT', 'MIC2', 'GATS4i', 'ATSC2dv', 'SIC3', 'GATS1s', 'AATSC0dv', 'HBA', 'MINssCH2', 'AATS4Z', 'BCUTc-1l', 'AXp-2d', 'AATSC0i', 'NsOH', 'BCUTi-1h', 'TPSA']

test_data = test_data.loc[:, five_fold_pi]
########################################################################################################################
# 📌 load autogluon model
autogluon_model=model_fn("/home/yavuz/yavuz_proje/KAT/AutogluonModels/ag-20250310_194508/")

# 📌 Get AutoGluon predictions on the test set
auto_predicted = autogluon_model.predict(test_data)
########################################################################################################################
# 📌 Load imputer_model
with open(f"imputer_model_{label}.pkl", "rb") as file:
    imputer = pickle.load(file)
# 📌 Transfrom test data
test_data_imputed = imputer.transform(test_data)
########################################################################################################################
################################                  META  LEVEL                 ##########################################
########################################################################################################################
# 📌 Load rulefit model
with open(f"rulefit_model_{label}.pkl", "rb") as file:
    rulefit_model = pickle.load(file)
# 📌 Make rulefit prediction
test_data["rulefit_pred"] = rulefit_model.predict(test_data_imputed)

# 📌 Load model
with open(f"boosted_model_{label}.pkl", "rb") as file:
    boosted_model = pickle.load(file)
# 📌 Make prediction
test_data["boosted_pred"] = boosted_model.predict(test_data_imputed)

# 📌 Load model
with open(f"tao_model_{label}.pkl", "rb") as file:
    tao_model = pickle.load(file)
# 📌 Make prediction
test_data["tao_pred"] = tao_model.predict(test_data_imputed)

########################################################################################################################
################################             Evaluate  performance            ##########################################
########################################################################################################################
# Evaluate performance
from sklearn.metrics import r2_score
print("AutoGluon R²:", r2_score(y_test, auto_predicted))
print("RuleFit R²:", r2_score(auto_predicted, test_data["rulefit_pred"]))
print("Boosted R²:", r2_score(auto_predicted, test_data["boosted_pred"]))
print("Tao R²:", r2_score(auto_predicted, test_data["tao_pred"]))

models = ["autogluon"]
for model in models:
    y_pred = auto_predicted

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print(f"📌 **{model.upper()} MODEL PERFORMANCE**")
    print(f"➡ R² Score: {r2:.4f}")
    print(f"➡ Explained Variance Score (EVS): {evs:.4f}")
    print(f"➡ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"➡ Mean Squared Error (MSE): {mse:.4f}")
    print(f"➡ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)
models = [ "rulefit", "boosted", "tao"]
for model in models:
    y_pred = test_data[f"{model}_pred"]

    mae = mean_absolute_error(auto_predicted, y_pred)
    mse = mean_squared_error(auto_predicted, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(auto_predicted, y_pred)
    evs = explained_variance_score(auto_predicted, y_pred)

    print(f"📌 **{model.upper()} MODEL PERFORMANCE**")
    print(f"➡ R² Score: {r2:.4f}")
    print(f"➡ Explained Variance Score (EVS): {evs:.4f}")
    print(f"➡ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"➡ Mean Squared Error (MSE): {mse:.4f}")
    print(f"➡ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)
