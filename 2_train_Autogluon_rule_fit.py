#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu
import time
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from rulefit import RuleFit
from sklearn.preprocessing import StandardScaler
import random
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from imodels import GreedyTreeRegressor, HSTreeRegressor, SLIMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PowerTransformer, SplineTransformer
from sklearn.impute import SimpleImputer

# 📌 Inhibit randomness
random.seed(42)

def model_fn(model_dir):
    """Loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)
    model.persist()
    return model

########################################################################################################################
# 📌 Upload the data
data = pd.read_csv("train_data.csv",index_col=0)
label="π*" # "α","β","π*"
########################################################################################################################
# 📌 Preprocess label
data = data.dropna(subset=[label])  # Removes NaN values
y=data[label]
########################################################################################################################
# selected_features
five_fold_alpha=['RNCG', 'AATSC1c', 'LogP', 'SIC3', 'IC2', 'MID_h', 'TPSA', 'NsOH', 'ETA_dEpsilon_D', 'AATSC1s', 'PEOE_VSA1', 'ATSC1are']

five_fold_beta=['MID_N', 'ATSC2c', 'BCUTdv-1l', 'RNCG', 'GATS1se', 'IC4', 'BCUTc-1h', 'GATS2pe', 'GATS1c', 'ATSC2dv', 'ATSC1s', 'SMR_VSA3', 'MATS4c', 'GATS1i', 'LogP', 'SMR_VSA6', 'AATS4Z', 'HBA', 'ATSC0c', 'MATS3s', 'AATSC2d', 'AATS4dv', 'TopoPSA', 'GATS3pe', 'GATS1dv', 'Mv', 'MZ', 'TPSA', 'Mare', 'SlogP_VSA2', 'BCUTc-1l', 'PEOE_VSA8', 'GATS2se', 'GATS2c', 'VSA_EState7', 'BCUTi-1h', 'AMID_N', 'AATS3v', 'MATS5c', 'MID_h', 'PEOE_VSA7', 'AATS3i']

five_fold_pi=['GATS2s', 'AATSC3d', 'VSA_EState8', 'SIC1', 'GATS4c', 'ATSC2dv', 'SIC0', 'AATSC2s', 'RNCG', 'MID_O', 'ATS0s', 'MINssCH2', 'MATS1se', 'GATS1s', 'AATS3i', 'ATSC3se', 'MAXsCH3', 'LogP', 'GATS4i', 'BCUTs-1l', 'BCUTd-1l', 'GATS1pe', 'GATS3se', 'MIC1', 'IC0', 'GATS1se', 'AATSC0i', 'SpMAD_Dzp', 'IC3', 'BertzCT', 'RPCG', 'MATS3d', 'GATS2d', 'AXp-2d', 'GATS1c', 'AATSC0dv', 'BCUTc-1l', 'AATS1s', 'MIC2', 'BCUTc-1h', 'AATS3dv']

merged_list=['MATS3s', 'LogP', 'ETA_dEpsilon_D', 'BCUTdv-1l', 'MATS4c', 'ATSC3se', 'BCUTd-1l', 'GATS2c', 'PEOE_VSA8', 'GATS4c', 'Mv', 'BCUTs-1l', 'ATSC2c', 'AATSC2d', 'TopoPSA', 'IC2', 'ATSC0c', 'SMR_VSA6', 'GATS3pe', 'BCUTc-1h', 'Mare', 'ATS0s', 'SIC1', 'AATS3v', 'AATS1s', 'GATS2pe', 'MATS5c', 'GATS1pe', 'GATS1dv', 'GATS2se', 'AATSC1c', 'PEOE_VSA1', 'ATSC1s', 'GATS3se', 'VSA_EState7', 'MAXsCH3', 'MZ', 'MID_N', 'AATS3i', 'AATS3dv', 'ATSC1are', 'SMR_VSA3', 'RNCG', 'MID_h', 'IC3', 'RPCG', 'MATS3d', 'AATSC1s', 'GATS2d', 'AATSC3d', 'SIC0', 'AMID_N', 'AATSC2s', 'GATS1i', 'GATS1se', 'VSA_EState8', 'MATS1se', 'IC4', 'AATS4dv', 'GATS1c', 'IC0', 'SlogP_VSA2', 'PEOE_VSA7', 'SpMAD_Dzp', 'MID_O', 'GATS2s', 'MIC1', 'BertzCT', 'MIC2', 'GATS4i', 'ATSC2dv', 'SIC3', 'GATS1s', 'AATSC0dv', 'HBA', 'MINssCH2', 'AATS4Z', 'BCUTc-1l', 'AXp-2d', 'AATSC0i', 'NsOH', 'BCUTi-1h', 'TPSA']

X_selected = data.loc[:, five_fold_pi]
########################################################################################################################
# 📌 Split the data
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 📌 Train the base model using AutoGluon
train_data = pd.concat([X_train, y_train], axis=1)
# 📌 Train
predictor = TabularPredictor(label=label, eval_metric="root_mean_squared_error").fit(
    train_data, auto_stack=True, time_limit=60*2, keep_only_best=True,
    num_stack_levels=1, num_bag_folds=8,   num_bag_sets=1, presets="best_quality") 

########################################################################################################################
# 📌 Make prediction on validation set, to train meta models
val_data_selected = X_val.copy()
val_data = pd.concat([X_val, y_val], axis=1)
val_data["autogluon_pred"] = predictor.predict(val_data_selected)
########################################################################################################################
# 📌 Prepare data for meta-level
rulefit_target = val_data["autogluon_pred"]
rulefit_features = val_data.drop(columns=[label,"autogluon_pred"])
#rulefit_features = rulefit_features.fillna(0)
imputer = SimpleImputer(strategy='median')  # Use 'mean', 'most_frequent', or 'constant' if needed
#imputer = StandardScaler()
rulefit_features = imputer.fit_transform(rulefit_features)

# 📌 Save the fitted scaler
with open(f"imputer_model_{label}.pkl", "wb") as file:
    pickle.dump(imputer, file)

########################################################################################################################
###########################                 Train Meta-level                         ###################################
########################################################################################################################
# 📌 Train rulefit model
rulefit_model = RuleFit(tree_size=20, rfmode="regress", lin_standardise=True, max_iter=20000, random_state=42)
rulefit_model.fit(rulefit_features, rulefit_target)

# 📌 Save rulefit model
with open(f"rulefit_model_{label}.pkl", "wb") as f:
    pickle.dump(rulefit_model, f)

print("Traning is finished!!!!!!!!!!!!!!!!!!!!!!!!!!")


