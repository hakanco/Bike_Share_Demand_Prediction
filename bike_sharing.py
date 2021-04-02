import streamlit as st 
import numpy as np 
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import os
import re
import csv
import pickle
import requests

import matplotlib.pyplot as plt
#from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

model_filename = "model"
model_extension = "pkl"

st.title(""" Bike Share Demand Prediction """)

st.write("""
# Using various Regressors
Which one gives the best error rate?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Bike Share','empty')
)

#st.write(f"## {dataset_name} Dataset")

regressor_name = st.sidebar.selectbox(
    'Select regressor',
    ('Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'kNN', 'Random Forest', 'XGBoost')
)

def get_dataset(name):
    df = None
    if name == 'Bike Share':
        store_sharing_filename = 'store_sharing.csv'
        df = pd.read_csv(store_sharing_filename)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
        df['year'] = pd.DatetimeIndex(df['timestamp']).year
        df['month'] = pd.DatetimeIndex(df['timestamp']).month
        df['day'] = pd.DatetimeIndex(df['timestamp']).day
        df['hour'] = pd.DatetimeIndex(df['timestamp']).hour
        df = df.drop(columns=['timestamp'], axis=1)

        columns = ['t1', 't2', 'hum', 'wind_speed']
        scaler = StandardScaler()
        df.loc[:, columns] = scaler.fit_transform(df[columns])

        def OneHotEncoder(df, column, prefix):
            df_column = pd.DataFrame(df, columns=[column])
            ohe_df = pd.get_dummies(df_column, columns=[column], prefix=[prefix], prefix_sep='-')
            df = df.join(ohe_df)
            return df.drop(columns=[column], axis=1)

        df = OneHotEncoder(df, 'weather_code', 'W')
        df = OneHotEncoder(df, 'season', 'Season')
        df.insert(22, 'count', df['cnt'])
        df = df.drop(columns=['cnt'], axis=1)

        X = df.iloc[:, :21].values.reshape(-1, 21)                                               
        y = df.iloc[:, -1].values.reshape(-1, 1).ravel()
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)

def add_parameter_ui(reg_name):
    params = dict()
    if reg_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    elif reg_name == 'kNN':
        weights = st.sidebar.selectbox('weights', ('uniform', 'distance'))
        params['weights'] = weights
    elif reg_name == 'XGBoost':
        max_depth = st.sidebar.slider('max_depth', 9, 12)
        params['max_depth'] = max_depth
        min_child_weights = st.sidebar.slider('min_child_weight', 5, 8)
        params['min_child_weights'] = min_child_weights
    return params

params = add_parameter_ui(regressor_name)

def get_regressor(reg_name, params):
    reg = None
    if reg_name == 'XGBoost':
        reg = GridSearchCV( estimator=XGBRegressor(), param_grid=params, cv=5, scoring='neg_mean_squared_error')
    elif reg_name == 'kNN':
        reg = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=params, cv=5, scoring='neg_mean_squared_error')
    elif reg_name == 'Random Forest':
        reg = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5, scoring='neg_mean_squared_error')
    elif reg_name == 'Linear Regression':
        reg = LinearRegression()
    elif reg_name == 'Ridge':
        reg = RidgeCV(cv=5)
    elif reg_name == 'Lasso':
        reg = LassoCV(cv=5, random_state=42)
    elif reg_name == 'ElasticNet':
        reg = ElasticNetCV(cv=5, random_state=42)
    return reg

reg = get_regressor(regressor_name, params)


#### REGRESSION ####

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

st.write('Shape of train set:', X_train.shape)
st.write('Shape of test set:', X_test.shape)

reg.fit(X_train, Y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)

st.write(f'Regression = **{regressor_name}**')
st.write(f'MSE =', mse)
st.write(f'RMSE =', rmse)



