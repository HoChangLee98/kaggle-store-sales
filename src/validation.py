import os
import pickle

import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.metrics import mean_squared_log_error
from tqdm.auto import tqdm

class Validator:
    def __init__(self, model_name:str, cv:bool=None, model:object=None, X:pd.DataFrame=None, y:pd.DataFrame=None, method:str="block", folder_path:str="./artifacts"):
        '''

        Args:
            method: "block" 과 "forward" 중 방법을 선택
        '''
        self.model_name = model_name
        self.cv = cv
        self.model = model
        self.X = X
        self.y = y
        self.method = method
        self.folder_path = folder_path
        
    def custom_walk_forward(self):
        '''walk forward 방법으로 train set과 validation set을 구분해주는 함수
        각 년도별 8월 16일부터 8월 31일까지의 데이터를 validation set으로 설정하고 
        각 년도별 8월 16일 이전의 모든 데이터를 train set으로 설정한다.
        
        '''
        X = self.X.copy()
        y = self.y.copy()

        period = {
            "2013" : (pd.Timestamp("2013-08-16"), pd.Timestamp("2013-08-31")),
            "2014" : (pd.Timestamp("2014-08-16"), pd.Timestamp("2014-08-31")),
            "2015" : (pd.Timestamp("2015-08-16"), pd.Timestamp("2015-08-31")),
            "2016" : (pd.Timestamp("2016-08-16"), pd.Timestamp("2016-08-31")),
        }
        
        y_valid_list = []
        y_valid_pred_list = []
      
        for i in tqdm(list(period.keys())):
            print(f"    ## Start custom walk forward Validation: {i}")
            X_train = X[(X["date"] < period[i][0])].drop(columns=["date"])
            y_train = y[X_train.index]
            # train_index = data[(data["date"] < period[i][0])].index
            
            X_valid = X[(X["date"] >= period[i][0]) & (X["date"] <= period[i][1])].drop(columns=["date"])
            y_valid = y[X_valid.index]
            # valid_index = data[(data["date"] >= period[i][0]) & (data["date"] <= period[i][1])].index    
            
            temp_model = deepcopy(self.model)
            temp_model.train(self.cv, X_train, y_train, X_valid, y_valid)
            y_val_pred = temp_model.predict(X_valid)
            # y_val_pred = np.round(np.expm1(y_val_pred))
            y_val_pred = np.array(y_val_pred)
            y_val_pred = np.where(y_val_pred < 1, 0, y_val_pred)
            
            rmsle = mean_squared_log_error(y_valid, y_val_pred)
            print(f"    rmsle for {i}: {rmsle}")
            
            y_valid_list += list(y_valid)
            y_valid_pred_list += list(y_val_pred)
        
        score = mean_squared_log_error(y_valid_list, y_valid_pred_list)
        print(f"    All rmsle : {score}")
        result = {"score" : score, "y_valid" : y_valid_list, "y_valid_pred" : y_valid_pred_list}

        return result
    
    def custom_blocked_split(self):
        '''blocked time series split 방법으로 train set과 validation set을 구분해주는 함수
        각 년도별 8월 16일부터 8월 31일까지의 데이터를 validation set으로 설정하고 
        각 년도별 1월 1일부터 8월 15일까지의 데이터를 train set으로 설정한다.
        
        '''
        X = self.X.copy()
        y = self.y.copy()

        period = {
            "2013" : (pd.Timestamp("2013-01-01"), pd.Timestamp("2013-08-15"), pd.Timestamp("2013-08-16"), pd.Timestamp("2013-08-31")),
            "2014" : (pd.Timestamp("2014-01-01"), pd.Timestamp("2014-08-15"), pd.Timestamp("2014-08-16"), pd.Timestamp("2014-08-31")),
            "2015" : (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-08-15"), pd.Timestamp("2015-08-16"), pd.Timestamp("2015-08-31")),
            "2016" : (pd.Timestamp("2016-01-01"), pd.Timestamp("2016-08-15"), pd.Timestamp("2016-08-16"), pd.Timestamp("2016-08-31")),
            }

        y_valid_list = []
        y_valid_pred_list = []
      
        for i in tqdm(list(period.keys())):    
            print(f"    ## Start custom blocked split Validation: {i}") 
            X_train = X[(X["date"] >= period[i][0]) & (X["date"] <= period[i][1])].drop(columns=["date"])
            y_train = y[X_train.index]
            # train_index = data[(data["date"] >= period[i][0]) & (data["date"] <= period[i][1])].index
            
            X_valid = X[(X["date"] >= period[i][2]) & (X["date"] <= period[i][3])].drop(columns=["date"])
            y_valid = y[X_valid.index]
            # valid_index = data[(data["date"] >= period[i][2]) & (data["date"] <= period[i][3])].index
            
            temp_model = deepcopy(self.model)
            temp_model.train(self.cv, X_train, y_train, X_valid, y_valid)
            y_val_pred = temp_model.predict(X_valid)
            # y_val_pred = np.round(np.expm1(y_val_pred))
            y_val_pred = np.array(y_val_pred)
            y_val_pred = np.where(y_val_pred < 1, 0, y_val_pred)
            
            rmsle = mean_squared_log_error(y_valid, y_val_pred)
            print(f"    rmsle for {i}: {rmsle}")
            
            y_valid_list += list(y_valid)
            y_valid_pred_list += list(y_val_pred)
        
        score = mean_squared_log_error(y_valid_list, y_valid_pred_list)
        print(f"    All rmsle : {score}")
        result = {"score" : score, "y_valid" : y_valid_list, "y_valid_pred" : y_valid_pred_list}

        return result
    
    def load_cv_result(self, version:str):
        with open(f"{self.folder_path}/cv_result_{self.model_name}_{version}.pkl", "rb") as f:
            cv_result = pickle.load(f)
            
        return cv_result
    
    def save_cv_result(self, cv_result:dict, version:str):
        with open(f"{self.folder_path}/cv_result_{self.model_name}_{version}.pkl", "wb") as f:
            pickle.dump(cv_result, f)
    