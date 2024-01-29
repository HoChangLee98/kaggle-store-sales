import os
import pickle

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

class RmsleMetric(object):
    def get_final_error(self, error, weight):
        
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approx[i] - target[i])**2)

        return error_sum, weight_sum

class Trainer:
    def __init__(
        self, 
        model_name:str, 
        params:dict=None, 
        ):
        self.train_params = params["train_params"]
        self.model_params = params["model_params"]
        if model_name == "catboost":
            self.model = CatBoostRegressor(**self.train_params)
        elif model_name == "xgboost":
            self.model = XGBRegressor(**self.train_params)
        elif model_name == "lightgbm":
            self.model = LGBMRegressor(**self.train_params)       
        elif model_name == "randomforest":
            self.model = RandomForestRegressor(**self.train_params)     
    
    def train(
        self, 
        cv:bool, 
        X_train:pd.DataFrame, 
        y_train:pd.DataFrame,
        X_valid:pd.DataFrame=None,
        y_valid:pd.DataFrame=None,
        ):
        if cv:
            self.train_params["eval_metric"] = RmsleMetric    
            self.model.fit(
                X_train, 
                y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)], 
                **self.model_params
            )
        else:
            self.model.fit(X_train, y_train, **self.model_params)
    
    def predict(self, X_test:pd.DataFrame):
        pred = self.model.predict(X_test)
        
        return pred

    def load_model(self, version:str, folder_path:str="./artifacts"):
        with open(f"{folder_path}/model_{version}.pkl", "rb") as f:
            trained_model = pickle.load(f)
            
        return trained_model

    def save_model(self, version:str, folder_path:str="./artifacts"):
        with open(f"{folder_path}/model_{version}.pkl", "wb") as f:
            pickle.dump(self.model, f)
        
