import os
import random

from datetime import datetime

import numpy as np
import pandas as pd

def seed_everything(seed:int=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def get_blocked_split_trainset(datasets):
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    
    period = (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-08-15"))
    X_train = X_train[(X_train["date"] >= period[0]) & (X_train["date"] <= period[1])].drop(columns=["date"])
    y_train = y_train[X_train.index]
    
    return X_train, y_train

def get_walk_forward_trainset(datasets):
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    
    period = (pd.Timestamp("2013-01-01"), pd.Timestamp("2017-08-15"))
    X_train = X_train[(X_train["date"] >= period[0]) & (X_train["date"] <= period[1])].drop(columns=["date"])
    y_train = y_train[X_train.index]
    
    return X_train, y_train

def generate_submission(y_pred, model, version, folder_path:str="./submit"):
    submission = pd.read_csv(f"{folder_path}/sample_submission.csv")
    submission["sales"] = y_pred
    time = datetime.today().strftime("%y%m%d%H%M%S")   
    print("Save Submission")
    submission.to_csv(f"{folder_path}/{version}_{model}_{time}.csv", index=False)

def after_process(y):
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = 0
    
    return y     