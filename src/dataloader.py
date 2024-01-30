import os
import pickle

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, folder_path:str="./data"):
        self.folder_path = folder_path
        self.train = pd.read_csv(f"{folder_path}/train.csv")
        # self.train = self.train.drop(columns="id")
        
        self.test = pd.read_csv(f"{folder_path}/test.csv")
        # self.test = self.test.drop(columns="id")
        
        self.oil = pd.read_csv(f"{folder_path}/oil.csv")
        self.oil["date"] = pd.to_datetime(self.oil["date"])
    
    def generate(self):
        file_path = f"{self.folder_path}/datasets.pkl"
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                datasets = pickle.load(f)
        else:
            X_train = self.date_features(self.train.drop(columns="sales"))
            X_test = self.date_features(self.test)
            X_train, X_test = self.merge_with_oil(X_train=X_train, X_test=X_test)
            X_train, X_test = self.label_transform(X_train=X_train, X_test=X_test)
            X_train, X_test = self.arrange_dtype(X_train), self.arrange_dtype(X_test)            
                                    
            datasets = {}
            datasets["X_train"] = X_train.drop(columns="id")
            datasets["y_train"] = self.train["sales"]
            datasets["X_test"] = X_test.drop(columns="id")

            with open(file_path, "wb") as f:
                pickle.dump(datasets, f)
            
        return datasets

    def label_transform(self, X_train, X_test, cat_features:list=["family"]):
        for col in cat_features:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col]) 
        
        return X_train, X_test        

    def date_features(self, data:pd.DataFrame):
        data["date"] = pd.to_datetime(data["date"])        
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["weekday"] = data["date"].dt.weekday
        data["quarter"] = data["date"].dt.quarter
        # data = data.drop(columns="date")
        
        return data

    def arrange_dtype(self, data:pd.DataFrame):
        catfeatures = ["store_nbr", "year", "family", "month", "day", "weekday", "quarter"]
        data[catfeatures] = data[catfeatures].astype("category")
    
        return data   
 
    def merge_with_oil(self, X_train, X_test):
        X_train = pd.merge(X_train, self.oil, on="date").fillna(method="backfill")
        X_train.index = X_train["id"]
        X_test = pd.merge(X_test, self.oil[self.oil["date"] >= pd.to_datetime("2017-08-16")], how="outer", on="date").fillna(method="backfill")
        X_test.index = X_test["id"]
        
        return X_train, X_test