import os
import pickle

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataLoader:
    def __init__(self, file_path:str="../data"):
        self.train = pd.read_csv(f"{file_path}/train.csv")
        self.train = self.train.drop(columns="id")
        self.test = pd.read_csv(f"{file_path}/test.csv")
        self.test = self.test.drop(columns="id")
    
    def run(self):
        ## Setting evaluation set
        train_set, eval_set = self.eval_splitter(train=self.train)

        train_set["X"] = self.date_processor(train_set["X"])
        eval_set["X"] = self.date_processor(eval_set["X"])
        test = self.date_processor(self.test)

        return train_set, eval_set, test
    
    def date_processor(self, data:pd.DataFrame):
        data["date"] = pd.to_datetime(data["date"])        
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["weekday"] = data["date"].dt.weekday
        data["quarter"] = data["date"].dt.quarter
        data = data.drop(columns="date")
        
        return data
    
    def eval_splitter(self, train:pd.DataFrame, random_state:int=0, eval_size:float=0.2):
        X = train.drop(columns="sales")
        y = train["sales"]        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=eval_size, random_state=random_state)
        train_set = {"X" : X_tr.reset_index(drop=True), "y" : y_tr.reset_index(drop=True)}
        eval_set = {"X" : X_val.reset_index(drop=True), "y" : y_val.reset_index(drop=True)}
        
        return train_set, eval_set
        

class PreProcess:
    def __init__(
        self, 
        X_tr:pd.DataFrame,  
        catfeature:list=["family"],
        encoding_mode:str="label"
        ):
        self.X_tr = X_tr
        self.catfeature = catfeature                
        self.encoding_mode = encoding_mode
        self.encoder = self.load_encoder()

    def transform(self, X:pd.DataFrame):
        if self.encoding_mode == "onehot":            
            catfeatures = pd.DataFrame(self.encoder.transform(X[self.catfeature]), columns=[col for col in self.encoder.categories_[0]])
            X = pd.concat([X, catfeatures], axis=1)
            X = X.drop(columns=self.catfeature)
            
        elif self.encoding_mode == "label": 
            X[self.catfeature] = self.encoder.transform(X[self.catfeature]).reshape(-1,1)
            X[self.catfeature] = X[self.catfeature].astype("category")
        
        X = self.arrange_dtype(X)
        
        return X        

    def fitting_encoder(self):
        if self.encoding_mode == "onehot":        
            encoder = OneHotEncoder(sparse_output=False)
            encoder.fit(self.X_tr[self.catfeature])     
            
        elif self.encoding_mode == "label": 
            encoder = LabelEncoder()
            encoder.fit(self.X_tr[self.catfeature])     
        
        return encoder        
    
    def load_encoder(self):
        if os.path.isfile(f"../artifacts/{self.encoding_mode}_encoder.pkl"):
            with open(f"../artifacts/{self.encoding_mode}_encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
                print(f"Loading {self.encoding_mode} encoder")

        else:
            encoder = self.save_encoder()

        return encoder

    def save_encoder(self):
        encoder = self.fitting_encoder()
        
        with open(f"../artifacts/{self.encoding_mode}_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
            print(f"Saving {self.encoding_mode} encoder")
            
        return encoder

    def arrange_dtype(self, X:pd.DataFrame):
        catfeatures = ["store_nbr", "year", "month", "day", "weekday", "quarter"]
        X[catfeatures] = X[catfeatures].astype("category")
    
        return X   