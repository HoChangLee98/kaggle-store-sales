import os
import pickle
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
        ## Setting Validation 
        train_set, valid_set = self.valid_splitter(train=self.train)

        train_set["X"] = self.date_processor(train_set["X"])
        valid_set["X"] = self.date_processor(valid_set["X"])
        test = self.date_processor(self.test)

        return train_set, valid_set, test
    
    def date_processor(self, data:pd.DataFrame):
        data["date"] = pd.to_datetime(data["date"])        
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["weekday"] = data["date"].dt.weekday
        data["quarter"] = data["date"].dt.quarter
        data = data.drop(columns="date")
        
        return data
    
    def valid_splitter(self, train:pd.DataFrame, random_state:int=0, valid_size:float=0.2):
        X = train.drop(columns="sales")
        y = train["sales"]        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=valid_size, random_state=random_state)
        train_set = {"X" : X_tr.reset_index(drop=True), "y" : y_tr.reset_index(drop=True)}
        valid_set = {"X" : X_val.reset_index(drop=True), "y" : y_val.reset_index(drop=True)}
        
        return train_set, valid_set
        

class PreProcess:
    def __init__(
        self, 
        X_tr:pd.DataFrame,  
        catfeature:list=["family"]
        ):
        self.X_tr = X_tr
        self.catfeature = catfeature                
        self.encoder = self.load_encoder()

    def transform(self, X:pd.DataFrame):
        catfeatures = pd.DataFrame(self.encoder.transform(X[self.catfeature]), columns=[col for col in self.encoder.categories_[0]])
        X = pd.concat([X, catfeatures], axis=1)
        # X = X.drop(columns=self.catfeature)
        
        return X        

    def fitting_encoder(self):
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(self.X_tr[self.catfeature])        
        
        return encoder        
    
    def load_encoder(self):
        if os.path.isfile("../artifacts/encoder.pkl"):
            with open("../artifacts/encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
                print("Loading encoder")

        else:
            encoder = self.save_encoder()

        return encoder

    def save_encoder(self):
        encoder = self.fitting_encoder()
        
        with open("../artifacts/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
            print("Saving encoder")
            
        return encoder
