import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error


class Trainer:
    def __init__(self, model_name:str="catboost"):
        self.model_name = model_name 


    def train(self, model, tune_parameters:dict, parameters:dict):
        

    def save_evaluation(self):
        
    def load_model(self, )
        



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


