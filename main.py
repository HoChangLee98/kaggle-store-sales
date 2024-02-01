import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../src")

from src.utils import *
from src.dataloader import *
from src.validation import *
from src.model import *

def main(args):
    '''model을 학습하고 validation set에 대한 결과를 저장하는 함수
    
    Describe:
        2013, 2014, 2015, 2016의 validation set에 대해 동일한 train-parameter로 모델에 대해
        학습을 진행한다.
        validation set에 대한 evaulation은 2013 ~ 2016까지의 08-16 ~ 08-31에 대해 rmsle로 
        계산한다.
    '''
    seed_everything()

    with open(f"./config/{args.yaml}.yaml") as f:
        config = yaml.full_load(f)

    dataloader = DataLoader()
    datasets = dataloader.generate()           

    model = Trainer(
        model_name=config["model"], 
        params=config["params"],
        )
    
    if args.mode == "cv":
        validator = Validator(
            model_name=config["model"], 
            cv=True, 
            model=model, 
            X=datasets["X_train"], 
            y=datasets["y_train"]
            )
        
        cv_result = validator.custom_blocked_split()
        validator.save_cv_result(cv_result=cv_result, version=config["version"])
    
    elif args.mode == "train":
        X_train, y_train = get_blocked_split_trainset(datasets=datasets)
        model.train(cv=False, X_train=X_train, y_train=y_train)   
        model.save_model(version=config["version"])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y", type=str, default="catboost", help="name of yaml file")    
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed number")
    parser.add_argument("--mode", "-m", type=str, default="cv", choices=["cv", "train"], help="choose trainer mode")
    args = parser.parse_args()

    main(args)