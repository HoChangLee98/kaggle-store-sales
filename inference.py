import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../src")

from src.dataloader import *
from src.model import *
from src.utils import *

def main(args):
    '''model을 불러와 추론하고 제출 파일을 만드는 함수
    Describe:
    test 셋에 대해서는 2017-08-15까지의 전체 데이터를 학습 데이터로 사용하고 사전의 학습에
    사용된 train-parameter를 이용해 예측한다.
    '''

    with open(f"./config/{args.yaml}.yaml") as f:
        config = yaml.full_load(f)

    dataloader = DataLoader()
    datasets = dataloader.generate()    
    
    model = Trainer(
        model_name=config["model"], 
        params=config["params"],
        )
    print("Load Model")
    trained_model = model.load_model(config["version"])
    y_pred = trained_model.predict(datasets["X_test"].drop(columns="date"))
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred < 1, 0, y_pred)
    generate_submission(y_pred, config["model"], config["version"])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y", type=str, default="baseline", help="name of yaml file")    
    args = parser.parse_args()

    main(args)