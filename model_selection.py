import os
from glob import glob

from src.model_select import best_test
from src.model_select import best_validation
from src.model_select import test_model

if __name__ == '__main__':
    metrics = test_model("experiments/**/*")
    metrics.to_csv("./logs/test_metrics.csv")
    best_validation(map(os.path.abspath, glob("./logs/*.csv")))
    best_test(metrics)
