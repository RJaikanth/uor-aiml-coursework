import os
import sys

from src.config import parse_hp_config
from src.hp_tune import tune_hp
from src.utils import Logger

if __name__ == '__main__':
    # Experiment Metadata
    exp_name = "hp_tuning"
    exp_path = f"experiments/{exp_name}"
    os.makedirs(exp_path, exist_ok = True)
    logger = Logger("logs/hp_tuning.log")
    sys.stdout = logger
    sys.stderr = sys.stdout

    # Config
    conf_ = parse_hp_config("./config/hyperparameter.yml")
    tune_hp(conf_)
