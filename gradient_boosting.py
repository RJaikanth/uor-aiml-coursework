import collections
import os
import sys

import pandas as pd

from src.config import parse_config
from src.train import train_sklearn
from src.utils import Logger

if __name__ == '__main__':
    # Set sys.stdout
    os.makedirs("./logs/", exist_ok = True)
    logger = Logger("logs/GradientBoosting.log")
    sys.stdout = logger
    sys.stderr = sys.stdout

    # Metrics
    metrics = collections.defaultdict(list)

    # Loop over all configurations
    config_path = 'config/GradientBoosting'
    for conf_file in sorted(os.listdir(config_path)):
        # Read config
        config = parse_config(os.path.join(config_path, conf_file))
        metrics = train_sklearn(config, metrics)

    pd.DataFrame(metrics).to_csv("logs/GradientBoosting.csv", index = False)
    logger.close()
