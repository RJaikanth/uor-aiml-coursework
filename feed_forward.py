import collections
import os.path
import sys

import pandas as pd

from src.config import parse_config
from src.train import train_feedforward
from src.utils import Logger

if __name__ == '__main__':
    # Set sys.stdout
    os.makedirs("./logs/", exist_ok = True)
    logger = Logger("logs/FeedForward.log")
    sys.stdout = logger
    sys.stderr = sys.stdout

    # Metrics
    metrics = collections.defaultdict(list)

    # Read config
    config_path = "config/FeedForward/"
    for file in sorted(os.listdir(config_path)):
        config = parse_config(os.path.join(config_path, file))
        metrics = train_feedforward(config, metrics)
        num_epochs = config.model.num_epochs

    # Keep only metrics of last epoch
    metrics = pd.DataFrame(metrics)
    metrics = metrics.loc[metrics['epoch'] == num_epochs]
    metrics = metrics.drop('epoch', axis = 1)
    metrics = metrics.reset_index(drop = True)
    pd.DataFrame(metrics).to_csv("logs/FeedForward.csv", index = False)
    logger.close()
