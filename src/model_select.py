"""
car_prices.model_select.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Functions for selecting the best model from results.
"""


import collections
from glob import glob

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.model import FeedForwardNetwork


def best_validation(file_list: list[str]) -> None:
    """
    Function to check the best model that performs validation data.

    Parameters
    ----------
    file_list: list[str]
        List of files containing metrics.
    """

    # Concatenate metrics.
    metrics = []
    for file in list(file_list):
        metrics.append(pd.read_csv(file))
    metrics = pd.concat(metrics)
    metrics = metrics \
        .sort_values(by = ['valid_rmse', 'valid_r2'], ascending = [True, False]) \
        .reset_index(drop = True)

    # Best model.
    best_model = {
            "model"           : metrics.loc[0, "model"],
            "valid_rmse"      : metrics.loc[0, "valid_rmse"],
            "valid_r2"        : metrics.loc[0, "valid_r2"],
            "num_transform"   : metrics.loc[0, "num_transform"],
            "cat_transform"   : metrics.loc[0, "cat_transform"],
            "target_transform": metrics.loc[0, "target_transform"]
    }

    # Print description string.
    desc_ = f"""The Best Model on Validation Data is:
    Model:
        Type: {best_model["model"]}
        Validation RMSE: {best_model["valid_rmse"]:.5f}
        Validation R-Squared: {best_model["valid_r2"]:.5f}
    Transforms:
        Numerical: {best_model["num_transform"]}
        Categorical: {best_model["cat_transform"]}
        Target: {best_model["target_transform"]}
    """
    print(desc_)


def test_sklearn(exp_path: str, metrics: collections.defaultdict[list]) -> collections.defaultdict[list]:
    """
    Function to test sklearn model and cross-validate.

    Parameters
    ----------
    exp_path: str
        Path to experiment folder
    metrics: collections.defaultdict[list]
        DefaultDict containing the metrics

    Returns
    -------
    metrics: collections.defaultdict[list]
        Updated metrics
    """

    # Load saved files
    data_dict = joblib.load(f"{exp_path}/data_dict.pkl")
    target_transform = joblib.load(f"{exp_path}/TargetTransform.pkl")
    model = joblib.load(f"{exp_path}/model.pkl")

    # Predict
    preds = model.predict(data_dict['x_test'])
    y_true = target_transform.transform(data_dict['y_test'].values.reshape(-1, 1))

    # Log Metrics
    metrics['model'].append(exp_path.split("/")[1])
    metrics['num_transform'].append(exp_path.split("/")[-1].split(".")[0])
    metrics['cat_transform'].append(exp_path.split("/")[-1].split(".")[1])
    metrics['target_transform'].append(exp_path.split("/")[-1].split(".")[2])
    metrics['test_r2'].append(r2_score(y_true, preds))
    metrics['test_rmse'].append(np.mean(mean_squared_error(y_true, preds)))
    return metrics


def test_nn(exp_path: str, metrics: collections.defaultdict[list]) -> collections.defaultdict[list]:
    """
    Function to test torch deep learning model.

    Parameters
    ----------
    exp_path: str
        Path to experiment folder
    metrics: collections.defaultdict[list]
        DefaultDict containing the metrics

    Returns
    -------
    metrics: collections.defaultdict[list]
        Updated metrics

    """
    # Load saved files
    dl_dict = joblib.load(f"{exp_path}/dl_dict.pkl")

    # Load model
    input_shape = next(iter(dl_dict['test_dl']))[0].shape[1]
    weights = torch.load(f"{exp_path}/FeedForwardModel.pkl")
    model = FeedForwardNetwork(input_shape, 1)
    model.load_state_dict(weights)
    model.eval()
    loss_fn = nn.MSELoss()

    # DataLoader Loop
    bar_format = "{desc:>15}({percentage:3.0f}%)|{bar:20}{r_bar}{bar:-10b}"
    test_loop = tqdm(
        dl_dict['test_dl'], unit = 'batch',
        bar_format = bar_format,
        ascii = " >=",
        total = len(dl_dict['test_dl'])
    )
    test_loop.set_description("Testing")

    # Test model
    test_losses, test_r2_scores = [], []
    for x, y in test_loop:
        # Forward
        prediction = model(x)
        loss = loss_fn(prediction.ravel(), y)
        loss = torch.sqrt(loss)

        # Metrics
        test_r2_scores.append(
            r2_score(
                y.detach().numpy().ravel(),
                prediction.detach().numpy().ravel()
            ))

        # Update loop
        test_losses.append(loss.item())
        test_loop.set_postfix(rmse = np.mean(test_losses), r2_score = np.mean(test_r2_scores))

    # Update metrics
    metrics['model'].append(exp_path.split("/")[1])
    metrics['test_r2'].append(np.mean(test_r2_scores))
    metrics['test_rmse'].append(np.mean(test_losses))
    metrics['num_transform'].append(exp_path.split("/")[-1].split(".")[0])
    metrics['cat_transform'].append(exp_path.split("/")[-1].split(".")[1])
    metrics['target_transform'].append(exp_path.split("/")[-1].split(".")[2])

    # Return
    return metrics


def test_model(exp_list: list[str]) -> pd.DataFrame:
    """
    Function to test all models

    Parameters
    ----------
    exp_list: list[str]
        List of experiments and their paths.

    Returns
    -------
    pd.DataFrame:
        DataFrame containing test metrics for all amodels.
    """

    # Initialize metrics
    metrics = collections.defaultdict(list)
    for exp_path in glob(exp_list):
        if "hp_tuning" in exp_path:
            continue
        if "FeedForward" in exp_path:
            metrics = test_nn(exp_path, metrics)
        else:
            metrics = test_sklearn(exp_path, metrics)

    # Return
    return pd.DataFrame(metrics)


def best_test(metrics: pd.DataFrame) -> None:
    """
    Function to check the best model that performs validation data.

    Parameters
    ----------
    metrics: collections.defaultdict[list]
        DataFrame containing test metrics
    """

    # Sort DataFrame
    metrics = metrics \
        .sort_values(by = ['test_rmse', 'test_r2'], ascending = [True, False]) \
        .reset_index(drop = True)

    # Best Model
    best_model = {
            "model"           : metrics.loc[0, "model"],
            "test_rmse"       : metrics.loc[0, "test_rmse"],
            "test_r2"         : metrics.loc[0, "test_r2"],
            "num_transform"   : metrics.loc[0, "num_transform"],
            "cat_transform"   : metrics.loc[0, "cat_transform"],
            "target_transform": metrics.loc[0, "target_transform"]
    }

    # Description String
    desc_ = f"""The Best Model on Testing Data is:
       Model:
           Type: {best_model["model"]}
           Testing RMSE: {best_model["test_rmse"]:.5f}
           Testing R-Squared: {best_model["test_r2"]:.5f}
       Transforms:
           Numerical: {best_model["num_transform"]}
           Categorical: {best_model["cat_transform"]}
           Target: {best_model["target_transform"]}
       """
    print(desc_)
