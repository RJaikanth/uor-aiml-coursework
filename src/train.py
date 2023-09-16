"""
car_prices.train.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Functions to train sklearn and torch models.
"""

import collections

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from tqdm import tqdm

from src.config import DotDict
from src.config import get_metadata
from src.data import get_data
from src.data import get_dl
from src.model import get_model
from src.model import get_pipeline
from src.model import get_transforms


def log_metadata(metrics: collections.defaultdict[list], config_: DotDict) -> collections.defaultdict[list]:
    """
    Record metadata of the experiment.

    Parameters
    ----------
    metrics: collections.defaultdict[list]
        DefaultDict to store experiment details.
    config_: DotDict
        Configuration for the experiment.

    Returns
    -------
    metrics: collections.defaultdict[list]
        Updated metrics DefaultDict

    """

    # Log metrics
    metrics['model'].append(f"{config_.model.name}")
    metrics['num_transform'].append(f"{config_.preprocess.numerical.name}")
    metrics['cat_transform'].append(f"{config_.preprocess.categorical.name}")
    metrics['target_transform'].append(f"{config_.preprocess.target.name}")

    return metrics


def train_sklearn(config_: DotDict, metrics: collections.defaultdict[list]) -> collections.defaultdict[list]:
    """
    Function to train sklearn models. Performs cross-validation as well.

    Parameters
    ----------
    config_: DotDict
        Configuration for experiment.
    metrics: collections.defaultdict[list]
        DefaultDict to store experiment metrics.
    Returns
    -------
    metrics: collections.defaultdict[list]
        Updated metrics default dict.

    """

    # Metadata and Logging
    exp_path, exp_name = get_metadata(config_)

    # Get data
    print("Fetching Data...")
    data_dict = get_data(config_.data)
    joblib.dump(data_dict, f"{exp_path}/data_dict.pkl")
    print(f"Data Saved at {exp_path}/data_dict.pkl")

    # Transforms and Pipeline
    print("Creating Transforms and Model Pipeline...")
    feature_transformer, target_transformer = get_transforms(config_.preprocess)
    model = get_pipeline(config_.model, feature_transformer)
    print("Model and Transformers are ready!")

    # Transform target
    print("Transforming Target Variable...")
    target_transformer.fit(data_dict['y_train'].values.reshape(-1, 1))
    data_dict['y_train'] = target_transformer.transform(data_dict['y_train'].values.reshape(-1, 1)).ravel()
    joblib.dump(target_transformer, f"{exp_path}/TargetTransform.pkl")
    print(f"Target Scaler Saved at {exp_path}/TargetTransform.pkl")

    # Fit model
    print("Fitting and Evaluating model on train data...")
    model.fit(data_dict['x_train'], data_dict['y_train'])
    predictions = model.predict(data_dict['x_train'])
    metrics['train_rmse'].append(np.sqrt(mean_squared_error(data_dict['y_train'], predictions)))
    metrics['train_r2'].append(r2_score(data_dict['y_train'], predictions))
    joblib.dump(model, f"{exp_path}/model.pkl")
    print(f"Model Pipeline saved at {exp_path}/TargetTransform.pkl")

    # Cross Validation
    cv = config_.data.split.n_splits
    print("Cross Validating model with {} splits...".format(cv))
    model = get_pipeline(config_.model, feature_transformer)
    scores = cross_validate(
        model,
        data_dict['x_train'], data_dict['y_train'],
        cv = cv, scoring = ['neg_root_mean_squared_error', 'r2'],
        n_jobs = 6
    )
    joblib.dump(scores, f"{exp_path}/cv_scores.pkl")

    # Update metrics
    metrics['valid_rmse'].append(np.mean(np.abs(scores['test_neg_root_mean_squared_error'])))
    metrics['valid_r2'].append(np.mean(scores['test_r2']))
    print("Cross Validation Complete!")

    # Log additional data
    print("Logging addition data...")
    metrics = log_metadata(metrics, config_)

    print("Experiment Complete!")
    print("==" * 40, "\n")

    return metrics


def train_feedforward(config_: DotDict, metrics: collections.defaultdict[list]) -> collections.defaultdict[list]:
    """
    Function to train sklearn models. Performs cross-validation as well.

    Parameters
    ----------
    config_: DotDict
        Configuration for experiment.
    metrics: collections.defaultdict[list]
        DefaultDict to store experiment metrics.
    Returns
    -------
    metrics: collections.defaultdict[list]
        Updated metrics default dict.

    """

    # Metadata and Logging
    exp_path, exp_name = get_metadata(config_)

    # Fetch dataloaders
    print("Fetching data...")
    data_dict = get_data(config_.data)
    input_size, dl_dict = get_dl(data_dict, config_.preprocess, exp_path)
    joblib.dump(dl_dict, f"{exp_path}/dl_dict.pkl")
    print(f"DataLoaders saved at {exp_path}/dl_dict.pkl")

    # Model
    print("Creating Model, Optimizer and, Loss Function...")
    config_.model.kwargs['num_inputs'] = input_size
    model = get_model(config_.model)
    opt = optim.SGD(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    print("Model, Loss Function and, Optimizer are initialized!")

    # Train model
    bar_format = "{desc:>15}({percentage:3.0f}%)|{bar:20}{r_bar}{bar:-10b}"
    print("Fitting and Evaluating model on train data...")
    for epoch in range(config_.model.num_epochs):
        # Training Loop
        print("Initialise Training Loop")
        train_loop = tqdm(
            dl_dict['train_dl'], unit = 'batch',
            bar_format = bar_format,
            ascii = " >=",
            total = len(dl_dict['train_dl'])
        )
        train_loop.set_description(f"Epoch: {epoch + 1}, Training")

        # Training
        train_losses, train_r2_scores = [], []
        for x, y in train_loop:
            model.train()
            opt.zero_grad()

            # Forward pass
            prediction = model(x)
            loss = loss_fn(prediction.ravel(), y)
            loss = torch.sqrt(loss)

            # Backward Pass
            loss.backward()
            opt.step()

            # Metrics
            model.eval()
            train_r2_scores.append(
                r2_score(
                    y.detach().numpy().ravel(),
                    prediction.detach().numpy().ravel()
                ))
            train_losses.append(loss.item())
            train_loop.set_postfix(rmse = np.mean(train_losses), r2_score = np.mean(train_r2_scores))

        # Validation Loop
        print("Initialise Validation Loop")
        valid_loop = tqdm(
            dl_dict['valid_dl'], unit = 'batch', bar_format = bar_format, ascii = " >=",
            total = len(dl_dict['valid_dl']))
        valid_loop.set_description(f"Epoch: {epoch + 1}, Validation")
        valid_losses, valid_r2_scores = [], []

        # Validation
        for x, y in valid_loop:
            # Forward
            model.eval()
            prediction = model(x)
            loss = loss_fn(prediction.ravel(), y)
            loss = torch.sqrt(loss)

            # Metrics
            valid_r2_scores.append(
                r2_score(
                    y.detach().numpy().ravel(),
                    prediction.detach().numpy().ravel()
                ))
            valid_losses.append(loss.item())
            valid_loop.set_postfix(rmse = np.mean(valid_losses), r2_score = np.mean(valid_r2_scores))

        # Log metrics
        metrics['train_r2'].append(np.mean(train_r2_scores))
        metrics['train_rmse'].append(np.mean(train_losses))
        metrics['valid_r2'].append(np.mean(valid_r2_scores))
        metrics['valid_rmse'].append(np.mean(valid_losses))
        metrics['epoch'].append(epoch + 1)
        metrics = log_metadata(metrics, config_)

    # Save model
    print("Saving Model...")
    torch.save(model.state_dict(), f"{exp_path}/FeedForwardModel.pkl")
    print(f"Model saved at {exp_path}/FeedForwardModel.pkl")

    return metrics
