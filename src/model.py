"""
car_prices.model.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Source file containing functions for creating transforms, models and pipelines.
"""

import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from src.config import DotDict


class FeedForwardNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        FeedForward Neural Network Class.

        Takes 2 inputs - Number of inputs and number of outputs.


        Parameters
        ----------
        num_inputs: int
            Input size of data
        num_outputs: int
            Number of target outputs
        """
        super().__init__()

        self.model = nn.Sequential(*[
                nn.Linear(num_inputs, 50),
                nn.ReLU(),
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, num_outputs),
        ])

    def forward(self, x: input):
        return self.model(x)


# Model Dictionary
model_dict = {
        "RandomForest"    : RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "FeedForward"     : FeedForwardNetwork
}

# Preprocessing Dictionary
preprocess_dict = {
        "RobustScaler"   : RobustScaler,
        "StandardScaler" : StandardScaler,
        "OneHotEncoding" : OneHotEncoder,
        "OrdinalEncoding": OrdinalEncoder
}


def get_model(model_conf: DotDict) -> tuple[BaseEstimator, nn.Module]:
    """
    Function to get model from model_dict using the kwargs from model_conf.

    Parameters
    ----------
    model_conf: DotDict
        DotDict containing configuration for the models.

    Returns
    -------
    tuple[BaseEstimator, nn.Module]
        Model to be trained and used.

    """
    return model_dict[model_conf.name](**model_conf.kwargs)


def get_pipeline(model_conf: DotDict, feature_transformer: ColumnTransformer) -> Pipeline:
    """
    Function to create the pipeline from the model and feature transforms.

    Parameters
    ----------
    model_conf: DotDict
        DotDict containing configuration for the models.
    feature_transformer: ColumnTransformer
        ColumnTransformer object containing the steps for column transformers.

    Returns
    -------
    Pipeline
        Pipeline containing the feature transforms and model.
    """
    return Pipeline(
        steps = [
                ('preprocessor', feature_transformer),
                ('regressor', get_model(model_conf))
        ])


def get_transforms(preprocess_conf: DotDict) -> tuple[ColumnTransformer, BaseEstimator]:
    """
    Function to create the ColumnTransformer and Target Transform for the preprocessing.

    Parameters
    ----------
    preprocess_conf: DotDict
        DotDict containing configuration for the preprocessing.

    Returns
    -------
    tuple[ColumnTransformer, BaseEstimator]
    feature_transformer: ColumnTransformer
        ColumnTransformer containing the feature transforms.
    target_transformer: BaseEstimator
        Transformer for the target column.
    """

    # Scaler and encoder
    scaler = preprocess_dict[preprocess_conf.numerical.name](**preprocess_conf.numerical.kwargs)
    encoder = preprocess_dict[preprocess_conf.categorical.name](**preprocess_conf.categorical.kwargs)

    # Column Transformer
    feature_transformer = ColumnTransformer(
        transformers = [
                ('numerical_transform', scaler, preprocess_conf.numerical.cols),
                ('categorical_transform', encoder, preprocess_conf.categorical.cols)
        ])

    # Target Transformer
    target_transformer = preprocess_dict[preprocess_conf.numerical.name](**preprocess_conf.numerical.kwargs)

    # Return
    return feature_transformer, target_transformer
