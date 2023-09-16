"""
car_prices.hp_tune.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Contains Functions to perform hyperparameter tuning on RandomForestRegressor.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import DotDict
from src.data import get_data
from src.model import get_pipeline
from src.model import get_transforms


def tune_hp(config_: DotDict) -> dict:
    """
    Function to tune hyperparameters using `GridSearchCV`.


    Parameters
    ----------
    config_: DotDict
        Configuration containing the settings for model, parameters and data.

    Returns
    -------
    dict:
        Dictionary containing the best parameters for the model and given data.
    """

    # Create Hyper parameter grid
    hp_grid = config_.hp_args

    # Get data, transforms and model pipeline
    data_dict = get_data(config_.data)
    feature_transformer, target_transformer = get_transforms(config_.preprocess)
    model = get_pipeline(config_.model, feature_transformer)

    # Fit and transform targets
    target_transformer.fit(data_dict['y_train'].values.reshape(-1, 1))
    data_dict['y_train'] = target_transformer.transform(data_dict['y_train'].values.reshape(-1, 1)).ravel()
    data_dict['y_test'] = target_transformer.transform(data_dict['y_test'].values.reshape(-1, 1)).ravel()

    # Initialize and fit GridSearchCV
    grid_cv = GridSearchCV(
        estimator = model,
        param_grid = hp_grid,
        cv = 5, n_jobs = 1,
        scoring = ['neg_root_mean_squared_error', 'r2'],
        refit = 'r2',
        verbose = 4
    )
    grid_cv.fit(data_dict['x_train'], data_dict['y_train'])
    joblib.dump(grid_cv.best_params_, f"experiments/hp_tuning/best_params.pkl")

    # Get best parameters and create new model using those
    best_params = {k.split("__")[1]: v for k, v in grid_cv.best_params_.items()}
    new_model = RandomForestRegressor(**best_params)
    feature_transformer, target_transformer = get_transforms(config_.preprocess)
    new_model = Pipeline(
        steps = [
                ("preprocessor", feature_transformer),
                ("regressor", new_model)
        ])

    # Fit model and predict on test data.
    new_model.fit(data_dict['x_train'], data_dict['y_train'])
    predictions = new_model.predict(data_dict['x_test'])

    # Create string with test metrics and print
    desc_ = f"""
    Best Model:
        {grid_cv.best_params_}
    Final Test Scores:
        R2 Score  : {r2_score(data_dict['y_test'], predictions)}
        RMSE Score: {np.sqrt(mean_squared_error(data_dict['y_test'], predictions))}
    """
    print(desc_)

    # Return
    return grid_cv.best_params_
