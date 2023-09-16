"""
car_prices.data.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Functions to read and handle data.
"""


import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.config import DotDict
from src.model import get_transforms


def read_data(conf: DotDict) -> pd.DataFrame:
    """
    Read CSV data using data configuration.

    | Config contains parameters for
    | - dtypes
    | - columns to drop
    | - date columns

    Parameters
    ----------
    conf: DotDict
       Data configuration for the project

    Returns
    -------
    df_: pd.DataFrame
       DataFrame object containing the raw dataset.

    """
    df_ = pd.read_csv(
        conf.path,  # Path to data file
        dtype = dict(conf.dtypes),  # Dtypes of columns
        usecols = lambda x: x not in conf.drop,  # Columns to drop
        parse_dates = conf.date,  # Date columns
    )

    # Return
    return df_


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe obtained by calling `read_data`.

    Categorical Data
    ================
    | - Use only `private` sellers.
    | - Use only `Angebot` offers.
    | - Fuel Type should be in `[lpg, benzin, diesel]`.
    | - Use for cars registered between 1995 and 2020.
    | - Remove rows where `monthOfRegistration` is 0.
    | - Keep `models` whose count is more than 1500.
    | - Keep `brand` whose count is more than 1500.

    Numerical Data
    ==============
    | - Keep ads for which the car has been driven more than 20000 kilometers.
    | - Keep ads for which the powerPS is less than 500.
    | - Keep rows where price is more than 1000 and less than 40000

    Add Features
    ============
    | - Add a new feature `age_of_car` obtained by calculating the difference between registration and ad publishing
    date

    Drop Columns
    ============
    | Drop the following columns no longer needed
    | - `dateCreated`
    | - `yearOfRegistration`
    | - `monthOfRegistration`
    | - `dateOfRegistration`
    | - `seller`
    | - `offerType`
    | - `postalCode`

    Drop Missing
    ============
    Drop all missing values

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing raw data

    Returns
    -------
    df: pd.DataFrame
        Returns the clean DataFrame

    """

    # Clean Categorical
    df = df.loc[df['seller'] == 'privat', :]
    df = df.loc[df['offerType'] == 'Angebot', :]
    df = df.loc[df['fuelType'].isin(['lpg', 'benzin', 'diesel']), :]
    df = df.loc[df['yearOfRegistration'].between(1995, 2020, 'both'), :]
    df = df.loc[df['monthOfRegistration'] != '0', :]
    df = df.groupby('model').filter(lambda x: len(x) > 1500)
    df = df.groupby('brand').filter(lambda x: len(x) > 1500)

    # Clean Numerical
    df = df.loc[df['kilometer'] > 20000, :]
    df = df.loc[df['powerPS'] < 500, :]
    df = df.loc[(df['price'] > 1000) & (df['price'] < 40000), :]

    # Add Features
    df['yearOfRegistration'] = df['yearOfRegistration'].astype(int)
    df['dateOfRegistration'] = pd.to_datetime(
        df['yearOfRegistration'].astype(str) + '-' +
        df['monthOfRegistration'].astype(str) + '-01'
    )
    df["age_of_car"] = df['dateCreated'] - pd.to_datetime(df['dateOfRegistration'])
    df['age_of_car'] = df['age_of_car'].dt.days

    # Drop columns
    cols_ = [
            "dateCreated", "yearOfRegistration",
            "monthOfRegistration", "dateOfRegistration",
            "seller", "offerType", "postalCode"
    ]
    df = df.drop(cols_, axis = 1)

    # Drop missing values
    df = df.dropna(axis = 0)
    df.reset_index(inplace = True, drop = True)

    return df


def split_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into features and targets

    Parameters
    ----------
    df: pd.DataFrame
        Cleaned DataFrame containing both features and target columns.

    Returns
    -------
    x, y: tuple[pd.DataFrame, pd.DataFrame]:
        | x: pd.DataFrame
        |   DataFrame containing features for Machine Learning Model
        | y: pd.DataFrame
        |   DataFrame containing target for Machine Learning Model

    """

    y = df['price']  # Target
    x = df.drop(['price'], axis = 1)  # Price
    return x, y  # Return


def preprocess(data_conf: DotDict) -> pd.DataFrame:
    """
    Single Function to call both `read_data` and `clean_data`.

    Parameters
    ----------
    data_conf: DotDict
        Configuration for handling data file.

    Returns
    -------
    df_: pd.DataFrame
        Cleaned DataFrame.
    """

    df_ = read_data(data_conf.read)  # Read Data
    df_ = clean_data(df_)  # Clean Data
    return df_


def get_data(data_conf: DotDict) -> dict[str, pd.DataFrame]:
    """
    Function to get data in a dictionary format for easy use.

    Parameters
    ----------
    data_conf: DotDict
        Configuration containing the settings for handling data.

    Returns
    -------
    dict[str, pd.DataFrame]
        | Contains the following
        | x_train: Training Feature Dataset
        | y_train: Training Target Dataset
        | x_test: Testing Feature Dataset
        | y_test: Testing Target Dataset

    """

    # Preprocess data
    df_ = preprocess(data_conf)

    # Train Test data
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_, train_size = 0.8, shuffle = False)

    # Get feature and transform feature
    x_train, y_train = split_cols(df_train)
    x_test, y_test = split_cols(df_test)

    return {
            "x_train": x_train,
            "y_train": y_train,
            "y_test" : y_test,
            "x_test" : x_test,
    }


def get_dl(
        data_dict: dict[str, pd.DataFrame],
        preprocess_conf: DotDict,
        exp_path: str
) -> tuple[tuple[int], dict[str, DataLoader]]:
    """
    Function to get data in a dictionary format for easy use.

    Parameters
    ----------
    data_dict: dict[str, pd.DataFrame]
        Dictionary containing DataFrames for training and testing.
    preprocess_conf: DotDict
        Configuration containing the settings for preprocessing the data.
    exp_path: str
        Path to experiment folder.

    Returns
    -------
    tuple[tuple[int], dict[str, DataLoader]]
        | input_size: tuple[int]
        |   Input Size for the feedforward model
        | dl_dict: dict[str, DataLoader]
        |   Contains the following
        |   x_train: Training Feature Dataset
        |   y_train: Training Target Dataset
        |   x_test: Testing Feature Dataset
        |   y_test: Testing Target Dataset

    """

    # Create transformers
    print("Creating Target and Feature Transformers...")
    feature_transformer, target_transformer = get_transforms(preprocess_conf)

    # Split train further
    print("Splitting train into train and validation...")
    x_train, x_valid, y_train, y_valid = train_test_split(
        data_dict['x_train'], data_dict['y_train'], train_size = 0.8, shuffle = False)
    data_dict['x_train'] = x_train
    data_dict['y_train'] = y_train
    data_dict['x_valid'] = x_valid
    data_dict['y_valid'] = y_valid

    # Fit transformers
    print("Fitting and saving transformers on train data")
    feature_transformer.fit(data_dict['x_train'])
    target_transformer.fit(data_dict['y_train'].values.reshape(-1, 1))
    joblib.dump(target_transformer, f"{exp_path}/TargetTransform.pkl")
    joblib.dump(feature_transformer, f"{exp_path}/FeatureTransform.pkl")
    print(f"Feature Transformer Saved at {exp_path}/FeatureTransform.pkl")
    print(f"Target Scaler Saved at {exp_path}/TargetTransform.pkl")

    # Transform data
    print("Transforming Data...")
    for ds in data_dict.keys():
        if 'x' in ds:
            data_dict[ds] = feature_transformer.transform(data_dict[ds])
        else:
            data_dict[ds] = target_transformer.transform(data_dict[ds].values.reshape(-1, 1)).ravel()
        data_dict[ds] = torch.Tensor(np.array(data_dict[ds], dtype = np.float32))
    input_size = data_dict['x_train'].shape[1]

    # Create DataLoaders
    print("Creating DataLoaders...")
    dl_dict = dict(
        train_dl = DataLoader(
            TensorDataset(data_dict['x_train'], data_dict['y_train']), shuffle = True, batch_size = 64),
        valid_dl = DataLoader(
            TensorDataset(data_dict['x_valid'], data_dict['y_valid']), shuffle = False, batch_size = 32),
        test_dl = DataLoader(TensorDataset(data_dict['x_test'], data_dict['y_test']), shuffle = False, batch_size = 32)
    )

    return input_size, dl_dict
