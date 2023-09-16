"""
car_prices.config.py
Author: Raghhuveer Jaikanth
Date  : 08/03/2023

Functions to parse configurations for the project.

"""

import os

import numpy as np
import yaml


class DotDict(dict):
    """
    Class to access dictionary values using dot notations.
    Nested dictionaries are also converted to DotDict type.
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()

        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value


def read_yaml(file_path: str) -> DotDict:
    """
    Function to read a yaml file with additional YAML preset for a pythonic tuple.

    Parameters
    ----------
    file_path: str
        Path to config file.

    Returns
    -------
    conf: DotDict
        DotDict containing the key-value pairs of the YAML file.

    """

    # Create Safe Loader and add python tuple constructor
    loader = yaml.SafeLoader
    loader.add_constructor(
        u'tag:yaml.org,2002:python/tuple',
        yaml.Loader.construct_python_tuple
    )

    # Open file and read
    with open(file_path, 'r') as f:
        conf = yaml.load(f, loader)

    # Return
    return DotDict(conf)


def parse_config(file_path: str) -> DotDict:
    """
    Function to additionally parse initial read of yaml file.

    The config file should contain path to data and preprocessing configurations.
    This function further parses the configurations for the data loading and preprocessing steps.

    This function calls the `read_yaml` function internally.
    Recommended to directly call this function.

    Parameters
    ----------
    file_path: str
        Path to config file.

    Returns
    -------
    conf: DotDict
        DotDict containing the key-value pairs of the YAML file.
    """

    # Read config file
    conf = read_yaml(file_path)

    # Read data config
    conf.data = read_yaml(conf.data)

    # Numerical config
    conf.preprocess.numerical = {**read_yaml(conf.preprocess.numerical.conf_path), **conf.preprocess.numerical}
    conf.preprocess.numerical.pop('conf_path')

    # Categorical config
    conf.preprocess.categorical = {**read_yaml(conf.preprocess.categorical.conf_path), **conf.preprocess.categorical}
    conf.preprocess.categorical.pop('conf_path')

    # Target config
    conf.preprocess.target = read_yaml(conf.preprocess.target.conf_path)

    # Return
    return DotDict(conf)


def parse_hp_config(file_path):
    """
    Function to parse the hyperparameter tuning configurations from a YAML file.

    In the config file, hyperparamter grid is defined using tuples. This function further converts them into numpy arrays
    using `np.arange`.

    This function calls the `read_yaml` function internally.
    Recommended to directly call this function.

    Parameters
    ----------
    file_path: str
        Path to config file.

    Returns
    -------
    conf: DotDict
        DotDict containing the key-value pairs of the YAML file.
    """

    # Read the config file
    conf = read_yaml(file_path)

    # Parse the experiment config file and append to the hyperparameter config.
    conf = {**conf, **parse_config(conf['experiment_file'])}
    conf.pop('experiment_file')

    # Convert tuple to range of numbers.
    for k, v in conf['hp_args'].items():
        conf['hp_args'][k] = np.arange(*conf['hp_args'][k])

    # Return
    return DotDict(conf)


def get_metadata(config: DotDict) -> tuple[str, str]:
    """
    Function to extract the experiment metadata from the loaded config.

    Prints the experiment details to stdout. Also creates an experiment folder to save models for inference.

    Parameters
    ----------
    config: DotDict
        DotDict object containing configuration for the current experiment

    Returns
    -------
    exp_path, exp_name: [str, str]
        | exp_path: Path to experiment folder. <br>
        | exp_name: Name of the experiment.
    """

    # Experiment Metadata.
    numerical_preprocess = config.preprocess.numerical.name
    categorical_preprocess = config.preprocess.categorical.name
    target_preprocess = config.preprocess.target.name

    # Print the experiment details to stdout.
    print("--" * 40)
    print(f"Model                 : {config.model.name}")
    print(f"Numerical Transform   : {numerical_preprocess}")
    print(f"Categorical Transform : {categorical_preprocess}")
    print(f"Target Transform      : {target_preprocess}")
    print("--" * 40)

    # Create an experiment folder to save the models.
    exp_name = f"{numerical_preprocess}.{categorical_preprocess}.{target_preprocess}"
    exp_path = os.path.join("experiments", config.model.name, exp_name)
    os.makedirs(exp_path, exist_ok = True)

    # Return
    return exp_path, exp_name
