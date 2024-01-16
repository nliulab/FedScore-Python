from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_dat_csv(file_name, directory, variable_list):
    file_path = os.path.join(directory, file_name)
    data = pd.read_csv(file_path)
    X_features = []
    for col in data.columns:
        for var in variable_list:
            if var == '_'.join(col.split('_')[:-1]):
                X_features.append(col)
    print('===================')
    print(X_features)
    print('===================')
    y = data.loc[:, 'label']
    X = data.loc[: , X_features]
    return X, y, len(X_features)


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes, n_features):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    print(">>>>>>>>>>>>> setting initial parameters")
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    print(model.coef_)
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
