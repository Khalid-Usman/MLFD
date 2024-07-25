import numpy as np
from typing import List
from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


def feature_selector(X: DataFrame, y: List, method: str):
    """
    This function will apply Genetic Algorithm to select features
    :param X: The input dataframe with independent variables
    :param y: The input dependent variable
    :param method: The name of method to select important features
    :return:
    """
    features = X.columns
    if method == "RFE":
        estimator = DecisionTreeClassifier()
        rfe = RFE(estimator, n_features_to_select=6, step=1)
        rfe.fit(X, y)
        return rfe.support_, rfe.ranking_
    else:
        rf = RandomForestRegressor(random_state=0)
        rf = rf.fit(X, y)
        sorted_idx = np.argsort(rf.feature_importances_)[::-1]
        return features[sorted_idx].tolist(), rf.feature_importances_[sorted_idx].tolist()