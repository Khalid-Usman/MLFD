import numpy as np
import pandas as pd
from typing import List
from geosketch import gs
from pandas import DataFrame
from visualizer import get_coorelation_heatmap
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def oversampling(X: DataFrame, y: DataFrame, method: str, features_cat: List) -> [DataFrame]:
    """
    This function is used to perform oversampling to the minor class
    :param X: The dependent variables
    :param y: The independent variable
    :param method: SMOTE or SMOTENC
    :param features_cat: The list of categorical features
    :return:
    """
    if method == "SMOTE":
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        y_res = pd.DataFrame(y_res)
        return X_res, y_res
    else:
        smote_nc = SMOTENC(categorical_features=features_cat, random_state=0)
        X_res, y_res = smote_nc.fit_resample(X, y)
        return X_res, y_res


def undersampling(X: DataFrame, y: DataFrame, method: str) -> [DataFrame]:
    """
        This function is used to perform oversampling to the minor class
        :param X: The dependent variables
        :param y: The independent variable
        :param method: GEO-SKETCH or CORESET
        :return:
        """
    if method == "GeoSketch":

        sketch_idx = gs(X=X, N=10, replace=False)
        X = X[sketch_idx]
        return X, y
    else:
        smote_nc = SMOTENC()
        X_res, y_res = smote_nc.fit_resample(X, y)
        return X_res, y_res


def feature_encoding(data: DataFrame, method: str) -> DataFrame:
    """
    This function will pre-process data and remove invalid rows
    :param data: The input dataframe before pre-processing
    :param method: The input method either 'label encoding' or the default 'category encoding'
    :return: The pre-processed dataframe
    """
    # return data
    if method == "One-Hot Encoding":
        onehot_encoder = OneHotEncoder(sparse=False)
        cols = data.columns
        for col in cols:
            data[col] = onehot_encoder.fit_transform(data[col])
        return data
    elif method == "Label Encoding":
        label_encoder = LabelEncoder()
        cols = data.columns
        for col in cols:
            data[col] = label_encoder.fit_transform(data[col])
        return data
    else:
        col_categorical = data.select_dtypes(include=['object', 'category', 'int64']).columns
        for col in col_categorical:
            data[col] = data[col].astype('category')
        data[col_categorical] = data[col_categorical].apply(lambda x: x.cat.codes)
        return data


def remove_duplicate(data: DataFrame, threshold: float):
    """
    This function will calculate correltaion between dependent variables and remove duplicate features having
    correlation more than defined threshold
    :param threshold: The define threshold
    :param data: The input dataframe before correlation
    :return: The dataframe after removing correlated features
    """
    df_corr = data.drop(['isFraud'], axis=1)
    fig_corr_heatmap = get_coorelation_heatmap(data_corr=df_corr)
    corr_matrix = df_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data = data.drop(data[to_drop], axis=1)
    return data, fig_corr_heatmap


def find_outliers_IQR(data: DataFrame):
    """
    This function will identify outliers and remove them from the data
    :param data: The input dataframe
    :return: The output dataframe after removing outliers
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    outliers = data[((data < (q1 - 1.5 * IQR)) | (data > (q3 + 1.5 * IQR)))]
    return outliers


def week_of_month(transaction_date):
    """
    This function will return week of transaction 1st week , 2nd week, 3rd week or 4th week
    :param transaction_date:
    :return:
    """
    first_day = transaction_date.replace(day=1)
    dom = transaction_date.day
    adjusted_dom = dom + first_day.weekday()
    return (adjusted_dom - 1) // 7 + 1


def day_of_week(transaction_date):
    """
    This function will return either the transaction is performed at weekend or working days
    :param transaction_date:
    :return:
    """
    day_of_week = transaction_date.weekday()


def derive_features(data: DataFrame):
    """
    This function will take dataframe as input and derive more features
    :param data: The input dataframe
    :return: It will return the dataframe with some extra features
    """
    data['Transaction since Profile Modified'] = (data['Transaction Date'] -
                                                  data['Debtor Profile Last Modified Date']).dt.days
    data['Transaction since Channel Registered'] = (data['Transaction Date'] -
                                                    data['Debtor Channel Registration Date']).dt.days
    data['Transaction since Channel Activated'] = (data['Transaction Date'] -
                                                   data['Debtor Channel Activation Date']).dt.days
    data['Debtor Profile Modified since Channel Activated'] = (data['Debtor Profile Last Modified Date'] -
                                                               data['Debtor Channel Activation Date']).dt.days
    data['Debtor Channel Activated since Registered'] = (data['Debtor Channel Activation Date'] -
                                                         data['Debtor Channel Registration Date']).dt.days
    data['Debtor Profile Modified since Channel Registered'] = (data['Debtor Profile Last Modified Date'] -
                                                                data['Debtor Channel Registration Date']).dt.days
    data['Transaction since Profile Modified'] = data['Transaction since Profile Modified'].abs()
    data['Transaction since Channel Registered'] = data['Transaction since Channel Registered'].abs()
    data['Transaction since Channel Activated'] = data['Transaction since Channel Activated'].abs()
    data['Debtor Profile Modified since Channel Activated'] = (data['Debtor Profile Modified since Channel Activated']
                                                               .abs())
    data['Debtor Channel Activated since Registered'] = data['Debtor Channel Activated since Registered'].abs()
    data['Debtor Profile Modified since Channel Registered'] = (data['Debtor Profile Modified since Channel Registered']
                                                                .abs())
    data['Week of Month'] = data['Transaction Date'].apply(lambda d: (d.day - 1) // 7 + 1)
    data["Day of Week"] = data['Transaction Date'].dt.dayofweek
    data['Is Weekend'] = data['Day of Week'].apply(lambda x: "Y" if x > 4 else "N")
    # b = [0, 4, 8, 12, 16, 20, 24]
    # l = ['Late Night', 'Early Morning', 'Morning', 'Noon', 'Evening', 'Night']
    # data['SOD'] = pd.cut(data['Time of Txn'], bins=b, labels=l, include_lowest=True).astype(str)
    data = data.drop(['Transaction Date', 'Debtor Profile Last Modified Date', 'Debtor Channel Registration Date',
                      'Debtor Channel Activation Date', 'Day of Week'], axis=1)
    data['isFraud'] = data.pop('isFraud')
    return data


def feature_scaling(data: DataFrame, features: List, method: str):
    """
    This function will take dataframe as input and scale the features to give them equal importance
    :param data: The input dataframe
    :param features: The list of features to scale
    :param method: The method of scaling either MinMax or Standard
    :return: It will return the dataframe where each feature is scaled
    """
    # print(data.dtypes)
    if method == "Standard":
        ss = StandardScaler()
        data[features] = ss.fit_transform(data[features])
        return data
    else:
        mms = MinMaxScaler()
        data[features] = mms.fit_transform(data[features])
        return data
