import os
import catboost
import numpy as np
import pandas as pd
import plotly.io as io
from sklearn import tree
from pandas import DataFrame
from models import model_GLM
from matplotlib import pyplot as plt
from training import run_pipeline, run_model
from feature_selection import feature_selector
from util.data_imputation import DataFrameImputer
from sklearn.model_selection import train_test_split
from FrankWolfeCoreset import FrankWolfeCoreset as FWC
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from preprocessor import feature_encoding, oversampling, remove_duplicate, feature_scaling, derive_features
from visualizer import compute_count, get_box_plot, get_bar_chart, show_bar_plot, show_histogram_plot

import copy
import time
import numpy as np
import FrankWolfeCoreset as FWC
from scipy.linalg import null_space


SMALL_NUMBER = 1e-7


def checkIfPointsAreBetween(p, start_line, end_line):
    return (np.all(p <= end_line) and np.all(p >= start_line)) or (np.all(p <= start_line) and np.all(p >= end_line))


def checkIfPointOnSegment(p, start_line, end_line, Y):
    if checkIfPointsAreBetween(p, start_line, end_line):
        _, D, V = np.linalg.svd(np.vstack((np.zeros(start_line.shape), end_line - start_line)))
        if np.linalg.norm(np.dot(p - start_line, Y)) < 1e-11:
            return True
    return False


def check_file_exists(path: str, name: str):
    """
    This function will check the path of csv, if it does not exist then raise an error
    :param path: path of file contains cnic
    :param name: name of file contains cnic
    :return:
    """
    if not os.path.exists(path):
        raise FileNotFoundError("{name} does not exist!".format(name=name))


def convert_features(df: DataFrame) -> DataFrame:
    """
    This function will convert integer features into categorical features
    :param df: Input Dataframe
    :return: Output Dataframe
    """
    # df = df[['Debtor Age', 'Debtor Gender', 'Transaction Amount', 'Transaction Type', 'Week of Month',
    #          'Debtor Account/Wallet Opened', 'Transaction since Profile Modified',
    #          'Debtor Profile Modified since Channel Activated', 'Transaction since Channel Registered',
    #          'Count of added new Beneficiaries in the last 24 hours', 'isFraud']]

    bins = [10, 40, 60, 100]
    labels = ['Young', 'Middle Age', 'Old']
    df['Debtor Age'] = pd.cut(df['Debtor Age'], bins=bins, labels=labels, right=False)

    occupation_dict = {"HBL'S OFFICE SUPPORT STAFF": 'OTHERS', 'OTHER': 'OTHERS',
                       'GOVERNMENT SERVICE': 'GOVERNMENT SERVANT',
                       'BUSINESSMAN': 'BUSINESS MAN', 'UN-EMPLYED': 'UNEMPLOYED',
                       'SELF EMPLOYED PROFESSIONAL': 'SELF EMPLOYED'}
    occupation_list = occupation_dict.keys()
    df['Debtor Occupation'] = df['Debtor Occupation'].apply(lambda x: occupation_dict[x] if x in occupation_list else x)

    '''
    occupation_dict = {'SALARIED PERSON': 'SALARIED PERSON', 'STUDENT': 'STUDENT', 'SELF EMPLOYED': 'SELF EMPLOYED',
                       'OTHERS': 'OTHERS', 'FREELANCER': 'SELF EMPLOYED', 'BUSINESS MAN': 'BUSINESS MAN',
                       'LABOURS': 'SELF EMPLOYED', 'PENSIONER': 'GOVERNMENT SERVANT', 'ZAMINDAR': 'SELF EMPLOYED',
                       'UNEMPLOYED': 'UNEMPLOYED', 'GOVERNMENT SERVANT': 'GOVERNMENT SERVANT',
                       'HOUSE WIFE': 'HOUSE WIFE', 'TEACHERS': 'SALARIED PERSON', 'DOCTOR': 'SALARIED PERSON',
                       'BANKER': 'SALARIED PERSON', 'DEFENCE PERSONAL': 'GOVERNMENT SERVANT',
                       'LAGESLATORS': 'GOVERNMENT SERVANT', 'PRIVATE SERVICE': 'PRIVATE SERVICE',
                       'JOURNALIST': 'SELF EMPLOYED', 'CHARTERED ACCOUNTANT': 'SALARIED PERSON',
                       'RETIRED': 'GOVERNMENT SERVANT', 'PROFESSORS': 'GOVERNMENT SERVANT',
                       'ENGINEER': 'GOVERNMENT SERVANT', 'DIPLOMAT': 'GOVERNMENT SERVANT',
                       'CONTRACTORS': 'SELF EMPLOYED', 'MUSTAHQEEN-E-ZAKAT': 'GOVERNMENT SERVANT',
                       'GUARDS-THIRD PARTY EMPLOYEES': 'GOVERNMENT SERVANT', 'BUSINESS WOMAN': 'BUSINESS WOMAN',
                       'RESEARCHERS': 'SELF EMPLOYED', 'ARTIST': 'SELF EMPLOYED', 'SCIENTIST': 'SELF EMPLOYED',
                       'TRANSPORTERS': 'SELF EMPLOYED', 'AGRICULTURE': 'SELF EMPLOYED', 'SALARIED': 'SALARIED',
                       'LAWYER': 'SELF EMPLOYED'}
    occupation_list = occupation_dict.keys()
    df['Debtor Occupation'] = df['Debtor Occupation'].apply(lambda x: occupation_dict[x] if x in occupation_list else x)
    '''

    device_dict = {'IPHONE': 'APPLE'}
    device_list = device_dict.keys()
    df['Transaction Device'] = df['Transaction Device'].apply(lambda x: device_dict[x] if x in device_list else x)

    bins = [0, 1, 10000, 30000, 50000, 100000, 1000000, 1000000000]
    labels = ['1', '<10K', '<30K', '<50K', '<100K', '<1000K', '>1000K']
    df['Transaction Amount'] = pd.cut(df['Transaction Amount'], bins=bins, labels=labels, right=False)

    bins = [0, 1, 7, 30, 120, 360000]
    labels = ['Same Day', 'Same Week', 'Same Month', 'Same Quarter', 'Same Year']
    df['Transaction since Profile Modified'] = pd.cut(df['Transaction since Profile Modified'], bins=bins,
                                                      labels=labels, right=False)

    bins = [0, 1, 2, 10, 50, 10000]
    labels = ['0', '1', '<10', '<50', '>50']
    df['Count of added new Beneficiaries in the last 24 hours'] = pd.cut(df['Count of added new Beneficiaries in '
                                                                            'the last 24 hours'], bins=bins,
                                                                         labels=labels, right=False)

    type_dict = {'INTRA-BANK': 'INTRA-BANK', 'IBFT': 'IBFT', 'FUNDS TRANSFER': 'IBFT',
                 'INTER BANK FUNDS TRANSFER': 'IBFT', 'INTER BANK FUNDS TRANSFER RAAST': 'RAAST',
                 'RAAST': 'RAAST', 'MOBILETOPUP': 'MOBILETOPUP', 'MOBILE TOP-UP': 'MOBILETOPUP'}
    type_list = type_dict.keys()
    df['Transaction Type'] = df['Transaction Type'].apply(lambda x: type_dict[x] if x in type_list else 'OTHER')

    # df = df.drop(['Debtor Age', 'Transaction Amount', 'Transaction since Profile Modified',
    #               'Transaction since Channel Registered', 'Debtor Profile Modified since Channel Activated',
    #               'Count of added new Beneficiaries in the last 24 hours'], axis=1)
    return df


def clean_data(data: DataFrame) -> DataFrame:
    """
    This function will clean the dataframe
    :param data:
    :return:
    """
    # data.rename(columns={'fraud': 'isFraud'}, inplace=True)
    data = data.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    data['Debtor Profile Last Modified Date'] = pd.to_datetime(data['Debtor Profile Last Modified Date'])
    data['Debtor Channel Registration Date'] = pd.to_datetime(data['Debtor Channel Registration Date'])
    data['Debtor Channel Activation Date'] = pd.to_datetime(data['Debtor Channel Activation Date'])
    data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
    data = data.dropna(subset=['Debtor Profile Last Modified Date'])
    data = data.loc[data['Transaction Amount'] > 0, :]
    data = data.loc[data['Debtor Age'] < 150, :]
    print(data.info())
    return data


def load_data(file_path: str) -> DataFrame:
    """
    This function will load the data into pandas dataframe
    :param file_path: The path of input data file
    :return: The output dataframe
    """
    check_file_exists(file_path, "Input File")
    if '.csv' in file_path:
        data = pd.read_csv(file_path, sep=',')
    else:
        data = pd.read_excel(file_path)
    print(data.info())
    return data


def attainCoresetByDanV2(P, u, eps, beta):
    ts = time.time()
    if u.ndim < 2:
        u = u[:, np.newaxis]
    E_u = np.sum(np.multiply(P, u), axis=0)
    x = np.sum(np.multiply(u.flatten(), np.sqrt(np.sum((P - E_u) ** 2, axis=1))))
    lifted_P = np.hstack((P - E_u, x * np.ones((P.shape[0], 1))))
    v = np.sum(np.multiply(u.flatten(), np.sqrt(np.sum(lifted_P ** 2, axis=1))))
    Q = np.multiply(lifted_P, 1 / np.linalg.norm(lifted_P, ord=2, axis=1)[:, np.newaxis])
    s = np.multiply(u.flatten(), 1 / v * np.linalg.norm(lifted_P, ord=2, axis=1))

    last_entry_vec = np.zeros((1, lifted_P.shape[1]))
    last_entry_vec[0, -1] = x / v

    H = Q - last_entry_vec

    tau = v / int(np.sqrt(1 / eps))
    alpha = 2 * (1 + 2 * (1 + tau ** 2) / (1 - tau) ** 2)

    # beta = int(np.ceil(alpha / eps))
    h = np.empty((beta, H.shape[1]))
    c_i = copy.deepcopy(h)
    c_i[0, :] = np.random.choice(np.arange(P.shape[0]))
    origin = np.zeros((H.shape[1],))
    for i in range(beta - 1):
        h[i, :] = H[np.argmax(np.linalg.norm(H - c_i[i, :], ord=2, axis=1)), :]
        _, D, V = np.linalg.svd(np.vstack((np.zeros(h[i, :].shape), h[i, :] - c_i[i, :])))
        orth_line_segment = null_space(V[np.where(D > 1e-11)[0], :])
        project_origin = -np.dot(origin - c_i[i, :], orth_line_segment.dot(orth_line_segment.T))
        if checkIfPointOnSegment(project_origin, c_i[i, :], h[i, :], orth_line_segment):
            c_i[i + 1, :] = project_origin
        else:
            dist1, dist2 = np.linalg.norm(project_origin - c_i[i, :]), np.linalg.norm(project_origin - h[i, :])
            c_i[i + 1, :] = h[i, :] if dist2 < dist1 else c_i[i, :]

    _, w_prime = FWC.FrankWolfeCoreset(Q, s[:, np.newaxis], eps, beta).computeCoreset()

    w_double_prime = np.multiply(v * w_prime.flatten(), 1 / np.linalg.norm(lifted_P, ord=2, axis=1))
    w = w_double_prime / np.sum(w_double_prime)

    S = P[np.where(w > 0)[0], :]

    return S, w[np.where(w > 0)[0]], time.time() - ts, np.where(w > 0)[0]


if __name__ == '__main__':
    print('#############################   DATA LOADING    ##################################')
    df = load_data(file_path='data/all_data.csv')
    fig1 = compute_count(data=df)

    print('#############################   DATA PREPROCESSING    ##################################')
    df = clean_data(data=df)
    # show_bar_plot(data=df)

    print('#############################   DATA IMPUTATION    ##################################')
    df = DataFrameImputer().fit_transform(df)
    df = derive_features(data=df)
    df_fraud = df[df.isFraud == 'Y']
    # fig2a = get_box_plot(data=df, x_legend='Debtor Age', y_legend='Week of Month', title='')
    print(df.columns)
    df = df[['Debtor Age', 'Debtor Gender', 'Debtor Occupation', 'Debtor Register Device', 'Transaction Amount',
             'Transaction Device', 'Transaction Type', 'Week of Month', 'Transaction since Profile Modified',
             'Is Weekend', 'Count of added new Beneficiaries in the last 24 hours', 'isFraud']]
    df.to_csv('cleaned_data.csv', sep=',', index=False)
    df = convert_features(df=df)
    df.to_csv('processed_data.csv', sep=',')

    print('#############################   FEATURE SCALING    ##################################')
    # show_histogram_plot(data=df, first_feature='Debtor Age', second_feature='Week of Month',
    #                     title="Transaction occur Week of Month by User of Age",
    #                     x_label="Feature Values before Normalization",
    #                     y_label="Count"),
    # sel_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    # df = feature_scaling(data=df, features=sel_cols, method='Standard')
    # show_histogram_plot(data=df, first_feature='Debtor Age', second_feature='Week of Month',
    #                     title="Transaction occur Week of Month by User of Age",
    #                     x_label="Feature Values after Normalization",
    #                     y_label="Count"
    #                     )

    print('#############################   FEATURE ENCODING / CORRELATION    ##################################')
    # print(df.info())
    # df.to_csv('preprocessed_WOS.csv', index=False)
    df = feature_encoding(data=df, method="Label Encoding")
    # print(df.info())
    # df, fig_corr = remove_duplicate(data=df, threshold=0.9)

    print('###################################   GLM  ####################################')
    # model_GLM(data=df)

    print('################   OVERSAMPLING USING SMOTE   #########################')
    '''
    df_leg = df[df.isFraud == 0]
    P = df_leg.drop(['isFraud'], axis=1).to_numpy()
    w = np.ones((P.shape[0], 1)) / P.shape[0]
    P = P - np.mean(P, 0)
    P = P / np.sqrt(np.sum(np.multiply(w.flatten(), np.sum(P ** 2, axis=1))))
    coreset_size = 100
    S, u, t, w_sub = attainCoresetByDanV2(P, w, eps=0.1, beta=coreset_size)
    df_fraud = df[df.isFraud == 1]
    df_leg = df_leg.loc[df_leg.index.isin(w_sub)]
    df = pd.concat([df_fraud, df_leg], axis=1)
    '''
    X = df.drop(['isFraud'], axis=1)
    y = df[['isFraud']]

    X, y = oversampling(X=X, y=y, method="SMOTE", features_cat=[])
    # data = pd.concat([X, y], axis=1)
    # data.to_csv('preprocessed_WS.csv', index=False)
    # fig5 = compute_count(data=pd.concat([X, y], axis=1))

    print('###################################   FEATURE SELECTION  ####################################')
    # features, feature_importance = feature_selector(X=X, y=y.values.ravel(), method='Random Forest')
    # important_features_dict = dict(zip(features, feature_importance))
    # fig_fs = get_bar_chart(important_features_dict, x_legend='Features', y_legend='Importance Score',
    #                        title='Important Features', height=500, width=0.2)
    # print(features, feature_importance)
    # X = X[features[:10]]

    print('###################################   DATA SPLITTING  ####################################')
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=y)
    df_testing = pd.concat([X_testing, y_testing], axis=1)
    df_testing.to_csv('test_data.csv', index=False)
    y_testing = y_testing.values.ravel()

    print('###################################   MODEL TRAINING / TESTING  ####################################')
    # model_names = ["KNN", "Logistic Regression", "SVM", "Decision Tree", "Random Forest", "AdaBoost"]
    model_names = ["Decision Tree", "Random Forest"]
    for model in model_names:
        fig = run_model(X_train=X_training, y_train=y_training, X_test=X_testing, y_test=y_testing, model_name=model)

    print('###################################   MODEL PIPELINE  ####################################')
    # run_pipeline(X=X, y=y, model_idx=0, is_oversamlping=True)
    # run_pipeline(X=X, y=y, model_idx=1, is_oversamlping=True)
    # run_pipeline(X=X, y=y, model_idx=2, is_oversamlping=True)
    # run_pipeline(X=X, y=y, model_idx=3, is_oversamlping=True)
    # run_pipeline(X=X, y=y, model_idx=4, is_oversamlping=True)
    # run_pipeline(X=X, y=y, model_idx=5, is_oversamlping=True)

    '''
    print('#############################   GENERATING FIGURES    ##################################')
    figs = [fig1, fig2a, fig2b, fig_corr, fig5, fig_fs, fig_knn, fig_lr, fig_svm, fig_dc, fig_rf, fig_ab]
    with open('./results/result.html', 'w') as f:
        [f.writelines(io.to_html(fig, include_plotlyjs='cnd', full_html=True)) for fig in figs]
    '''
