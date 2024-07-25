import os
import gzip
import pandas as pd
import joblib, pickle
from pandas import DataFrame
from feature_selection import feature_selector
from sklearn.preprocessing import LabelEncoder
from util.data_imputation import DataFrameImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessor import derive_features, oversampling

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


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
                       'TRANSPORTERS': 'SELF EMPLOYED', 'AGRICULTURE': 'SELF EMPLOYED', 'SALARIED': 'SALARIED PERSON',
                       'LAWYER': 'SELF EMPLOYED'}
    occupation_list = occupation_dict.keys()
    df['Debtor Occupation'] = df['Debtor Occupation'].apply(lambda x: occupation_dict[x] if x in occupation_list else x)


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
    return df


def clean_data(data: DataFrame) -> DataFrame:
    """
    This function will clean the dataframe
    :param data:
    :return:
    """
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


if __name__ == '__main__':
    print('#############################   DATA LOADING    ##################################')
    df = load_data(file_path='data/all_data.csv')
    df = clean_data(data=df)
    df = DataFrameImputer().fit_transform(df)
    df = derive_features(data=df)
    df_fraud = df[df.isFraud == 'Y']
    df = df[['Debtor Age', 'Debtor Gender', 'Debtor Occupation', 'Debtor Register Device', 'Transaction Amount',
             'Transaction Device', 'Transaction Type', 'Week of Month', 'Transaction since Profile Modified',
             'Is Weekend', 'Count of added new Beneficiaries in the last 24 hours', 'isFraud']]
    df.to_csv('cleaned_data.csv', sep=',', index=False)

    df = convert_features(df=df)
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv('processed_data.csv', sep=',')

    X = df.drop(['isFraud'], axis=1)
    y = df[['isFraud']]
    # X, y = oversampling(X=X, y=y, method="SMOTE", features_cat=[])

    print('###################################   DATA SPLITTING  ####################################')
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=y)

    df_training = pd.concat([X_training, y_training], axis=1)

    # df_training = feature_encoding(data=df_training, method="Label Encoding")
    label_encoders = {}

    # Fit and transform each column using LabelEncoder
    for column in df_training.columns:
        le = LabelEncoder()
        df_training[column] = le.fit_transform(df_training[column])
        label_encoders[column] = le

    # Save each LabelEncoder to a .joblib file
    for column, le in label_encoders.items():
        joblib.dump(le, f'{column}.joblib', compress=1)

    # Print the encoding values for each category in all columns
    for column, le in label_encoders.items():
        category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"\n{column} encoding values:")
        for original, encoded in category_mapping.items():
            print(f"{original}: {encoded}")

    df_testing = pd.concat([X_testing, y_testing], axis=1)
    df_testing.to_csv('test_data.csv', index=False)

    X = df_training.drop(['isFraud'], axis=1)
    y = df_training[['isFraud']]
    # X, y = oversampling(X=X, y=y, method="SMOTE", features_cat=[])

    print('###################################   FEATURE SELECTION  ####################################')
    features, feature_importance = feature_selector(X=X, y=y.values.ravel(), method='Random Forest')
    important_features_dict = dict(zip(features, feature_importance))
    print(features, feature_importance)

    '''
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42)
    xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
    xgb.plot_importance(xgb_model)
    xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
    joblib.dump(xgb_model, "xgBoost.joblib")
    '''

    # Best parameters found:  {'rf__max_depth': 20, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 1,
    # 'rf__min_samples_split': 5, 'rf__n_estimators': 300}

    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, verbose=1)
    rf_clf.fit(X, y.values.ravel())
    joblib.dump(rf_clf, "rForest.joblib", compress=1)

    model_filename = 'compressed_rf_model.pkl.gz'
    with gzip.open(model_filename, 'wb') as f:
        pickle.dump(rf_clf, f)

    pickle_out = open("classifier.pkl", "wb")
    pickle.dump(rf_clf, pickle_out)
    pickle_out.close()