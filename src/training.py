import time
import numpy as np
import pandas as pd
from typing import List
from pandas import DataFrame
from sklearn.svm import LinearSVC
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool, metrics, cv
from visualizer import get_roc_curve, get_models_reports
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import EditedNearestNeighbours
from models import model_KNN, model_LogisticRegression, model_SVM, model_DecisionTree, model_RandomForest, \
    model_AdaBoost


def run_model(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: List, model_name: str) -> object:
    """
    This function will run model and generate reports
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param model_name: Name of supervised model
    :return:
    """
    print(model_name)
    y_pred = []
    if model_name == 'KNN':
        y_pred = model_KNN(X_train=X_train, y_train=y_train, X_test=X_test, neighbor=5)
    elif model_name == 'Logistic Regression':
        y_pred = model_LogisticRegression(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_name == 'SVM':
        y_pred = model_SVM(X_train=X_train, y_train=y_train, X_test=X_test)
    elif model_name == 'Decision Tree':
        y_pred = model_DecisionTree(X_train=X_train, y_train=y_train, X_test=X_test)
    elif model_name == 'Random Forest':
        y_pred = model_RandomForest(X_train=X_train, y_train=y_train, X_test=X_test)
    else:
        y_pred = model_AdaBoost(X_train=X_train, y_train=y_train, X_test=X_test)

    # res = pd.DataFrame(y_pred)
    # res.columns = [model_name+'_Predictions']
    # res.to_csv(f'{model_name}.csv')

    get_models_reports(y_gt=y_test, y_pred=y_pred, model_name=model_name)
    fig = get_roc_curve(y_gt=y_test, y_pred=y_pred, model_name=model_name)

    return fig


def run_pipeline(X: DataFrame, y: DataFrame, model_idx: int, is_oversamlping: bool) -> object:
    """
    This function will run model and generate reports
    :param X:
    :param y:
    :return:
    """
    model = KNeighborsClassifier()
    model_names = ["KNN", "Logistic Regression", "SVM", "Decision Tree", "Random Forest", "AdaBoost"]
    categorical_features_indices = np.where(X.dtypes != float)[0]
    print(model_names[model_idx])
    start_time = time.time()
    if model_idx == 0:
        model = KNeighborsClassifier(n_neighbors=5, p=1)
    elif model_idx == 1:
        model = LogisticRegression(solver='lbfgs', max_iter=3000)
    elif model_idx == 2:
        model = LinearSVC()
    elif model_idx == 3:
        model = DecisionTreeClassifier(class_weight=None, criterion="entropy", max_depth=3, max_features=None,
                                       max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_split=2,
                                       min_samples_leaf=5, random_state=100, splitter='best')
    elif model_idx == 4:
        model = RandomForestClassifier(max_depth=2, n_estimators=30, min_samples_split=3, max_leaf_nodes=5,
                                       random_state=22)
    elif model_idx == 5:
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1)

    if is_oversamlping:
        resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='all'))
        pipeline = Pipeline(steps=[('r', resample), ('m', model)])
    else:
        pipeline = Pipeline(steps=[('m', model)])

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # scoring = ['precision_macro', 'recall_macro']
    scoring = ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
               'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
               'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'r2', 'roc_auc']
    scores = cross_validate(pipeline, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)
    print("--- %s seconds ---" % (time.time() - start_time))
    print_scores(scores)


def print_scores(scores):
    """
    This function will print different scores e.g. Precision, Recall and F-Score
    """
    print('Mean Recall: %.4f' % np.mean(scores['test_recall']))
    print('Mean Precision: %.4f' % np.mean(scores['test_precision']))
    print('Mean F1: %.4f' % np.mean(scores['test_f1']))
    print('Mean Macro Recall: %.4f' % np.mean(scores['test_recall_macro']))
    print('Mean Macro Precision: %.4f' % np.mean(scores['test_precision_macro']))
    print('Mean Macro F1: %.4f' % np.mean(scores['test_f1_macro']))
    print('Mean Micro Recall: %.4f' % np.mean(scores['test_recall_micro']))
    print('Mean Micro Precision: %.4f' % np.mean(scores['test_precision_micro']))
    print('Mean Micro F1: %.4f' % np.mean(scores['test_f1_micro']))
    print('Mean Weighted Recall: %.4f' % np.mean(scores['test_recall_weighted']))
    print('Mean Weighted Precision: %.4f' % np.mean(scores['test_precision_weighted']))
    print('Mean Weighted F1: %.4f' % np.mean(scores['test_f1_weighted']))
    print('Mean accuracy: %.4f' % np.mean(scores['test_accuracy']))
    print('Mean r2: %.4f' % np.mean(scores['test_r2']))
    print('Mean roc_auc: %.4f' % np.mean(scores['test_roc_auc']))

