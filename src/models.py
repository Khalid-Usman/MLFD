import joblib
from typing import List
from sklearn import tree
from pandas import DataFrame
import statsmodels.api as sm
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def model_KNN(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, neighbor: int) -> List:
    """
    This model will find the nearest neighbors and assign the majority class the unseen data
    :param X_train: The training data features
    :param y_train: The training target class
    :param X_test: The testing data features
    :param neighbor: The nearest neighbors to find the Euclidean Distance
    :return: The predicted results
    """
    knn = KNeighborsClassifier(n_neighbors=neighbor, p=1)
    knn.fit(X_train, y_train.values.ravel())
    print(knn)
    y_pred = knn.predict(X_test)
    return y_pred


def model_LogisticRegression(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: List) -> List:
    """
    This model will find the nearest neighbors and assign the majority class the unseen data
    :param X_train: The training data features
    :param y_train: The training target class
    :param X_test: The testing data features
    :param y_test: The ground truth target class
    :return: The predicted results
    """
    model = LogisticRegression(solver='lbfgs', max_iter=3000)
    model = model.fit(X_train, y_train.values.ravel())
    mse, bias, var = bias_variance_decomp(model, X_train.values, y_train.values.ravel(), X_test.values, y_test,
                                          loss='mse', num_rounds=200, random_seed=1)
    print('MSE: ', mse, ' Bias: ', bias, ' Variance: ', var)
    y_pred = model.predict(X_test)
    print('Intercept = ', model.intercept_)
    print('Coefficients = ', model.coef_)
    return y_pred


def model_SVM(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame) -> List:
    """
    This model will generate multiple decision tree and voting on the output of unseen data
    :param X_train: The training data features
    :param y_train: The training target class
    :param X_test: The testing data features
    :return: The predicted results
    """
    svm_clf = LinearSVC()
    svm_clf.fit(X_train, y_train.values.ravel())
    y_pred = svm_clf.predict(X_test)
    print(svm_clf)
    print('weights: ')
    print(svm_clf.coef_)
    print('Intercept: ')
    print(svm_clf.intercept_)
    return y_pred


def model_DecisionTree(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame) -> List:
    """
    This model will generate multiple decision tree and voting on the output of unseen data
    :param X_train: The training data features
    :param y_train: The training target class
    :param X_test: The testing data features
    :return: The predicted results
    """
    features = list(X_train.columns)
    target = ['No', 'Yes']
    dtree = DecisionTreeClassifier(max_depth=10, random_state=1234)
    model = dtree.fit(X_train, y_train.values.ravel())
    joblib.dump(model, "dTree.joblib")
    text_representation = tree.export_text(dtree)
    print(text_representation)
    y_pred = model.predict(X_test)
    fig = plt.figure(figsize=(150, 100))
    _ = tree.plot_tree(dtree,
                       impurity=False,
                       feature_names=features,
                       class_names=target,
                       filled=True)
    fig.savefig("decision_tree.png")
    return y_pred


def model_RandomForest(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame) -> List:
    """
    This model will generate multiple decision tree and voting on the output of unseen data
    :param X_train: The training data features
    :param y_train: The training target class
    :param X_test: The testing data features
    :return: The predicted results
    """
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, verbose=1)
    rf_clf.fit(X_train, y_train.values.ravel())
    joblib.dump(rf_clf, "rForest.joblib")
    print(rf_clf)
    y_pred = rf_clf.predict(X_test)
    return y_pred


def model_AdaBoost(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame) -> List:
    """
    This model will find the XGBoosting bagging
    :param X_train: The training data features
    :param y_train: The training target class
    :param X_test: The testing data features
    :return: The predicted results
    """
    XGBoost_CLF = AdaBoostClassifier(n_estimators=50, learning_rate=1)

    XGBoost_CLF.fit(X_train, y_train.values.ravel())
    print(XGBoost_CLF)
    y_pred = XGBoost_CLF.predict(X_test)
    return y_pred


def model_GLM(data: DataFrame):
    """
    Generalized Linear Model is one of many models to form the linear relationship between the dependent variable
    and its predictors.
    """
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('t/W', 't_W')
    cols = data.columns
    formula = cols[-1] + " ~ " + ' + '.join(cols[:-1])
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())

    # Fit the model
    result = model.fit()
    # Display and interpret results
    print(result.summary())
    # Estimated default probabilities
    predictions = result.predict()
    print(predictions)
    # Calculate the Akaike criterion
    print(result.aic)
    # Calculate the Bayesian information criterion
    print(result.bic)