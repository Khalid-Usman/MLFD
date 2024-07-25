import joblib
import pandas as pd
from visualizer import get_models_reports

if __name__ == '__main__':
    print('#############################   DATA LOADING    ##################################')
    df = pd.read_csv('demo.csv')
    X = df.drop(['isFraud'], axis=1)
    y_actual = df[['isFraud']].values.ravel()

    for column in X.columns:
        le = joblib.load(f'{column}.joblib')
        X[column] = le.transform(X[column])
    loaded_rf = joblib.load("rForest.joblib")
    y_pred = loaded_rf.predict(X).tolist()
    y_pred = ['Y' if x == 1 else 'N' for x in y_pred]
    df['Predicted'] = y_pred
    get_models_reports(y_gt=y_actual, y_pred=y_pred, model_name='Random Forest')
    df.to_csv('demo-results.csv')
    print('Done')
