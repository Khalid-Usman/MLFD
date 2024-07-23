import gzip
import pandas as pd
import pickle, joblib
from fastapi import FastAPI
from pydantic import BaseModel


class MLFD(BaseModel):
    Debtor_Age: str
    Debtor_Gender: str
    Debtor_Occupation: str
    Debtor_Register_Device: str
    Transaction_Amount: str
    Transaction_Device: str
    Transaction_Type: str
    Week_of_Month: str
    Transaction_since_Profile_Modified: str
    Is_Weekend: str
    Count_of_added_new_Beneficiaries_in_the_last_24_hours: str


def load_compressed_model(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


app = FastAPI()
# pickle_in = open("classifier.pkl", "rb")
# classifier = pickle.load(pickle_in)
classifier = load_compressed_model("compressed_rf_model.pkl.gz")

@app.get("/")
async def root():
    return {"message": "Welcome to Fraud Detection System!"}


@app.post('/predict')
async def predict_fraud(data: MLFD):
    data = data.dict()
    X = pd.DataFrame([data])
    for column in X.columns:
        le = joblib.load(f'{column}.joblib')
        X[column] = le.transform(X[column])

    expected_columns = [
        "Debtor_Age", "Debtor_Gender", "Debtor_Occupation",
        "Debtor_Register_Device", "Transaction_Amount",
        "Transaction_Device", "Transaction_Type", "Week_of_Month",
        "Transaction_since_Profile_Modified", "Is_Weekend",
        "Count_of_added_new_Beneficiaries_in_the_last_24_hours"
    ]
    X = X[expected_columns]
    var = X.iloc[0, :].tolist()

    prediction = classifier.predict([var])
    prediction = "Y" if prediction[0] > 0.5 else "N"
    return {
        'isFraud': prediction
    }