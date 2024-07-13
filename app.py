from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Path to the trained model
MODEL_PATH = "rForest.joblib"

# Load the trained model
loaded_rf = joblib.load(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json

        # Convert the data into a DataFrame
        X = pd.DataFrame([data])

        # Make predictions
        y_pred = loaded_rf.predict(X).tolist()

        # Return the result as JSON
        return jsonify({"predictions": y_pred})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')