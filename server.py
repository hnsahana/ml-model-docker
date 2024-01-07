#importing necessary libraries
from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    #loading saved model here in this python file
    model = joblib.load('model/rf_model.pkl')
    #creating data frame of JSON data
    df = pd.DataFrame(json, index=[0])

    from sklearn.preprocessing import StandardScaler
    #performing preprocessing steps
    scaler = StandardScaler()
    scaler.fit(df)
    
    x_scaled = scaler.transform(df)

    x_scaled = pd.DataFrame(x_scaled, columns=df.columns)
    y_predict = model.predict(x_scaled)

    res= {"Predicted Price of House" : y_predict[0]}
    return jsonify(res)

if __name__ == "__main__":
    app.run(host='0.0.0.0')