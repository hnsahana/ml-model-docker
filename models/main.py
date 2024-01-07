# Data Manipulation libraries
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

df = pd.read_csv('house.csv')  # Load the dataset

x_train = df[['location', 'bedroom_count', 'locality', 'carpet_area']]
y_train = df[['price']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
X_train, X_test, Y_train, Y_test = train_test_split(x_train_scaled, test_size = 0.33, random_state = 5)

mlp = MLPRegressor(hidden_layer_sizes=(60), max_iter=1000)
mlp.fit(X_train, Y_train)
Y_predict = mlp.predict(X_test)

#Saving the machine learning model to a file
joblib.dump(mlp, "model/price_model.pkl")

