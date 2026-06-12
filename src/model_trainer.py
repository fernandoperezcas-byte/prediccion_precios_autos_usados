from sklearn.linear_model import LinearRegression
import joblib
import os

def entrenar(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

def guardar_artefactos(modelo, columnas, folder, model_name, col_name):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(modelo, os.path.join(folder, model_name))
    joblib.dump(columnas, os.path.join(folder, col_name))