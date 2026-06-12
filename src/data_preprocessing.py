import pandas as pd
import os

def cargar_datos(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Error: No existe el archivo en {ruta}")
    return pd.read_csv(ruta)

def transformar_datos(df, current_year):
    df = df.copy()
    # Ingeniería de variables
    df['Años_Antiguedad'] = current_year - df['Year']
    df = df.drop(['Car_Name', 'Year'], axis=1)
    
    # One-Hot Encoding
    df_final = pd.get_dummies(df, drop_first=True)
    return df_final.astype(float)

def separar_x_y(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y