import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def preparar_datos(ruta_csv):
    """Carga, limpia y transforma las variables del dataset."""
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv}")
        
    df = pd.read_csv(ruta_csv)
    
    # Ingeniería de variables (Año actual 2026)
    df['Años_Antiguedad'] = 2026 - df['Year']
    df_limpio = df.drop(['Car_Name', 'Year'], axis=1)
    
    # Transformación categórica (One-Hot Encoding)
    df_final = pd.get_dummies(df_limpio, drop_first=True)
    df_final = df_final.astype(float)
    
    return df_final

def entrenar_modelo():
    print("⏳ Iniciando el pipeline de entrenamiento...")
    
    # 1. Procesar datos
    ruta_datos = os.path.join('data', 'raw', 'cardata.csv')
    df_final = preparar_datos(ruta_datos)
    
    # 2. Dividir características y target
    X = df_final.drop('Selling_Price', axis=1)
    y = df_final['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Entrenar Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("✅ ¡Modelo entrenado con éxito!")
    
    # 4. Evaluar el modelo
    predicciones = modelo.predict(X_test)
    r2 = r2_score(y_test, predicciones)
    mae = mean_absolute_error(y_test, predicciones)
    
    print(f"\n📊 Métricas de Evaluación (Test Set):")
    print(f"   - R² Score (Precisión): {r2:.4f}")
    print(f"   - MAE (Error Medio Absoluto): {mae:.4f} (unidades de precio)")
    
    # 5. Guardar las columnas necesarias para el despliegue
    # Esto evitará errores de dimensiones cuando el usuario ingrese datos en la web
    columnas_X = X.columns.tolist()
    
    # 6. Guardar el modelo y los metadatos
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo_regresion_lineal.pkl')
    joblib.dump(columnas_X, 'models/columnas_modelo.pkl')
    print("\n💾 Archivos guardados en la carpeta 'models/'")

if __name__ == "__main__":
    entrenar_modelo()
