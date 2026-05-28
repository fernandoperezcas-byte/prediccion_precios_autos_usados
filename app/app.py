import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Precios de Autos", page_icon="🚗", layout="centered")

st.title("🚗 Simulador de Precios de Autos Usados")
st.write("Introduce las características del vehículo para estimar su precio de venta de mercado.")

# 2. Cargar el modelo y las columnas esperadas
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('models/modelo_regresion_lineal.pkl')
    columnas = joblib.load('models/columnas_modelo.pkl')
    return modelo, columnas

try:
    modelo, columnas_entrenamiento = cargar_artefactos()
except Exception as e:
    st.error("Error al cargar el modelo. ¿Ejecutaste primero src/train.py?")
    st.stop()

# 3. Formulario de entrada de datos en la interfaz de usuario
st.header("📋 Características del Vehículo")

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Precio del auto nuevo (en miles)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
    kms_driven = st.number_input("Kilometraje acumulado (Kms)", min_value=0, max_value=500000, value=30000, step=1000)
    año_fabricacion = st.number_input("Año de fabricación", min_value=2000, max_value=2026, value=2018, step=1)
    antiguedad = 2026 - año_fabricacion

with col2:
    fuel_type = st.selectbox("Tipo de Combustible", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Tipo de Vendedor", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmisión", ["Manual", "Automatic"])

# 4. Procesar la entrada para que coincida exactamente con el One-Hot Encoding del entrenamiento
if st.button("🚀 Calcular Precio Estimado", use_container_width=True):
    # Crear un diccionario base con ceros para todas las columnas requeridas
    datos_usuario = {col: 0.0 for col in columnas_entrenamiento}
    
    # Asignar los valores numéricos directos
    datos_usuario['Present_Price'] = float(present_price)
    datos_usuario['Kms_Driven'] = float(kms_driven)
    datos_usuario['Años_Antiguedad'] = float(antiguedad)
    
    # Modificar las columnas del One-Hot Encoding según la selección del usuario
    # El modelo espera columnas tipo: 'Fuel_Type_Diesel', 'Seller_Type_Individual', etc.
    if fuel_type == "Diesel": datos_usuario['Fuel_Type_Diesel'] = 1.0
    if fuel_type == "Petrol": datos_usuario['Fuel_Type_Petrol'] = 1.0
    if seller_type == "Individual": datos_usuario['Seller_Type_Individual'] = 1.0
    if transmission == "Manual": datos_usuario['Transmission_Manual'] = 1.0
    
    # Convertir a DataFrame asegurando el orden exacto de las columnas
    df_pred = pd.DataFrame([datos_usuario])[columnas_entrenamiento]
    
    # Realizar la predicción
    prediccion = modelo.predict(df_pred)[0]
    
    # Mostrar el resultado (evitando valores negativos raros)
    precio_final = max(0.0, prediccion)
    st.success(f"### 💰 El precio estimado de venta es: **{precio_final:.2f}** unidades de precio")
