import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la página y bloqueo de traductores de navegador
st.set_page_config(page_title="Predicción de Precios de Autos", page_icon="🚗", layout="centered")
st.markdown('<html lang="es" class="notranslate" translate="no">', unsafe_allow_html=True)

st.title("🚗 Predicador de Precios de Autos Usados")
st.write("Introduce las características del vehículo para estimar su precio de venta de mercado.")

# 2. Cargar el modelo y las columnas con manejo explícito de rutas
@st.cache_resource
def cargar_artefactos():
    # Usar os.path asegura compatibilidad total entre Windows y Linux (Streamlit Cloud)
    ruta_modelo = os.path.join("models", "modelo_regresion_lineal.pkl")
    ruta_columnas = os.path.join("models", "columnas_modelo.pkl")
    
    modelo = joblib.load(ruta_modelo)
    columnas = joblib.load(ruta_columnas)
    return modelo, columnas

try:
    modelo, columnas_entrenamiento = cargar_artefactos()
except Exception as e:
    st.error(f"Error al cargar los artefactos del modelo: {e}")
    st.stop()

# 3. Interfaz del Formulario de Entrada
st.header("📋 Características del Vehículo")

with st.form(key='formulario_prediccion'):
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
    
    # Sintaxis limpia y correcta del botón
    submit_button = st.form_submit_button(label="🚀 Calcular Precio Estimado", use_container_width=True)

# 4. Procesamiento de la Predicción
if submit_button:
    # Generar diccionario base con floats
    datos_usuario = {col: 0.0 for col in columnas_entrenamiento}
    
    # Mapeo de datos numéricos directos (Uso seguro de strings con caracteres especiales)
    datos_usuario['Present_Price'] = float(present_price)
    datos_usuario['Kms_Driven'] = float(kms_driven)
    
    # Manejo dinámico de la clave por si existe variación de codificación de la 'ñ'
    if 'Años_Antiguedad' in datos_usuario:
        datos_usuario['Años_Antiguedad'] = float(antiguedad)
    elif 'Anos_Antiguedad' in datos_usuario:
        datos_usuario['Anos_Antiguedad'] = float(antiguedad)
        
    # Variables categóricas del One-Hot Encoding
    if fuel_type == "Diesel": datos_usuario['Fuel_Type_Diesel'] = 1.0
    if fuel_type == "Petrol": datos_usuario['Fuel_Type_Petrol'] = 1.0
    if seller_type == "Individual": datos_usuario['Seller_Type_Individual'] = 1.0
    if transmission == "Manual": datos_usuario['Transmission_Manual'] = 1.0
    
    # Conversión a DataFrame ordenado
    df_pred = pd.DataFrame([datos_usuario])[columnas_entrenamiento]
    
    # Predicción final escalar sin mutaciones de interfaz
    prediccion = modelo.predict(df_pred)
    precio_final = max(0.0, float(prediccion[0]))
    
    st.balloons()
    st.success(f"### 💰 El precio estimado de venta es: **{precio_final:.2f}** unidades de precio")


