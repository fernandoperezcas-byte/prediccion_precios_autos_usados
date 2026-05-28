# 🚗 Predicción de Precios de Autos Usados (End-to-End ML Project)

Este es un proyecto completo de Machine Learning que abarca desde la exploración de datos y el entrenamiento de un modelo de regresión hasta su despliegue final como una aplicación web interactiva en la nube.

🌍 **Link de la Aplicación en Vivo:** https://prediccionpreciosautosusados-ctlgvd5vvobzuypwurmjcv.streamlit.app/

---

## 📊 Descripción del Problema y Dataset
El objetivo del proyecto es predecir el precio de venta de un vehículo usado basándose en sus características físicas y de uso. 
El dataset utilizado es el **Vehicle Dataset (CarDekho)** de Kaggle.

### Características clave utilizadas:
*   **Present_Price:** Precio del automóvil cuando era nuevo (en miles).
*   **Kms_Driven:** Kilometraje acumulado.
*   **Años_Antiguedad:** Variable calculada (2026 - Año de fabricación).
*   **Fuel_Type, Seller_Type, Transmission:** Variables categóricas procesadas mediante One-Hot Encoding.

---

## 🛠️ Estructura del Proyecto
El proyecto sigue los estándares de la industria para código limpio y modular:

```text
├── data/               # Datos crudos (raw) y procesados
├── notebooks/          # Jupyter Notebooks de experimentación y EDA
├── src/                # Scripts modulares de Python (Pipeline de entrenamiento)
│   └── train.py        # Carga, procesa datos, entrena y exporta el modelo
├── models/             # Artefactos del modelo entrenado (.pkl)
└── app/                # Código de la interfaz web
    └── app.py          # Aplicación construida con Streamlit
```

---

## 🚀 Modelo y Rendimiento
Se implementó un modelo de **Regresión Lineal** como línea base, obteniendo un rendimiento sólido que demuestra la calidad de la ingeniería de variables realizada:

*   **R² Score (Precisión):** 0.8490 (Explica el ~85% de la variabilidad de los precios).
*   **MAE (Error Medio Absoluto):** 1.21 unidades de precio.

---

## ⚙️ Instalación y Uso Local

1. Clonar el repositorio:
   ```bash
   git clone https://github.com
   cd prediccion_precios_autos_usados
   ```

2. Instalar las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

3. Entrenar el modelo desde la raíz:
   ```bash
   python src/train.py
   ```

4. Lanzar la aplicación local de Streamlit:
   ```bash
   streamlit run app/app.py
   ```
