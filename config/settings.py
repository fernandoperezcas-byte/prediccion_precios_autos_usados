import os

# Rutas
DATA_PATH = os.path.join('data', 'raw', 'cardata.csv')
MODEL_DIR = 'models'
MODEL_NAME = 'modelo_regresion_lineal.pkl'
COLUMNS_NAME = 'columnas_modelo.pkl'

# Parámetros de ingeniería
CURRENT_YEAR = 2026
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'Selling_Price'