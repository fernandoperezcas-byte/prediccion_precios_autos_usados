from config import settings
from src import data_preprocessing as dp
from src import model_trainer as mt
from src import evaluator as ev
from sklearn.model_selection import train_test_split

def run_pipeline():
    print("⏳ Iniciando Pipeline Profesional...")

    # 1. Carga y Procesamiento
    df = dp.cargar_datos(settings.DATA_PATH)
    df_limpio = dp.transformar_datos(df, settings.CURRENT_YEAR)
    X, y = dp.separar_x_y(df_limpio, settings.TARGET_COLUMN)

    # 2. División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    # 3. Entrenamiento
    modelo = mt.entrenar(X_train, y_train)
    print("✅ Entrenamiento completado.")

    # 4. Evaluación
    ev.evaluar_modelo(modelo, X_test, y_test)

    # 5. Guardado
    mt.guardar_artefactos(
        modelo, 
        X.columns.tolist(), 
        settings.MODEL_DIR, 
        settings.MODEL_NAME, 
        settings.COLUMNS_NAME
    )
    print(f"💾 Modelos exportados a la carpeta {settings.MODEL_DIR}")

if __name__ == "__main__":
    run_pipeline()