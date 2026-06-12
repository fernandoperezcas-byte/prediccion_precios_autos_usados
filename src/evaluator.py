from sklearn.metrics import mean_absolute_error, r2_score

def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    r2 = r2_score(y_test, predicciones)
    mae = mean_absolute_error(y_test, predicciones)
    
    print(f"\n📊 Métricas de Evaluación:")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - MAE: {mae:.4f}")
    return {"r2": r2, "mae": mae}