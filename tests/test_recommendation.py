import pandas as pd
import numpy as np
import modeloV01 as m


class DummyModel:
    def predict(self, X):
        # determinístico: valores descendentes (mayor = mejor)
        return np.linspace(100, 50, num=len(X))


def test_recomendar_recetas_y_plan_sin_entrenar(monkeypatch):
    # 1) recetas_df mínima ya "preprocesada" (con columnas que el filtro usa)
    recetas = pd.DataFrame([
        {"ID_Receta": 1, "Nombre_Receta": "R1", "Tipo_Comida": "Desayuno", "Categoria_Plato": "X",
         "Calorias": 300, "Proteinas": 10, "Grasas": 5, "Carbohidratos": 50, "Fibra": 1, "Azucares": 2, "Sodio": 100,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_BajaCarbo": 0,
         "Contiene_Lactosa": 0, "Compatible_SinGluten": 1, "Sin_Frutos_Secos": 1,
         "Bajo_En_Sodio": 1, "Alto_Proteico": 0, "Bajo_En_Grasa": 1, "Alto_En_Fibra": 0, "Apto_Diabetico": 1},

        {"ID_Receta": 2, "Nombre_Receta": "R2", "Tipo_Comida": "Almuerzo", "Categoria_Plato": "X",
         "Calorias": 600, "Proteinas": 25, "Grasas": 20, "Carbohidratos": 70, "Fibra": 5, "Azucares": 3, "Sodio": 120,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_BajaCarbo": 0,
         "Contiene_Lactosa": 0, "Compatible_SinGluten": 1, "Sin_Frutos_Secos": 1,
         "Bajo_En_Sodio": 1, "Alto_Proteico": 1, "Bajo_En_Grasa": 0, "Alto_En_Fibra": 1, "Apto_Diabetico": 1},

        {"ID_Receta": 3, "Nombre_Receta": "R3", "Tipo_Comida": "Cena", "Categoria_Plato": "X",
         "Calorias": 500, "Proteinas": 30, "Grasas": 15, "Carbohidratos": 40, "Fibra": 4, "Azucares": 2, "Sodio": 130,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_BajaCarbo": 0,
         "Contiene_Lactosa": 0, "Compatible_SinGluten": 1, "Sin_Frutos_Secos": 1,
         "Bajo_En_Sodio": 1, "Alto_Proteico": 1, "Bajo_En_Grasa": 1, "Alto_En_Fibra": 1, "Apto_Diabetico": 1},
    ])

    # 2) inyectar globals (evita entrenar / cargar artifacts)
    monkeypatch.setattr(m, "recetas_df", recetas)
    monkeypatch.setattr(m, "modelo", DummyModel())
    monkeypatch.setattr(m, "_MODELO_LISTO", True)

    # 3) columnas categóricas esperadas por construir_usuario_desde_parametros()
    monkeypatch.setattr(m, "user_cat_cols", ["Dieta_Equilibrada", "Obj_mantener"])

    # 4) features mínimas que existirán en el merge user_df + recetas_df
    monkeypatch.setattr(m, "feature_cols", ["Edad", "Peso", "Talla", "Comidas_Diarias", "Calorias", "Proteinas"])

    # --------- DOCUMENTACIÓN (impresiones para evidencia) ----------
    print("\n[UT-ML-12] recomendar_recetas_y_plan_sin_entrenar")
    print("  Objetivo: Validar el flujo de recomendación sin entrenar (mock de modelo y data mínima).")
    print("  Entradas (usuario): sexo=M, edad=25, peso=70, talla=175, actividad=moderado, comidas=3, dieta=equilibrada, objetivo=mantener")
    print("  Entradas (recetas): 3 recetas mock (Desayuno/Almuerzo/Cena) con macros y flags")
    print("  Esperado:")
    print("    - recomendaciones_top tiene 2 filas (top_n_recetas=2)")
    print("    - incluye columna score_modelo")
    print("    - plan llega hasta Dia=3 (dias=3)")

    recs_top, plan = m.recomendar_recetas_y_plan(
        sexo="M", edad=25, peso=70, talla=175,
        nivel_actividad="moderado",
        comidas_diarias=3,
        tipo_dieta="equilibrada",
        restricciones_lista=[],
        objetivo_nutricional="mantener",
        dias=3,
        top_n_recetas=2
    )

    # --------- OBTENIDO (para tu captura del documento) ----------
    print("  Obtenido:")
    print(f"    recomendaciones_top filas = {len(recs_top)}")
    print(f"    columnas recomendaciones_top = {list(recs_top.columns)}")
    print(f"    IDs recomendados = {list(recs_top['ID_Receta'])}")

    if "score_modelo" in recs_top.columns:
        top_scores = recs_top[["ID_Receta", "score_modelo"]].to_dict(orient="records")
        print(f"    score_modelo (top) = {top_scores}")

    print(f"    plan filas = {len(plan)}")
    print(f"    plan Dia min/max = {plan['Dia'].min()} / {plan['Dia'].max()}")
    print(f"    plan comidas únicas = {sorted(plan['Comida'].unique())}")

    # --------- ASSERTS (validación real) ----------
    assert len(recs_top) == 2
    assert "score_modelo" in recs_top.columns
    assert plan["Dia"].max() == 3
