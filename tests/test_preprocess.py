import pandas as pd
import modeloV01 as m

# -----------------------------
# UT-ML-02: preprocesar_recetas()
# -----------------------------

def test_preprocesar_recetas_crea_columnas_y_convierte():
    df = pd.DataFrame([{
        "ID_Receta": "1",
        "Calorias": "320",
        "Proteinas": "9",
        "Grasas": "10",
        "Carbohidratos": "45",
        "Fibra": "5",
        "Azucares": "2",
        "Sodio": "120",
        "Compatible_Vegana": "Si",
        "Contiene_Lactosa": "No",
        "Compatible_SinGluten": "Si",
        "Tipo_Comida": "Almuerzo",
        "Categoria_Plato": "Ensalada",
    }])

    print("\n[UT-ML-02] preprocesar_recetas_crea_columnas_y_convierte")
    print("  Entradas (receta original):")
    print(df.to_dict(orient='records')[0])

    out, cols_num, cols_flags, cat_cols = m.preprocesar_recetas(df)

    print("  Obtenido (receta preprocesada):")
    print(out.to_dict(orient='records')[0])

    print("  Columnas numéricas detectadas:", cols_num)
    print("  Columnas flags detectadas:", cols_flags)
    print("  Columnas categóricas (dummies):", cat_cols)

    print("  Esperado:")
    print("    - Calorias y Sodio convertidos a numérico")
    print("    - Flags convertidos a binario (Si->1, No->0)")
    print("    - Dummies creadas para Tipo_Comida y Categoria_Plato")

    # numéricas como número
    assert out.loc[0, "Calorias"] == 320
    assert out.loc[0, "Sodio"] == 120

    # flags a binario
    assert out.loc[0, "Compatible_Vegana"] == 1
    assert out.loc[0, "Contiene_Lactosa"] == 0

    # dummies creadas
    assert any(c.startswith("Tipo_Comida_") for c in cat_cols)
    assert any(c.startswith("Categoria_") for c in cat_cols)


# -----------------------------
# UT-ML-03: preprocesar_usuarios()
# -----------------------------

def test_preprocesar_usuarios_sexo_actividad_y_dummies():
    df = pd.DataFrame([{
        "ID_Usuario": 1,
        "Edad": 25,
        "Peso": 70,
        "Talla": 175,
        "Comidas_Diarias": 3,
        "Calorias_Ajustadas": 2000,
        "Macro_Prot_g": 120,
        "Macro_Carb_g": 200,
        "Macro_Grasas_g": 60,
        "Sexo": "M",
        "Nivel_Actividad": "Moderado",
        "Tipo_Dieta": "Equilibrada",
        "Objetivo_Nutricional": "Mantener",
    }])

    print("\n[UT-ML-03] preprocesar_usuarios_sexo_actividad_y_dummies")
    print("  Entradas (usuario original):")
    print(df.to_dict(orient='records')[0])

    out, user_cat_cols = m.preprocesar_usuarios(df)

    print("  Obtenido (usuario preprocesado):")
    print(out.to_dict(orient='records')[0])

    print("  Columnas categóricas generadas:", user_cat_cols)

    print("  Esperado:")
    print("    - Sexo_M = 1 para sexo masculino")
    print("    - Nivel_Actividad_val = 2 para 'Moderado'")
    print("    - Dummies creadas para Tipo_Dieta y Objetivo_Nutricional")

    assert out.loc[0, "Sexo_M"] == 1
    assert out.loc[0, "Nivel_Actividad_val"] == 2
    assert any(c.startswith("Dieta_") for c in user_cat_cols)
    assert any(c.startswith("Obj_") for c in user_cat_cols)
