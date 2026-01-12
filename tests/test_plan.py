import pandas as pd
import modeloV01 as m

# -----------------------------
# UT-ML-10: armar_plan_por_dias()
# -----------------------------

def test_armar_plan_por_dias_estructura_y_cantidad():
    recs = pd.DataFrame([
        {"ID_Receta": 1, "Nombre_Receta": "D1", "Tipo_Comida": "Desayuno", "Categoria_Plato": "X",
         "Calorias": 300, "Proteinas": 10, "Grasas": 5, "Carbohidratos": 50, "score_modelo": 10},

        {"ID_Receta": 2, "Nombre_Receta": "A1", "Tipo_Comida": "Almuerzo", "Categoria_Plato": "X",
         "Calorias": 600, "Proteinas": 25, "Grasas": 20, "Carbohidratos": 70, "score_modelo": 9},

        {"ID_Receta": 3, "Nombre_Receta": "C1", "Tipo_Comida": "Cena", "Categoria_Plato": "X",
         "Calorias": 500, "Proteinas": 30, "Grasas": 15, "Carbohidratos": 40, "score_modelo": 8},

        {"ID_Receta": 4, "Nombre_Receta": "S1", "Tipo_Comida": "Merienda", "Categoria_Plato": "Snack",
         "Calorias": 200, "Proteinas": 5, "Grasas": 5, "Carbohidratos": 30, "score_modelo": 7},
    ])

    dias = 7
    comidas = 3

    print("\n[UT-ML-10] armar_plan_por_dias_estructura_y_cantidad")
    print("  Objetivo: Validar estructura, rango de días y cantidad total de registros del plan.")
    print(f"  Entradas: dias={dias}, comidas_diarias={comidas}, recetas_mock={len(recs)}")
    print("  Esperado:")
    print("    - plan es DataFrame")
    print("    - contiene columnas mínimas (Dia, Comida, ID_Receta, Nombre_Receta, Tipo_Comida, macros, score_modelo)")
    print("    - Dia min = 1 y Dia max = dias")
    print(f"    - filas = dias * comidas = {dias * comidas}")

    plan = m.armar_plan_por_dias(recs_ordenadas=recs, dias=dias, comidas_diarias=comidas)

    print("  Obtenido:")
    print(f"    tipo(plan)={type(plan)}")
    print(f"    filas={len(plan)}")
    print(f"    Dia min/max = {plan['Dia'].min()} / {plan['Dia'].max()}")
    print(f"    comidas únicas = {sorted(plan['Comida'].unique())}")

    # preview útil para evidencia
    print("    preview (primeras 5 filas):")
    cols_preview = ["Dia", "Comida", "ID_Receta", "Nombre_Receta", "Tipo_Comida", "score_modelo"]
    cols_preview = [c for c in cols_preview if c in plan.columns]
    print(plan[cols_preview].head(5).to_string(index=False))

    assert isinstance(plan, pd.DataFrame)

    required_cols = set([
        "Dia", "Comida", "ID_Receta", "Nombre_Receta", "Tipo_Comida",
        "Calorias", "Proteinas", "Grasas", "Carbohidratos", "score_modelo"
    ])
    assert required_cols.issubset(plan.columns)

    assert plan["Dia"].min() == 1
    assert plan["Dia"].max() == dias
    assert len(plan) == dias * comidas


# -----------------------------
# UT-ML-11: armar_plan_por_dias() con 4 comidas incluye Snack
# -----------------------------

def test_armar_plan_por_dias_4_comidas_incluye_snack():
    recs = pd.DataFrame([
        {"ID_Receta": 1, "Nombre_Receta": "D1", "Tipo_Comida": "Desayuno", "Categoria_Plato": "X", "score_modelo": 10},
        {"ID_Receta": 2, "Nombre_Receta": "A1", "Tipo_Comida": "Almuerzo", "Categoria_Plato": "X", "score_modelo": 9},
        {"ID_Receta": 3, "Nombre_Receta": "C1", "Tipo_Comida": "Cena", "Categoria_Plato": "X", "score_modelo": 8},
        {"ID_Receta": 4, "Nombre_Receta": "S1", "Tipo_Comida": "Merienda", "Categoria_Plato": "Snack", "score_modelo": 7},
    ])

    dias = 2
    comidas = 4

    print("\n[UT-ML-11] armar_plan_por_dias_4_comidas_incluye_snack")
    print("  Objetivo: Validar que con 4 comidas se incluya el slot Snack.")
    print(f"  Entradas: dias={dias}, comidas_diarias={comidas}, recetas_mock={len(recs)}")
    print("  Esperado:")
    print(f"    - filas = dias * comidas = {dias * comidas}")
    print("    - existe 'Snack' en la columna Comida")

    plan = m.armar_plan_por_dias(recs_ordenadas=recs, dias=dias, comidas_diarias=comidas)

    print("  Obtenido:")
    print(f"    filas={len(plan)}")
    print(f"    comidas únicas = {sorted(plan['Comida'].unique())}")
    print("    preview (todas las filas):")
    print(plan[["Dia", "Comida", "ID_Receta", "Nombre_Receta", "Tipo_Comida", "score_modelo"]].to_string(index=False))

    assert len(plan) == dias * comidas
    assert "Snack" in set(plan["Comida"])
