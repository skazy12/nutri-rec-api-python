import pandas as pd
import modeloV01 as m

# -----------------------------
# UT-ML-08: filtrar_recetas_por_restricciones_y_exclusiones()
# -----------------------------

def test_filtrar_recetas_dieta_y_restricciones_y_exclusiones():
    df = pd.DataFrame([
        {"ID_Receta": 1, "Nombre_Receta": "A", "Compatible_Vegana": 1, "Compatible_SinGluten": 1, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Azucares": 2, "Carbohidratos": 20, "Sodio": 100},
        {"ID_Receta": 2, "Nombre_Receta": "B", "Compatible_Vegana": 0, "Compatible_SinGluten": 1, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Azucares": 2, "Carbohidratos": 20, "Sodio": 100},
        {"ID_Receta": 3, "Nombre_Receta": "C", "Compatible_Vegana": 1, "Compatible_SinGluten": 0, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Azucares": 2, "Carbohidratos": 20, "Sodio": 100},
    ])

    tipo_dieta = "vegana"
    restricciones = ["sin gluten"]
    excluir_ids = [1]

    print("\n[UT-ML-07] filtrar_recetas_dieta_y_restricciones_y_exclusiones")
    print("  Objetivo: Validar filtro por dieta + restricciones + exclusiones.")
    print(f"  Entradas: tipo_dieta={tipo_dieta}, restricciones={restricciones}, excluir_ids={excluir_ids}")
    print("  Dataset entrada (IDs):", list(df["ID_Receta"]))

    out = m.filtrar_recetas_por_restricciones_y_exclusiones(
        recetas_df=df,
        tipo_dieta=tipo_dieta,
        restricciones_lista=restricciones,
        excluir_ids=excluir_ids,
        excluir_nombres=None
    )

    print("  Esperado:")
    print("    - ID 1 cumple dieta+restricción pero está excluido")
    print("    - ID 2 no es vegana")
    print("    - ID 3 no es sin gluten")
    print("    => resultado vacío (0 filas)")

    print("  Obtenido:")
    print("    filas =", len(out))
    if len(out) > 0:
        print("    IDs =", list(out["ID_Receta"]))
    else:
        print("    IDs = []")

    assert len(out) == 0


def test_filtrar_diabetico_umbral():
    df = pd.DataFrame([
        {"ID_Receta": 1, "Azucares": 10, "Carbohidratos": 30, "Sodio": 100,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_SinGluten": 1, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Nombre_Receta": "OK"},
        {"ID_Receta": 2, "Azucares": 11, "Carbohidratos": 30, "Sodio": 100,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_SinGluten": 1, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Nombre_Receta": "BAD"},
    ])

    print("\n[UT-ML-08] filtrar_diabetico_umbral")
    print("  Objetivo: Validar umbral de diabético según tu código (Azucares<=10 y Carbohidratos<=30).")
    print("  Entradas: restricciones=['diabetico']")
    print("  Dataset entrada (ID -> Azucares, Carbohidratos):")
    for _, r in df.iterrows():
        print(f"    {r['ID_Receta']} -> Azucares={r['Azucares']}, Carbohidratos={r['Carbohidratos']}")

    out = m.filtrar_recetas_por_restricciones_y_exclusiones(df, "equilibrada", ["diabetico"])

    print("  Esperado: solo ID 1 (Azucares=10, Carbohidratos=30) pasa el filtro; ID 2 se excluye (Azucares=11).")
    print("  Obtenido IDs:", list(out["ID_Receta"]) if len(out) > 0 else [])

    assert list(out["ID_Receta"]) == [1]


def test_filtrar_hipertenso_umbral_sodio():
    df = pd.DataFrame([
        {"ID_Receta": 1, "Sodio": 150, "Azucares": 0, "Carbohidratos": 0,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_SinGluten": 1, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Nombre_Receta": "OK"},
        {"ID_Receta": 2, "Sodio": 151, "Azucares": 0, "Carbohidratos": 0,
         "Compatible_Vegana": 1, "Compatible_Vegetariana": 1, "Compatible_SinGluten": 1, "Contiene_Lactosa": 0,
         "Sin_Frutos_Secos": 1, "Nombre_Receta": "BAD"},
    ])

    print("\n[UT-ML-9] filtrar_hipertenso_umbral_sodio")
    print("  Objetivo: Validar umbral de hipertenso según tu código (Sodio <= 150).")
    print("  Entradas: restricciones=['hipertenso']")
    print("  Dataset entrada (ID -> Sodio):")
    for _, r in df.iterrows():
        print(f"    {r['ID_Receta']} -> Sodio={r['Sodio']}")

    out = m.filtrar_recetas_por_restricciones_y_exclusiones(df, "equilibrada", ["hipertenso"])

    print("  Esperado: solo ID 1 (Sodio=150) pasa el filtro; ID 2 se excluye (Sodio=151).")
    print("  Obtenido IDs:", list(out["ID_Receta"]) if len(out) > 0 else [])

    assert list(out["ID_Receta"]) == [1]
