import numpy as np
import modeloV01 as m

# -----------------------------
# UT-ML-01: si_no_a_bin()
# -----------------------------

def test_si_no_a_bin_basicos():
    casos = [
        ("Si", 1),
        ("sí", 1),
        ("YES", 1),
        ("true", 1),
        ("No", 0),
        ("false", 0),
        ("", 0),
    ]

    print("\n[UT-ML-01] si_no_a_bin_basicos (Entrada -> Esperado -> Obtenido)")
    for entrada, esperado in casos:
        obtenido = m.si_no_a_bin(entrada)
        print(f"  {repr(entrada):>10} -> {esperado} -> {obtenido}")
        assert obtenido == esperado


def test_si_no_a_bin_numericos_y_none():
    casos = [
        (1, 1),
        (0, 0),
        (2, 1),
        (0.0, 0),
        (0.2, 1),
        (None, 0),
        (np.nan, 0),
    ]

    print("\n[UT-ML-01] si_no_a_bin_numericos_y_none (Entrada -> Esperado -> Obtenido)")
    for entrada, esperado in casos:
        obtenido = m.si_no_a_bin(entrada)
        print(f"  {repr(entrada):>10} -> {esperado} -> {obtenido}")
        assert obtenido == esperado


def test_si_no_a_bin_string_raro_parseable():
    casos = [
        ("2", 1),
        ("0", 0),
    ]

    print("\n[UT-ML-01] si_no_a_bin_string_raro_parseable (Entrada -> Esperado -> Obtenido)")
    for entrada, esperado in casos:
        obtenido = m.si_no_a_bin(entrada)
        print(f"  {repr(entrada):>10} -> {esperado} -> {obtenido}")
        assert obtenido == esperado


def test_si_no_a_bin_string_raro_no_parseable():
    entrada, esperado = "cualquier cosa", 0
    obtenido = m.si_no_a_bin(entrada)

    print("\n[UT-ML-01] si_no_a_bin_string_raro_no_parseable")
    print(f"  Entrada:  {repr(entrada)}")
    print(f"  Esperado: {esperado}")
    print(f"  Obtenido: {obtenido}")

    assert obtenido == esperado


# -----------------------------
# UT-ML-06: calcular_score_heuristico()
# -----------------------------

def test_score_heuristico_mayor_es_mejor():
    base = {
        "Comidas_Diarias": 3,
        "Calorias_Ajustadas": 1800,
        "Macro_Prot_g": 120,
        "Macro_Carb_g": 210,
        "Macro_Grasas_g": 60,
        "Restricciones": "Ninguna",
        "Objetivo_Nutricional": "mantener",
        "Contiene_Lactosa": 0,
        "Compatible_SinGluten": 1,
        "Apto_Diabetico": 1,
        "Bajo_En_Sodio": 1,
        "Bajo_En_Grasa": 1,
        "Alto_Proteico": 1,
        "Azucares": 3,
    }

    receta_buena = dict(base, Calorias=610, Proteinas=42, Carbohidratos=68, Grasas=21)
    receta_mala  = dict(base, Calorias=1200, Proteinas=10, Carbohidratos=10, Grasas=60)

    s_buena = m.calcular_score_heuristico(receta_buena)
    s_mala  = m.calcular_score_heuristico(receta_mala)

    print("\n[UT-ML-06] score_heuristico_mayor_es_mejor")
    print(f"  Score receta_buena: {s_buena}")
    print(f"  Score receta_mala : {s_mala}")
    print("  Esperado: score_buena > score_mala")

    assert s_buena > s_mala


def test_score_penaliza_sin_lactosa_si_contiene():
    row_ok = {
        "Comidas_Diarias": 3,
        "Calorias_Ajustadas": 1800,
        "Macro_Prot_g": 120,
        "Macro_Carb_g": 210,
        "Macro_Grasas_g": 60,
        "Restricciones": "sin lactosa",
        "Objetivo_Nutricional": "mantener",
        "Calorias": 600, "Proteinas": 40, "Carbohidratos": 70, "Grasas": 20,
        "Contiene_Lactosa": 0, "Compatible_SinGluten": 1, "Apto_Diabetico": 1, "Bajo_En_Sodio": 1,
        "Bajo_En_Grasa": 1, "Alto_Proteico": 1, "Azucares": 3,
    }
    row_bad = dict(row_ok, Contiene_Lactosa=1)

    s_ok  = m.calcular_score_heuristico(row_ok)
    s_bad = m.calcular_score_heuristico(row_bad)

    print("\n[UT-ML-06] score_penaliza_sin_lactosa_si_contiene")
    print(f"  Score sin lactosa (OK) : {s_ok}")
    print(f"  Score con lactosa (BAD): {s_bad}")
    print("  Esperado: score_ok > score_bad")

    assert s_ok > s_bad
