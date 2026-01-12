import modeloV01 as m

# -----------------------------
# UT-ML-04: calcular_requerimientos_nutricionales()
# -----------------------------

def test_calcular_requerimientos_nutricionales_retorna_4_valores():
    print("\n[UT-ML-04] calcular_requerimientos_nutricionales_retorna_4_valores")
    print("  Entradas: sexo=M, edad=25, peso=70, talla=175, actividad=moderado, objetivo=mantener")

    calorias, prot, carb, grasas = m.calcular_requerimientos_nutricionales(
        sexo="M",
        edad=25,
        peso=70,
        talla=175,
        nivel_actividad="moderado",
        objetivo_nutricional="mantener"
    )

    print("  Obtenido:")
    print(f"    calorias={calorias}, prot_g={prot}, carb_g={carb}, grasas_g={grasas}")
    print("  Esperado: calorias>0, prot>0, carb>=0, grasas>0")

    assert calorias > 0
    assert prot > 0
    assert carb >= 0
    assert grasas > 0


def test_calcular_requerimientos_disminuir_peso_reduce_calorias():
    print("\n[UT-ML-04] calcular_requerimientos_disminuir_peso_reduce_calorias")
    print("  Entradas: (M,25,70,175,moderado) comparar objetivo manteniendo vs disminuir_peso")
    print("  Esperado: calorias_disminuir_peso < calorias_mantener (tdee*0.8)")

    cal_mant, prot_m, carb_m, grasas_m = m.calcular_requerimientos_nutricionales(
        "M", 25, 70, 175, "moderado", "mantener"
    )
    cal_cut, prot_c, carb_c, grasas_c = m.calcular_requerimientos_nutricionales(
        "M", 25, 70, 175, "moderado", "disminuir_peso"
    )

    print("  Obtenido:")
    print(f"    Mantener:       calorias={cal_mant}, prot={prot_m}, carb={carb_m}, grasas={grasas_m}")
    print(f"    Disminuir_peso: calorias={cal_cut}, prot={prot_c}, carb={carb_c}, grasas={grasas_c}")

    assert cal_cut < cal_mant  # tdee*0.8


def test_calcular_requerimientos_ganar_musculo_increase_calorias():
    print("\n[UT-ML-04] calcular_requerimientos_ganar_musculo_increase_calorias")
    print("  Entradas: (M,25,70,175,moderado) comparar objetivo mantener vs ganar_musculo")
    print("  Esperado: calorias_ganar_musculo > calorias_mantener (tdee*1.1)")

    cal_mant, prot_m, carb_m, grasas_m = m.calcular_requerimientos_nutricionales(
        "M", 25, 70, 175, "moderado", "mantener"
    )
    cal_bulk, prot_b, carb_b, grasas_b = m.calcular_requerimientos_nutricionales(
        "M", 25, 70, 175, "moderado", "ganar_musculo"
    )

    print("  Obtenido:")
    print(f"    Mantener:     calorias={cal_mant}, prot={prot_m}, carb={carb_m}, grasas={grasas_m}")
    print(f"    Ganar_musculo:calorias={cal_bulk}, prot={prot_b}, carb={carb_b}, grasas={grasas_b}")

    assert cal_bulk > cal_mant  # tdee*1.1


# -----------------------------
# UT-ML-05: calcular_requerimientos_api()
# -----------------------------

def test_calcular_requerimientos_api_division_y_minimo():
    payload = {
        "sexo": "M",
        "edad": 25,
        "peso": 70,
        "talla": 175,
        "nivel_actividad": "moderado",
        "objetivo_nutricional": "mantener",
        "comidas_diarias": 4
    }

    print("\n[UT-ML-05] calcular_requerimientos_api_division_y_minimo")
    print(f"  Entradas payload: {payload}")
    print("  Esperado: comidas_diarias=4 y calorias_por_comida*4 ≈ calorias_diarias (tolerancia 0.1)")

    res = m.calcular_requerimientos_api(payload)

    print("  Obtenido:")
    print(f"    comidas_diarias={res['comidas_diarias']}")
    print(f"    calorias_diarias={res['calorias_diarias']}")
    print(f"    calorias_por_comida={res['calorias_por_comida']}")
    print(f"    prot_diarias={res['proteinas_g']} | prot_por_comida={res['proteinas_por_comida_g']}")
    print(f"    carb_diarias={res['carbohidratos_g']} | carb_por_comida={res['carbohidratos_por_comida_g']}")
    print(f"    grasas_diarias={res['grasas_g']} | grasas_por_comida={res['grasas_por_comida_g']}")

    assert res["comidas_diarias"] == 4
    assert abs(res["calorias_por_comida"] * 4 - res["calorias_diarias"]) <= 0.1


def test_calcular_requerimientos_api_comidas_cero_forza_1():
    payload = {
        "sexo": "M",
        "edad": 25,
        "peso": 70,
        "talla": 175,
        "nivel_actividad": "moderado",
        "objetivo_nutricional": "mantener",
        "comidas_diarias": 0
    }

    print("\n[UT-ML-05] calcular_requerimientos_api_comidas_cero_forza_1")
    print(f"  Entradas payload: {payload}")
    print("  Esperado: comidas_diarias=1 y calorias_por_comida ≈ calorias_diarias (tolerancia 0.1)")

    res = m.calcular_requerimientos_api(payload)

    print("  Obtenido:")
    print(f"    comidas_diarias={res['comidas_diarias']}")
    print(f"    calorias_diarias={res['calorias_diarias']}")
    print(f"    calorias_por_comida={res['calorias_por_comida']}")

    assert res["comidas_diarias"] == 1
    assert abs(res["calorias_por_comida"] - res["calorias_diarias"]) <= 0.1
