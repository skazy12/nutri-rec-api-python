"""
modeloV01.py

Modelo de recomendación nutricional basado en contenido.

✅ CORRECCIÓN PEDIDA (SIN tocar el entrenamiento):
- Si almuerzo_cena_misma=True:
  -> la Cena SIEMPRE es igual al Almuerzo del mismo día
  -> y se elige el Almuerzo con MEJOR score_modelo disponible para ese día (top)
- Si almuerzo_cena_misma=False:
  -> se intenta que Almuerzo y Cena sean DIFERENTES dentro del mismo día (si hay opciones)
  -> sin meter meriendas/postres en Cena por culpa del fallback
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# -------------------------------------------------------------------
# 1. RUTAS DE ARCHIVOS
# -------------------------------------------------------------------

BASE_DIR = "F:/ALL_STUDIES/SEMESTRE1_2025/TALLER_2/DATASET_DEF"

RUTA_RECETAS = os.path.join(BASE_DIR, "recetas_bolivianas_PRO_250.csv")
RUTA_USUARIOS = os.path.join(BASE_DIR, "usuarios_con_dietas_restricciones.csv")

# -------------------------------------------------------------------
# 1.1 ARTEFACTOS (para NO re-entrenar en cada arranque)
# -------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "rf_model.joblib"
META_PATH = ARTIFACTS_DIR / "meta.joblib"
RECETAS_PREP_PATH = ARTIFACTS_DIR / "recetas_preprocesadas.joblib"
DATA_FINGERPRINT_PATH = ARTIFACTS_DIR / "data_fingerprint.txt"

# ============================================================
# 2) GLOBALES (se cargan en memoria al arrancar API)
# ============================================================
recetas_df: Optional[pd.DataFrame] = None
feature_cols: Optional[List[str]] = None
user_cat_cols: Optional[List[str]] = None
modelo: Optional[RandomForestRegressor] = None
_MODELO_LISTO = False


# ============================================================
# 3) PREPROCESAMIENTO RECETAS, columnas numéricas y categóricas
# ============================================================
def si_no_a_bin(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return 0

    if isinstance(valor, str):
        v = valor.strip().lower()
        if v in ["si", "sí", "s", "yes", "y", "true", "t", "1"]:
            return 1
        if v in ["no", "n", "false", "f", "0", ""]:
            return 0
        try:
            return 1 if float(v) != 0 else 0
        except Exception:
            return 0

    if isinstance(valor, (int, np.integer)):
        return 1 if int(valor) != 0 else 0
    if isinstance(valor, (float, np.floating)):
        return 1 if float(valor) != 0 else 0

    return 0


def preprocesar_recetas(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    cols_numericas_receta = [
        "ID_Receta", "Calorias", "Proteinas", "Grasas", "Carbohidratos",
        "Fibra", "Azucares", "Sodio"
    ]
    for col in cols_numericas_receta:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0
    df[cols_numericas_receta] = df[cols_numericas_receta].fillna(0)

    cols_flags = [
        "Compatible_Vegana", "Compatible_Vegetariana", "Compatible_BajaCarbo",
        "Contiene_Lactosa", "Compatible_SinGluten",
        "Sin_Frutos_Secos", "Alto_Proteico", "Bajo_En_Grasa", "Bajo_En_Sodio",
        "Alto_En_Fibra", "Apto_Diabetico"
    ]

    for col in cols_flags:
        if col in df.columns:
            df[col] = df[col].apply(si_no_a_bin)
        else:
            df[col] = 0

    cat_cols = []

    if "Tipo_Comida" in df.columns:
        dummies_tipo = pd.get_dummies(df["Tipo_Comida"], prefix="Tipo_Comida")
        df = pd.concat([df, dummies_tipo], axis=1)
        cat_cols += list(dummies_tipo.columns)
    else:
        df["Tipo_Comida"] = ""

    if "Categoria_Plato" in df.columns:
        dummies_cat = pd.get_dummies(df["Categoria_Plato"], prefix="Categoria")
        df = pd.concat([df, dummies_cat], axis=1)
        cat_cols += list(dummies_cat.columns)
    else:
        df["Categoria_Plato"] = ""

    return df, cols_numericas_receta, cols_flags, cat_cols


# ============================================================
# 4) PREPROCESAMIENTO USUARIOS
# ============================================================
def preprocesar_usuarios(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    cols_numericas_usuario = [
        "ID_Usuario", "Edad", "Peso", "Talla",
        "Comidas_Diarias", "Calorias_Ajustadas",
        "Macro_Prot_g", "Macro_Carb_g", "Macro_Grasas_g"
    ]
    for col in cols_numericas_usuario:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0
    df[cols_numericas_usuario] = df[cols_numericas_usuario].fillna(0)

    if "Sexo" in df.columns:
        df["Sexo_M"] = (
            df["Sexo"].astype(str).str.strip().str.upper()
            .map({"M": 1, "F": 0})
            .fillna(0)
            .astype(int)
        )
    else:
        df["Sexo"] = "M"
        df["Sexo_M"] = 1

    map_act = {"bajo": 1, "moderado": 2, "alto": 3}
    if "Nivel_Actividad" in df.columns:
        df["Nivel_Actividad_val"] = (
            df["Nivel_Actividad"].astype(str).str.lower()
            .map(map_act)
            .fillna(2)
            .astype(int)
        )
    else:
        df["Nivel_Actividad"] = "Moderado"
        df["Nivel_Actividad_val"] = 2

    user_cat_cols_local = []

    if "Tipo_Dieta" in df.columns:
        dummies_dieta = pd.get_dummies(df["Tipo_Dieta"], prefix="Dieta")
        df = pd.concat([df, dummies_dieta], axis=1)
        user_cat_cols_local += list(dummies_dieta.columns)
    else:
        df["Tipo_Dieta"] = "Equilibrada"

    if "Objetivo_Nutricional" in df.columns:
        dummies_obj = pd.get_dummies(df["Objetivo_Nutricional"], prefix="Obj")
        df = pd.concat([df, dummies_obj], axis=1)
        user_cat_cols_local += list(dummies_obj.columns)
    else:
        df["Objetivo_Nutricional"] = "Mantener"

    return df, user_cat_cols_local


# ============================================================
# 5) SCORE HEURÍSTICO PARA (USUARIO, RECETA)
# ============================================================
def calcular_score_heuristico(row):
    comidas = row.get("Comidas_Diarias", 3)
    if comidas <= 0:
        comidas = 3

    target_cal = row.get("Calorias_Ajustadas", 0) / comidas
    target_prot = row.get("Macro_Prot_g", 0) / comidas
    target_carb = row.get("Macro_Carb_g", 0) / comidas
    target_grasa = row.get("Macro_Grasas_g", 0) / comidas

    diff_cal = abs(row.get("Calorias", 0) - target_cal)
    diff_prot = abs(row.get("Proteinas", 0) - target_prot)
    diff_carb = abs(row.get("Carbohidratos", 0) - target_carb)
    diff_grasa = abs(row.get("Grasas", 0) - target_grasa)

    penalty_restr = 0.0
    restricciones = str(row.get("Restricciones", "")).lower()

    if "sin lactosa" in restricciones and row.get("Contiene_Lactosa", 0) == 1:
        penalty_restr += 200

    if "sin gluten" in restricciones and row.get("Compatible_SinGluten", 0) == 0:
        penalty_restr += 200

    if ("diabético" in restricciones or "diabetico" in restricciones) and row.get("Apto_Diabetico", 0) == 0:
        penalty_restr += 150

    if "hipertenso" in restricciones and row.get("Bajo_En_Sodio", 0) == 0:
        penalty_restr += 100

    objetivo = str(row.get("Objetivo_Nutricional", "")).lower()
    bonus = 0.0

    if "disminuir_peso" in objetivo:
        if row.get("Bajo_En_Grasa", 0) == 1:
            bonus += 50
        if row.get("Azucares", 0) < 5:
            bonus += 40
        if row.get("Calorias", 0) < target_cal:
            bonus += 30

    if "ganar_músculo" in objetivo or "ganar_musculo" in objetivo:
        if row.get("Alto_Proteico", 0) == 1:
            bonus += 80
        if diff_prot < 10:
            bonus += 40

    if "mantener" in objetivo:
        if row.get("Bajo_En_Grasa", 0) == 1:
            bonus += 30
        if diff_cal < 50:
            bonus += 30

    w_cal = 0.2
    w_prot = 5.0
    w_carb = 1.0
    w_grasa = 3.0

    costo = (
        w_cal * diff_cal +
        w_prot * diff_prot +
        w_carb * diff_carb +
        w_grasa * diff_grasa +
        penalty_restr
    )

    score = bonus - costo
    return score


# ============================================================
# 6) REQUERIMIENTOS Y CONSTRUCCIÓN USUARIO
# ============================================================
def calcular_requerimientos_nutricionales(sexo, edad, peso, talla, nivel_actividad, objetivo_nutricional):
    sexo = str(sexo).upper()
    if sexo == "M":
        bmr = 10 * peso + 6.25 * talla - 5 * edad + 5
    else:
        bmr = 10 * peso + 6.25 * talla - 5 * edad - 161

    nivel = str(nivel_actividad).lower()
    if nivel == "bajo":
        factor_act = 1.2
    elif nivel == "moderado":
        factor_act = 1.55
    else:
        factor_act = 1.75

    tdee = bmr * factor_act

    obj = str(objetivo_nutricional).lower()
    if "disminuir_peso" in obj:
        calorias_ajustadas = tdee * 0.8
        prot_kg = 1.8
    elif "ganar_músculo" in obj or "ganar_musculo" in obj:
        calorias_ajustadas = tdee * 1.1
        prot_kg = 2.0
    else:
        calorias_ajustadas = tdee
        prot_kg = 1.6

    macro_prot_g = prot_kg * peso

    restantes_kcal = calorias_ajustadas - macro_prot_g * 4
    grasas_kcal = restantes_kcal * 0.30
    carb_kcal = restantes_kcal * 0.70

    macro_grasas_g = grasas_kcal / 9
    macro_carb_g = carb_kcal / 4

    return (
        round(calorias_ajustadas, 2),
        round(macro_prot_g, 2),
        round(macro_carb_g, 2),
        round(macro_grasas_g, 2)
    )


def calcular_requerimientos_api(payload: dict) -> dict:
    calorias, prot, carb, grasas = calcular_requerimientos_nutricionales(
        sexo=payload["sexo"],
        edad=payload["edad"],
        peso=payload["peso"],
        talla=payload["talla"],
        nivel_actividad=payload["nivel_actividad"],
        objetivo_nutricional=payload["objetivo_nutricional"],
    )

    comidas_raw = payload.get("comidas_diarias")
    try:
        comidas = int(comidas_raw)
    except (TypeError, ValueError):
        comidas = 3
    comidas = max(1, comidas)

    return {
        "calorias_diarias": calorias,
        "proteinas_g": prot,
        "carbohidratos_g": carb,
        "grasas_g": grasas,
        "comidas_diarias": comidas,
        "calorias_por_comida": round(calorias / comidas, 2),
        "proteinas_por_comida_g": round(prot / comidas, 2),
        "carbohidratos_por_comida_g": round(carb / comidas, 2),
        "grasas_por_comida_g": round(grasas / comidas, 2),
    }


def construir_usuario_desde_parametros(
    sexo,
    edad,
    peso,
    talla,
    nivel_actividad,
    comidas_diarias,
    tipo_dieta,
    restricciones_lista,
    objetivo_nutricional
):
    global user_cat_cols

    restricciones_str = "; ".join(restricciones_lista) if restricciones_lista else "Ninguna"

    calorias_ajustadas, macro_prot_g, macro_carb_g, macro_grasas_g = calcular_requerimientos_nutricionales(
        sexo=sexo,
        edad=edad,
        peso=peso,
        talla=talla,
        nivel_actividad=nivel_actividad,
        objetivo_nutricional=objetivo_nutricional
    )

    data = {
        "ID_Usuario": 0,
        "Sexo": sexo,
        "Edad": edad,
        "Peso": peso,
        "Talla": talla,
        "Nivel_Actividad": nivel_actividad,
        "Comidas_Diarias": comidas_diarias,
        "Tipo_Dieta": tipo_dieta,
        "Restricciones": restricciones_str,
        "Objetivo_Nutricional": objetivo_nutricional,
        "Calorias_Ajustadas": calorias_ajustadas,
        "Macro_Prot_g": macro_prot_g,
        "Macro_Carb_g": macro_carb_g,
        "Macro_Grasas_g": macro_grasas_g
    }

    user_df = pd.DataFrame([data])

    cols_numericas_usuario_loc = [
        "ID_Usuario", "Edad", "Peso", "Talla",
        "Comidas_Diarias", "Calorias_Ajustadas",
        "Macro_Prot_g", "Macro_Carb_g", "Macro_Grasas_g"
    ]
    for col in cols_numericas_usuario_loc:
        user_df[col] = pd.to_numeric(user_df[col], errors="coerce")
    user_df[cols_numericas_usuario_loc] = user_df[cols_numericas_usuario_loc].fillna(0)

    user_df["Sexo_M"] = (
        user_df["Sexo"].astype(str).str.strip().str.upper()
        .map({"M": 1, "F": 0})
        .fillna(0)
        .astype(int)
    )

    map_act = {"bajo": 1, "moderado": 2, "alto": 3}
    user_df["Nivel_Actividad_val"] = (
        user_df["Nivel_Actividad"].astype(str).str.lower()
        .map(map_act)
        .fillna(2)
        .astype(int)
    )

    dummies_dieta = pd.get_dummies(user_df["Tipo_Dieta"], prefix="Dieta")
    dummies_obj = pd.get_dummies(user_df["Objetivo_Nutricional"], prefix="Obj")
    user_df = pd.concat([user_df, dummies_dieta, dummies_obj], axis=1)

    for col in user_cat_cols:
        if col not in user_df.columns:
            user_df[col] = 0

    return user_df


# ============================================================
# 7) FILTRADO DURO
# ============================================================
def filtrar_recetas_por_restricciones_y_exclusiones(
    recetas_df,
    tipo_dieta,
    restricciones_lista,
    excluir_ids=None,
    excluir_nombres=None
):
    df = recetas_df.copy()

    restricciones_lower = [r.lower() for r in (restricciones_lista or [])]
    dieta_lower = str(tipo_dieta).lower()

    if dieta_lower == "vegana":
        df = df[df["Compatible_Vegana"] == 1]

    if dieta_lower == "vegetariana":
        df = df[df["Compatible_Vegetariana"] == 1]

    if any("sin lactosa" in r for r in restricciones_lower):
        df = df[df["Contiene_Lactosa"] == 0]

    if any("sin gluten" in r for r in restricciones_lower):
        df = df[df["Compatible_SinGluten"] == 1]

    if any("alergia a frutos secos" in r for r in restricciones_lower):
        df = df[df["Sin_Frutos_Secos"] == 1]

    if any("diabético" in r or "diabetico" in r for r in restricciones_lower):
        df = df[(df["Azucares"] <= 10) & (df["Carbohidratos"] <= 30)]

    if any("hipertenso" in r for r in restricciones_lower):
        df = df[df["Sodio"] <= 150]

    if excluir_ids:
        df = df[~df["ID_Receta"].isin(excluir_ids)]

    if excluir_nombres and "Nombre_Receta" in df.columns:
        df = df[~df["Nombre_Receta"].isin(excluir_nombres)]

    return df


# ============================================================
# 8) ARMAR PLAN POR DÍAS  (✅ AQUÍ ESTÁ LA CORRECCIÓN)
# ============================================================
def armar_plan_por_dias(recs_ordenadas, dias, comidas_diarias, almuerzo_cena_misma: bool = False, seed: int = 42):
    """
    FIX:
    - Normaliza Tipo_Comida/Categoria_Plato (lower/strip) para que los pools no queden vacíos por strings distintos.
    - almuerzo_cena_misma=True => Cena = Almuerzo del MISMO día, pero el Almuerzo varía entre días (si hay opciones).
    - almuerzo_cena_misma=False => intenta Cena != Almuerzo, y evita snacks/postres en Cena por fallback.
    """
    plan = []

    # 1) Slots
    if comidas_diarias >= 5:
        slots = ["Desayuno", "Almuerzo", "Cena", "Snack1", "Snack2"]
    elif comidas_diarias == 4:
        slots = ["Desayuno", "Almuerzo", "Cena", "Snack"]
    elif comidas_diarias == 3:
        slots = ["Desayuno", "Almuerzo", "Cena"]
    else:
        slots_base = ["Desayuno", "Almuerzo", "Cena", "Snack1", "Snack2"]
        slots = slots_base[:max(1, comidas_diarias)]

    # 2) Copia + normalización para clasificación robusta
    df = recs_ordenadas.copy()

    if "Tipo_Comida" not in df.columns:
        df["Tipo_Comida"] = ""
    if "Categoria_Plato" not in df.columns:
        df["Categoria_Plato"] = ""

    df["_tipo"] = df["Tipo_Comida"].astype(str).str.strip().str.lower()
    df["_cat"]  = df["Categoria_Plato"].astype(str).str.strip().str.lower()

    # Helpers de pertenencia (tolerantes a variaciones)
    def is_breakfast(row):
        return row["_tipo"] in {"desayuno"}

    def is_lunch_like(row):
        # acepta "almuerzo" o categorias equivalentes a plato fuerte
        return (row["_tipo"] in {"almuerzo"}) or ("plato" in row["_cat"] and "fuerte" in row["_cat"])

    def is_dinner_like(row):
        return (row["_tipo"] in {"cena"}) or ("plato" in row["_cat"] and "fuerte" in row["_cat"])

    def is_snack_like(row):
        # meriendas, snacks y postres
        if row["_tipo"] in {"merienda", "snack"}:
            return True
        if "snack" in row["_cat"] or "postre" in row["_cat"]:
            return True
        return False

    # 3) Pools (en orden score_modelo desc ya viene df)
    pool_desayuno = df[df.apply(is_breakfast, axis=1)].to_dict("records")
    pool_almuerzo = df[df.apply(is_lunch_like, axis=1)].to_dict("records")
    pool_cena     = df[df.apply(is_dinner_like, axis=1)].to_dict("records")
    pool_snack    = df[df.apply(is_snack_like, axis=1)].to_dict("records")

    # "todo" excluyendo snacks (para fallback de almuerzo/cena)
    pool_todo_no_snack = df[~df.apply(is_snack_like, axis=1)].to_dict("records")
    pool_todo = df.to_dict("records")

    # 4) Selector (mejor score primero, con memoria y bloqueos)
    def pick_from_pool(pool, usadas_global, usadas_hoy, banned_ids=None):
        if banned_ids is None:
            banned_ids = set()
        # intento A: nuevo global y no usado hoy
        for r in pool:
            rid = r.get("ID_Receta")
            if rid is None:
                continue
            if rid not in usadas_global and rid not in usadas_hoy and rid not in banned_ids:
                return r
        # intento B: no usado hoy
        for r in pool:
            rid = r.get("ID_Receta")
            if rid is None:
                continue
            if rid not in usadas_hoy and rid not in banned_ids:
                return r
        return None

    usadas_global = set()

    for dia in range(1, dias + 1):
        usadas_hoy = set()
        almuerzo_hoy = None

        for slot in slots:
            receta_sel = None

            # Definir pool según slot
            if "Desayuno" in slot:
                base_pool = pool_desayuno
            elif "Almuerzo" in slot:
                base_pool = pool_almuerzo
            else:
                base_pool = pool_cena if ("Cena" in slot) else pool_snack

            # --- Lógica por slot ---
            if "Cena" in slot and almuerzo_cena_misma and almuerzo_hoy is not None:
                # Cena = almuerzo del mismo día
                receta_sel = almuerzo_hoy
            else:
                banned = set()

                # Si es Cena y NO deben ser iguales, prohibimos el almuerzo
                if "Cena" in slot and (not almuerzo_cena_misma) and almuerzo_hoy is not None:
                    banned.add(almuerzo_hoy["ID_Receta"])

                receta_sel = pick_from_pool(base_pool, usadas_global, usadas_hoy, banned_ids=banned)

                # Fallbacks inteligentes:
                if receta_sel is None:
                    if "Cena" in slot:
                        # Para cena, NO uses pool_todo con snacks/postres: usa pool_todo_no_snack
                        receta_sel = pick_from_pool(pool_todo_no_snack, usadas_global, usadas_hoy, banned_ids=banned)
                        # Si aún nada, intenta pool_almuerzo (platos fuertes) excluyendo almuerzo_hoy
                        if receta_sel is None and almuerzo_hoy is not None:
                            receta_sel = pick_from_pool(pool_almuerzo, usadas_global, usadas_hoy, banned_ids=banned)
                    elif "Almuerzo" in slot:
                        # Almuerzo: fallback a "todo sin snack"
                        receta_sel = pick_from_pool(pool_todo_no_snack, usadas_global, usadas_hoy)
                    else:
                        # Snacks: fallback a todo (si no hay snack)
                        receta_sel = pick_from_pool(pool_todo, usadas_global, usadas_hoy)

            # Último recurso (si TODO está vacío)
            if receta_sel is None:
                if len(pool_todo) == 0:
                    raise RuntimeError("No hay recetas disponibles tras el filtrado.")
                receta_sel = pool_todo[dia % len(pool_todo)]

            # Guardar almuerzo del día
            if "Almuerzo" in slot:
                almuerzo_hoy = receta_sel

            # Memoria global: siempre registrar la receta del slot,
            # EXCEPTO cuando es Cena clonada del almuerzo (para no “gastar” dos veces la misma receta)
            if not ("Cena" in slot and almuerzo_cena_misma and almuerzo_hoy is not None):
                usadas_global.add(receta_sel["ID_Receta"])

            usadas_hoy.add(receta_sel["ID_Receta"])

            plan.append({
                "Dia": dia,
                "Comida": slot,
                "ID_Receta": receta_sel.get("ID_Receta"),
                "Nombre_Receta": receta_sel.get("Nombre_Receta"),
                "Tipo_Comida": receta_sel.get("Tipo_Comida", ""),
                "Calorias": receta_sel.get("Calorias", 0),
                "Proteinas": receta_sel.get("Proteinas", 0),
                "Grasas": receta_sel.get("Grasas", 0),
                "Carbohidratos": receta_sel.get("Carbohidratos", 0),
                "score_modelo": receta_sel.get("score_modelo", 0)
            })

    return pd.DataFrame(plan)


# ============================================================
# 9) FUNCIÓN PRINCIPAL de RECOMENDACIÓN
# ============================================================
def recomendar_recetas_y_plan(
    sexo,
    edad,
    peso,
    talla,
    nivel_actividad,
    comidas_diarias,
    tipo_dieta,
    restricciones_lista,
    objetivo_nutricional,
    dias,
    top_n_recetas=50,
    excluir_ids=None,
    excluir_nombres=None,
    almuerzo_cena_misma: bool = False,
):
    global recetas_df, feature_cols, modelo

    if not _MODELO_LISTO or modelo is None or recetas_df is None or feature_cols is None:
        raise RuntimeError("Modelo no inicializado. Llama a inicializar_modelo() al iniciar la API.")

    user_df = construir_usuario_desde_parametros(
        sexo=sexo,
        edad=edad,
        peso=peso,
        talla=talla,
        nivel_actividad=nivel_actividad,
        comidas_diarias=comidas_diarias,
        tipo_dieta=tipo_dieta,
        restricciones_lista=restricciones_lista,
        objetivo_nutricional=objetivo_nutricional
    )

    recetas_filtradas = filtrar_recetas_por_restricciones_y_exclusiones(
        recetas_df,
        tipo_dieta=tipo_dieta,
        restricciones_lista=restricciones_lista,
        excluir_ids=excluir_ids,
        excluir_nombres=excluir_nombres
    )

    user_df = user_df.copy()
    user_df["key"] = 1
    recs = recetas_filtradas.copy()
    recs["key"] = 1

    pares = pd.merge(user_df, recs, on="key", suffixes=("_user", "_receta"))
    pares.drop(columns=["key"], inplace=True)

    for col in feature_cols:
        if col not in pares.columns:
            pares[col] = 0

    pares_X = pares[feature_cols].copy()
    pares["score_modelo"] = modelo.predict(pares_X)

    pares["score_heuristico"] = pares.apply(calcular_score_heuristico, axis=1)

    pares_ordenado = pares.sort_values("score_modelo", ascending=False).reset_index(drop=True)

    cols_salida = [
        "ID_Receta", "Nombre_Receta", "Tipo_Comida", "Categoria_Plato",
        "Calorias", "Proteinas", "Grasas", "Carbohidratos",
        "Fibra", "Azucares", "Sodio",
        "Compatible_Vegana", "Compatible_Vegetariana",
        "Compatible_BajaCarbo", "Contiene_Lactosa", "Compatible_SinGluten",
        "Sin_Frutos_Secos",
        "Bajo_En_Sodio", "Alto_Proteico", "Bajo_En_Grasa", "Alto_En_Fibra",
        "Apto_Diabetico",
        "score_modelo", "score_heuristico"
    ]
    cols_salida = [c for c in cols_salida if c in pares_ordenado.columns]
    recomendaciones_top = pares_ordenado[cols_salida].head(top_n_recetas)

    seed = int(edad) * 100 + int(peso)
    plan_df = armar_plan_por_dias(
        recs_ordenadas=pares_ordenado,
        dias=dias,
        comidas_diarias=comidas_diarias,
        almuerzo_cena_misma=almuerzo_cena_misma,
        seed=seed
    )

    return recomendaciones_top, plan_df


def recomendar_recetas_y_plan_api(payload: dict):
    requerimientos = calcular_requerimientos_api(payload)

    almuerzo_cena_misma = bool(
        payload.get("almuerzoCenaMisma", payload.get("almuerzo_cena_misma", False))
    )

    recs_df, plan_df = recomendar_recetas_y_plan(
        sexo=payload["sexo"],
        edad=payload["edad"],
        peso=payload["peso"],
        talla=payload["talla"],
        nivel_actividad=payload["nivel_actividad"],
        comidas_diarias=payload["comidas_diarias"],
        tipo_dieta=payload["tipo_dieta"],
        restricciones_lista=payload.get("restricciones", []),
        objetivo_nutricional=payload["objetivo_nutricional"],
        dias=int(payload.get("dias_plan", 7)),
        top_n_recetas=int(payload.get("top_n_recetas", 50)),
        excluir_ids=payload.get("excluir_ids"),
        excluir_nombres=payload.get("excluir_nombres"),
        almuerzo_cena_misma=almuerzo_cena_misma,
    )

    cols_flags_resp = [
        "Compatible_Vegana", "Compatible_Vegetariana", "Compatible_BajaCarbo",
        "Contiene_Lactosa", "Compatible_SinGluten",
        "Sin_Frutos_Secos", "Alto_Proteico", "Bajo_En_Grasa", "Bajo_En_Sodio",
        "Alto_En_Fibra", "Apto_Diabetico"
    ]

    for c in cols_flags_resp:
        if c in recs_df.columns:
            recs_df[c] = recs_df[c].apply(si_no_a_bin).astype(int)
        if c in plan_df.columns:
            plan_df[c] = plan_df[c].apply(si_no_a_bin).astype(int)

    return {
        "requerimientos": requerimientos,
        "recomendaciones": recs_df.to_dict(orient="records"),
        "plan": plan_df.to_dict(orient="records"),
    }


# ============================================================
# 10) ENTRENAMIENTO (solo 1 vez)  ✅ NO TOCADO
# ============================================================
def _entrenar_y_guardar():
    global recetas_df, feature_cols, user_cat_cols, modelo, _MODELO_LISTO

    recetas_raw = pd.read_csv(RUTA_RECETAS)
    usuarios_raw = pd.read_csv(RUTA_USUARIOS)

    recetas_prep, cols_numericas_receta, cols_flags, cat_cols = preprocesar_recetas(recetas_raw)
    usuarios_prep, user_cat_cols_local = preprocesar_usuarios(usuarios_raw)

    usuarios_sample = usuarios_prep.copy()
    recetas_sample = recetas_prep.copy()

    usuarios_sample["key"] = 1
    recetas_sample["key"] = 1

    pairs_df = pd.merge(
        usuarios_sample,
        recetas_sample,
        on="key",
        suffixes=("_user", "_receta")
    )
    pairs_df.drop(columns=["key"], inplace=True)

    pairs_df["score_heuristico"] = pairs_df.apply(calcular_score_heuristico, axis=1)

    feature_cols_usuario = [
        "Edad", "Peso", "Talla", "Comidas_Diarias",
        "Calorias_Ajustadas", "Macro_Prot_g", "Macro_Carb_g", "Macro_Grasas_g",
        "Sexo_M", "Nivel_Actividad_val"
    ] + user_cat_cols_local

    feature_cols_receta = cols_numericas_receta + cols_flags + cat_cols

    feature_cols_usuario = [c for c in feature_cols_usuario if c in pairs_df.columns]
    feature_cols_receta = [c for c in feature_cols_receta if c in pairs_df.columns]

    feature_cols_local = feature_cols_usuario + feature_cols_receta

    X = pairs_df[feature_cols_local].copy()
    y = pairs_df["score_heuristico"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo_local = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    modelo_local.fit(X_train, y_train)
    y_pred = modelo_local.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    print(f"[NutriSmart] Entrenamiento OK | MSE={mse:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
    print(f"[NutriSmart] Features: {len(feature_cols_local)} | Pairs: {pairs_df.shape}")

    recetas_df = recetas_prep
    feature_cols = feature_cols_local
    user_cat_cols = user_cat_cols_local
    modelo = modelo_local
    _MODELO_LISTO = True

    joblib.dump(modelo_local, MODEL_PATH)
    joblib.dump(
        {
            "feature_cols": feature_cols_local,
            "user_cat_cols": user_cat_cols_local
        },
        META_PATH
    )
    joblib.dump(recetas_prep, RECETAS_PREP_PATH)

    fp = _fingerprint_datasets()
    DATA_FINGERPRINT_PATH.write_text(fp, encoding="utf-8")


# ============================================================
# 11) CARGA ARTEFACTOS (arranque rápido)
# ============================================================
def _fingerprint_datasets() -> str:
    try:
        m1 = os.path.getmtime(RUTA_RECETAS)
        m2 = os.path.getmtime(RUTA_USUARIOS)
        return f"{m1}-{m2}"
    except Exception:
        return "unknown"


def _artefactos_existen() -> bool:
    return MODEL_PATH.exists() and META_PATH.exists() and RECETAS_PREP_PATH.exists()


def _artefactos_validos() -> bool:
    if not _artefactos_existen():
        return False
    if DATA_FINGERPRINT_PATH.exists():
        fp_guardado = DATA_FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
        fp_actual = _fingerprint_datasets()
        return fp_guardado == fp_actual
    return True


def _cargar_artefactos():
    global recetas_df, feature_cols, user_cat_cols, modelo, _MODELO_LISTO

    modelo = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    feature_cols = meta["feature_cols"]
    user_cat_cols = meta["user_cat_cols"]
    recetas_df = joblib.load(RECETAS_PREP_PATH)

    _MODELO_LISTO = True
    print("[NutriSmart] Modelo cargado desde artifacts/ (sin reentrenar).")


def inicializar_modelo():
    global _MODELO_LISTO
    if _MODELO_LISTO:
        return

    if MODEL_PATH.exists() and META_PATH.exists() and RECETAS_PREP_PATH.exists():
        _cargar_artefactos()
        return

    print("[NutriSmart] No hay artefactos válidos. Entrenando 1 vez...")
    _entrenar_y_guardar()
    print("[NutriSmart] Artefactos guardados. Próximos arranques cargarán rápido.")


if __name__ == "__main__":
    print("Entrenamiento manual iniciado")
    _entrenar_y_guardar()
