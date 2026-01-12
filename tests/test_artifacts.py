import modeloV01 as m

# -----------------------------
# UT-ML-13: validación de artefactos y fingerprint
# -----------------------------

def test_artefactos_existen_y_validos(tmp_path, monkeypatch):
    model = tmp_path / "rf_model.joblib"
    meta  = tmp_path / "meta.joblib"
    recs  = tmp_path / "recetas_preprocesadas.joblib"
    fp    = tmp_path / "data_fingerprint.txt"

    # inyectar rutas temporales
    monkeypatch.setattr(m, "MODEL_PATH", model)
    monkeypatch.setattr(m, "META_PATH", meta)
    monkeypatch.setattr(m, "RECETAS_PREP_PATH", recs)
    monkeypatch.setattr(m, "DATA_FINGERPRINT_PATH", fp)

    print("\n[UT-ML-13] artefactos_existen_y_validos")
    print("  Objetivo: Validar existencia y validez de artefactos del modelo usando fingerprint.")
    print("  Rutas temporales usadas:")
    print(f"    MODEL_PATH={model}")
    print(f"    META_PATH={meta}")
    print(f"    RECETAS_PREP_PATH={recs}")
    print(f"    DATA_FINGERPRINT_PATH={fp}")

    # -------------------------------------------------
    # Caso 1: no existen artefactos
    # -------------------------------------------------
    print("\n  Caso 1: Artefactos NO existen")
    existen = m._artefactos_existen()
    validos = m._artefactos_validos()

    print(f"    _artefactos_existen() -> {existen}")
    print(f"    _artefactos_validos() -> {validos}")
    print("  Esperado: existen=False, validos=False")

    assert existen is False
    assert validos is False

    # -------------------------------------------------
    # Caso 2: existen artefactos y fingerprint coincide
    # -------------------------------------------------
    print("\n  Caso 2: Artefactos existen y fingerprint coincide")

    model.write_bytes(b"dummy")
    meta.write_bytes(b"dummy")
    recs.write_bytes(b"dummy")

    # fingerprint estable
    monkeypatch.setattr(m, "_fingerprint_datasets", lambda: "ABC")
    fp.write_text("ABC", encoding="utf-8")

    existen = m._artefactos_existen()
    validos = m._artefactos_validos()

    print(f"    _artefactos_existen() -> {existen}")
    print(f"    _artefactos_validos() -> {validos}")
    print("  Esperado: existen=True, validos=True")

    assert existen is True
    assert validos is True

    # -------------------------------------------------
    # Caso 3: fingerprint cambia (datasets modificados)
    # -------------------------------------------------
    print("\n  Caso 3: Fingerprint cambia (datasets modificados)")
    fp.write_text("XYZ", encoding="utf-8")

    validos = m._artefactos_validos()

    print(f"    _artefactos_validos() -> {validos}")
    print("  Esperado: validos=False")

    assert validos is False
