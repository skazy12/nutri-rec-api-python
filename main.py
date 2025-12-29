from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from modeloV01 import inicializar_modelo, recomendar_recetas_y_plan_api

app = FastAPI(title="NutriSmart Model API")


class UsuarioPayload(BaseModel):
    sexo: str
    edad: int
    peso: float
    talla: float
    nivel_actividad: str
    comidas_diarias: int
    tipo_dieta: str
    restricciones: List[str]
    objetivo_nutricional: str
    dias_plan: int = 7
    top_n_recetas: int = 30
    excluir_ids: Optional[List[int]] = None
    excluir_nombres: Optional[List[str]] = None


@app.on_event("startup")
def startup_event():
    # ✅ Carga artefactos o entrena SOLO 1 vez si no existen
    inicializar_modelo()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recomendar")
def recomendar(payload: UsuarioPayload):
    result = recomendar_recetas_y_plan_api(payload.dict())
    return result

