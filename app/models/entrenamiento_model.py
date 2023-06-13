from pydantic import BaseModel
from typing import List

class InfoEntrenamiento(BaseModel):
    nombre_algoritmo: str
    columnas_x: List[str]
    objetivo_y: str
    tecnica: str
    cantidad: int
    normalizacion: str
    nombre_dataset: str
