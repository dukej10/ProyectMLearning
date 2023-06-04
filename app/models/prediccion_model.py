from typing import Union
from pydantic import BaseModel


class PrediccionModel(BaseModel):
    algoritmo: str
    area: str
    categoria: str
    genero: str
    agrupa: str
    valor: Union[int, float]
    año: int
    mes: str