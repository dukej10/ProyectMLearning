from typing import Dict, Union
from pydantic import BaseModel


class PrediccionModel(BaseModel):
    algoritmo: str
    valores_predecir: Dict[str, Union[int, float, str]]