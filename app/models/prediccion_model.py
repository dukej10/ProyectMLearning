from typing import Dict, Union
from pydantic import BaseModel

'''
clase que define los parametros para la prediccion del modelo
'''
class PrediccionModel(BaseModel):
    algoritmo: str
    valores_predecir: Dict[str, Union[int, float, str]]