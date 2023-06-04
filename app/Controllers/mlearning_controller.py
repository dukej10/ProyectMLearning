from app.models.entrenamiento_model import InfoEntrenamiento
from app.models.prediccion_model import PrediccionModel
from app.services.mlearning_service import MLearningService


class MLearningController:

    def __init__(self):
        self.mlearning_service = MLearningService()

    def algoritmos(self, entrenamiento: InfoEntrenamiento):
        if entrenamiento.nombre_algoritmo.upper() == "KNN":
            return self.mlearning_service.knn(entrenamiento)
        elif entrenamiento.nombre_algoritmo.upper() == "NAIVE BAYES":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "REGRESION LOGISTICA":
            return self.mlearning_service.regresion_logistica(entrenamiento)
        elif entrenamiento.nombre_algoritmo.upper() == "SVM":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "ARBOL DE DECISION":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "REGRESION LINEAL":
            pass
        else:
            return "Algoritmo no encontrado"
        
    def prediccion(self, prediccion: PrediccionModel):
        return self.mlearning_service.prediccion(prediccion)