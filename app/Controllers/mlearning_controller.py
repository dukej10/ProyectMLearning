from app.models.entrenamiento_model import InfoEntrenamiento
from app.services.mlearning_service import MLearningService


class MLearningController:

    def __init__(self):
        self.mlearning_service = MLearningService()

    def knn(self, entrenamiento: InfoEntrenamiento):
        if entrenamiento.nombre_algoritmo.upper() == "KNN":
            return self.mlearning_service.knn(entrenamiento)
        elif entrenamiento.nombre_algoritmo.upper() == "NAIVE BAYES":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "REGRESION LOGISTICA":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "SVM":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "ARBOL DE DECISION":
            pass
        elif entrenamiento.nombre_algoritmo.upper() == "REGRESION LOGISTICA":
            pass
        else:
            return "Algoritmo no encontrado"

    
    def reg_logistica(self):
        #return self.mlearning_service.regresion_logistica()
        return "ok"