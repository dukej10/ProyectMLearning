from app.models.entrenamiento_model import InfoEntrenamiento
from app.models.prediccion_model import PrediccionModel
from app.services.mlearning_service import MLearningService
from app.utils.utils import Utils


class MLearningController:

    def __init__(self):
        self.mlearning_service = MLearningService()
        self.utils = Utils()

    def algoritmos(self, entrenamiento: InfoEntrenamiento):
        
        entrenamiento.nombre_algoritmo = self.utils.arreglar_nombre(entrenamiento.nombre_algoritmo)
        if entrenamiento.nombre_algoritmo == "KNN":
            return self.utils.prueba(msg="Métricas algormitmo KNN",datos=self.mlearning_service.knn(entrenamiento))
        

        elif entrenamiento.nombre_algoritmo == "NAIVEBAYES":
            return self.utils.prueba(msg="Métricas algoritmo de Naive Bayes ", datos = self.mlearning_service.naive_bayes(entrenamiento))
        
        elif entrenamiento.nombre_algoritmo == "REGRESIONLOGISTICA":
            return self.utils.prueba(msg="Métricas algormitmo Regresion Logistica",datos=self.mlearning_service.regresion_logistica(entrenamiento))
        elif entrenamiento.nombre_algoritmo == "SVM":
            return self.utils.prueba(msg="Métricas algoritmo de Maquinas de soporte vectorial SVM ", datos = self.mlearning_service.svm(entrenamiento))
        
        elif entrenamiento.nombre_algoritmo == "ARBOLDEDECISION":
            return self.utils.prueba(msg="Métricas algormitmo Regresion Árbol de decisión",datos=self.mlearning_service.arbol_decision(entrenamiento))
        elif entrenamiento.nombre_algoritmo == "REGRESIONLINEAL":
            return self.utils.prueba(msg="Métricas algormitmo Regresion Lineal",datos=self.mlearning_service.regresion_lineal(entrenamiento))
        else:
            return "Algoritmo no encontrado"
        
    def prediccion(self, prediccion: PrediccionModel):
        return self.mlearning_service.prediccion(prediccion)
    
    def obtener_mejores_algoritmos(self):
        return self.mlearning_service.obtener_top3_algoritmos()
   