from app.models.entrenamiento_model import InfoEntrenamiento
from app.models.prediccion_model import PrediccionModel
from app.services.mlearning_service import MLearningService
from app.utils.utils import Utils

'''
esta clase tiene los llamados a los diferentes metodos que ofrece el servicio de mlearning 
'''
class MLearningController:

    def __init__(self):
        self.mlearning_service = MLearningService()
        self.utils = Utils()


    # El siguiente metodo define el llamado a cada uno de los modelos que estan en los servicios y devuelve un mensaje con las metricas obtenidas en el entrenamiento

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

#metodo que iniciar el servicio de prediccion   
    def prediccion(self, prediccion: PrediccionModel, nombre_dataset):
        return self.mlearning_service.prediccion(prediccion, nombre_dataset)

#metodo para obtener los mejores resultados obtenidos por los diferentes algoritmos o modelos
    def obtener_mejores_algoritmos(self, nombre_dataset: str):
        return self.mlearning_service.obtener_top3_algoritmos(nombre_dataset=nombre_dataset)
   