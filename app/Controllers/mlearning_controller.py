from app.services.mlearning_service import MLearningService


class MLearningController:

    def __init__(self):
        self.mlearning_service = MLearningService()

    def knn(self, entrenamiento):
        return self.mlearning_service.knn(entrenamiento)
    
    def reg_logistica(self):
        #return self.mlearning_service.regresion_logistica()
        return "ok"