from app.services.mlearning_service import MLearningService


class MLearningController:

    def __init__(self):
        self.mlearning_service = MLearningService()

    def knn(self):
        self.mlearning_service.knn()
        return "KNN"