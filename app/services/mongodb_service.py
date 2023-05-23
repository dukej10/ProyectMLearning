from pymongo import MongoClient
from app.db.db import MongoConnection

class MongoDBService:
    def __init__(self):
        mongoDB = MongoConnection()
        self.client = mongoDB.get_mongo_connection()
        self.db = self.client['MLearning']
        self.collection = self.db['Dataset']

    def guardar_json(self, datos_json):
        try:
            result = self.collection.insert_one(datos_json)
            print(f'JSON guardado en MongoDB : {str(result)}')
            self.obtener_ultimo_registro()
            return result.inserted_id
        except Exception as e:
            print(f'Error al guardar el JSON en MongoDB: {str(e)}')
            return None
        
    def obtener_ultimo_registro(self):
        try:
            result = self.collection.find_one(sort=[('_id', -1)])
            print(f'JSON obtenido en MongoDB\n : {str(result)}')
            print(type(result))
            return result
        except Exception as e:
            print(f'Error al obtener el ultimo registro: {str(e)}')
            return None
        
    