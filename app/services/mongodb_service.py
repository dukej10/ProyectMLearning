from pymongo import MongoClient
from app.db.db import MongoConnection

class MongoDBService:
    def __init__(self):
        mongoDB = MongoConnection()
        self.client = mongoDB.get_mongo_connection()
        self.db = self.client['MLearning']
        self.collection = None

    def guardar_json(self, datos_json, coleccion):
        try:
            self.collection = self.db[coleccion]
            result = self.collection.insert_one(datos_json)
            print(f'JSON guardado en MongoDB : {str(result)}')
            return result.inserted_id
        except Exception as e:
            print(f'Error al guardar el JSON en MongoDB: {str(e)}')
            return None
        
    def guardar_json_metricas(self, datos_json, coleccion):
        try:
            self.collection = self.db[coleccion]
            
            # Verificar si existe un registro con el nombre del algoritmo, normalización y técnica
            existe_registro = self.collection.find_one({
                "nombre_algoritmo": datos_json['nombre_algoritmo'],
                "normalizacion": datos_json['normalizacion'],
                "tecnica": datos_json['tecnica']
            })
            
            if existe_registro:
                # Actualizar los datos del registro existente
                result = self.collection.update_one(
                    {
                        "nombre_algoritmo": datos_json['nombre_algoritmo'],
                        "normalizacion": datos_json['normalizacion'],
                        "tecnica": datos_json['tecnica'],
                        "fecha": datos_json['fecha']

                    },
                    {"$set": datos_json}
                )
                print(f'Datos actualizados en MongoDB: {str(result)}')
                return result.upserted_id
            else:
                # Insertar un nuevo registro
                result = self.collection.insert_one(datos_json)
                print(f'JSON guardado en MongoDB: {str(result)}')
                return result.inserted_id
        except Exception as e:
            print(f'Error al guardar o actualizar el JSON en MongoDB: {str(e)}')
            return None
    

    def obtener_ultimo_registro(self, coleccion):
        try:
            self.collection = self.db[coleccion]
            result = self.collection.find_one(sort=[('_id', -1)])
            #print(f'JSON obtenido en MongoDB\n : {str(result)}')
            # print(type(result))
            return result
        except Exception as e:
            print(f'Error al obtener el ultimo registro: {str(e)}')
            return None
        
    def obtener_fecha_mas_reciente(self, coleccion):
        try:
            self.collection = self.db[coleccion]
            result = self.collection.find_one(sort=[('fecha', -1)])
            # print(f'Fecha más recientess: {result}')
            if result:
                fecha_mas_reciente = result['fecha']
                return fecha_mas_reciente
            return ""
        except Exception as e:
            print(f'Error al obtener la fecha más reciente: {str(e)}')
            return None


    def obtener_registros_metricas_recientes(self, coleccion):
        try:
            fecha_mas_reciente = self.obtener_fecha_mas_reciente(coleccion)
            print(f'Fecha más reciente: {fecha_mas_reciente}')
            if fecha_mas_reciente != "":
                self.collection = self.db[coleccion]
                result = self.collection.find(
                    filter={'fecha': fecha_mas_reciente}
                )
                registros = [registro for registro in result]
                return registros
            else:
                return []
        except Exception as e:
            print(f'Error al obtener los registros de la fecha más reciente: {str(e)}')
            return None

    