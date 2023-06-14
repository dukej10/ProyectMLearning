import re
from pymongo import MongoClient
from app.db.db import MongoConnection

'''
clase que maneja los servicios relacionados con mongoDB
'''

class MongoDBService:
    def __init__(self):
        mongoDB = MongoConnection()
        self.client = mongoDB.get_mongo_connection()
        self.db = self.client['MLearning']
        self.collection = None
    '''
    Guarda un objeto JSON en la colección especificada de MongoDB. Verifica si ya existe un registro con ciertos
      criterios y lo elimina antes de insertar el nuevo registro. Devuelve el ID del registro insertado.
    '''
    def guardar_json(self, datos_json, coleccion):
        try:
            # print("COLLECTION ", coleccion)
            self.collection = self.db[coleccion]
            
            existe_registro = None  # Inicializar la variable antes de la verificación
            
            if coleccion == 'RepresentacionCodificacion':
                existe_registro = self.collection.find_one({
                    "nombre_algoritmo": datos_json['nombre_algoritmo'],
                    "nombre_dataset": datos_json['nombre_dataset']
                })
            elif coleccion == 'Dataset':
                existe_registro = self.collection.find_one({
                    "nombreDoc": datos_json['nombreDoc'],
                    "version": datos_json['version'],
                    "nombre_dataset": datos_json['nombre_dataset']
                })
            
            if existe_registro:
               if coleccion == 'RepresentacionCodificacion':
                    self.collection.delete_one({"nombre_algoritmo": datos_json['nombre_algoritmo'], "nombre_dataset": datos_json['nombre_dataset']})
               elif coleccion == 'Dataset':
                    self.collection.delete_one({"nombreDoc": datos_json['nombreDoc'], "version": datos_json['version']})
            
            result = self.collection.insert_one(datos_json)
            print(f'JSON guardado en MongoDB : {str(result)}')
            return result.inserted_id
            
        except Exception as e:
            print(f'Error al guardar el JSON en MongoDB: {str(e)}')
            return None

    '''
    Guarda un objeto JSON de métricas en la colección especificada de MongoDB. Verifica si ya existe un registro con ciertos criterios (nombre del algoritmo, normalización, técnica y fecha) y,
      si existe, actualiza los datos del registro. Si no existe, inserta un nuevo registro. Devuelve el ID del registro insertado.
    '''   
    def guardar_json_metricas(self, datos_json, coleccion):
        try:
            self.collection = self.db[coleccion]
            
            # Verificar si existe un registro con el nombre del algoritmo, normalización y técnica
            existe_registro = self.collection.find_one({
                "nombre_algoritmo": datos_json['nombre_algoritmo'],
                "normalizacion": datos_json['normalizacion'],
                "tecnica": datos_json['tecnica'],
                "fecha": datos_json['fecha'],
                "nombre_dataset": datos_json['nombre_dataset']
            })
            
            if existe_registro:
                # Actualizar los datos del registro existente
                self.collection.delete_one({
                        "nombre_algoritmo": datos_json['nombre_algoritmo'],
                        "normalizacion": datos_json['normalizacion'],
                        "tecnica": datos_json['tecnica'],
                        "fecha": datos_json['fecha'],
                        "nombre_dataset": datos_json['nombre_dataset']

                    })

                # Insertar un nuevo registro
            result = self.collection.insert_one(datos_json)
            print(f'JSON guardado en MongoDB: {str(result)}')
            return result.inserted_id
        except Exception as e:
            print(f'Error al guardar o actualizar el JSON en MongoDB: {str(e)}')
            return None
    
    '''
    Obtiene el último registro de la colección especificada en MongoDB. Ordena los registros por ID de forma descendente y devuelve el primer registro.
    '''
    def obtener_ultimo_registro(self, coleccion, nombre_dataset):
        try:
            self.collection = self.db[coleccion]
            regex_pattern = f".*{re.escape(nombre_dataset)}.*"
            query = {"nombre_dataset": {"$regex": regex_pattern, "$options": "i"}}
            sort_query = [('_id', -1)]
            result = self.collection.find_one(query, sort=sort_query)
            #print(f'JSON obtenido en MongoDB\n : {str(result)}')
            # print(type(result))
            return result
        except Exception as e:
            print(f'Error al obtener el ultimo registro: {str(e)}')
            return None
        
    def obtener_ultimo_registro_por_nombre(self, coleccion, nombre):
        try:
            self.collection = self.db[coleccion]
            regex_pattern = f".*{re.escape(nombre)}.*"
            query = {"nombre_dataset": {"$regex": regex_pattern, "$options": "i"}}
            sort_query = [('_id', -1)]
            result = self.collection.find_one(query, sort=sort_query)
            # print(f'JSON obtenido en MongoDB\n : {str(result)}')
            # print(type(result))
            return result
        except Exception as e:
            print(f'Error al obtener el ultimo registro: {str(e)}')
            return None

    '''
     Obtiene los datos de un algoritmo específico en la colección especificada de MongoDB. Busca un registro con el nombre del algoritmo en mayúsculas y devuelve el resultado.
    '''
    def obtener_datos_algoritmo(self, coleccion, nombre_algoritmo, nombre_dataset):
        try:
            print(nombre_dataset)
            print(nombre_algoritmo)
            self.collection = self.db[coleccion]
            regex_pattern = f".*{re.escape(nombre_dataset)}.*"
            query = {"nombre_dataset": {"$regex": regex_pattern, "$options": "i"},"nombre_algoritmo": nombre_algoritmo.upper()}
            result = self.collection.find_one(query)
            if result:
                print(result)
                return result
            return None
        except Exception as e:
            print(f'Error al obtener el ultimo registro: {str(e)}')
            return None
    '''
    Obtiene la fecha más reciente de los registros en la colección especificada de MongoDB. Ordena los registros por fecha de forma descendente y devuelve la fecha del primer registro.
    '''   
    def obtener_fecha_mas_reciente(self, coleccion, nombre):
        try:
            self.collection = self.db[coleccion]
            regex_pattern = f".*{re.escape(nombre)}.*"
            query = {"nombre_dataset": {"$regex": regex_pattern}}
            sort_query = [('fecha', -1)]
            result = self.collection.find_one(query, sort=sort_query)
            # print(f'Fecha más reciente: {result}')
            if result:
                fecha_mas_reciente = result['fecha']
                return fecha_mas_reciente
            return ""
        except Exception as e:
            print(f'Error al obtener la fecha más reciente: {str(e)}')
            return None

    '''
    Obtiene los registros de métricas más recientes en la colección especificada de MongoDB. Utiliza la función obtener_fecha_mas_reciente 
    para obtener la fecha más reciente y luego busca los registros que coinciden con esa fecha. Devuelve una lista de registros.
    '''
    def obtener_registros_metricas_recientes(self, coleccion, nombre_dataset):
        try:
            fecha_mas_reciente = self.obtener_fecha_mas_reciente(coleccion, nombre_dataset)
            print(f'Fecha más reciente: {fecha_mas_reciente}')
            if fecha_mas_reciente:
                self.collection = self.db[coleccion]
                regex_pattern = f".*{re.escape(nombre_dataset)}.*"
                query = {"nombre_dataset": {"$regex": regex_pattern}, "fecha": fecha_mas_reciente}
                result = self.collection.find(query)
                registros = [registro for registro in result]
                return registros
            else:
                return []
        except Exception as e:
            print(f'Error al obtener los registros de la fecha más reciente: {str(e)}')
            return None
    '''
     Obtiene los registros de métricas más recientes en la colección especificada de MongoDB. Utiliza la función obtener_registros_metricas_recientes para obtener los registros y los devuelve
    '''
    def obtener_mejores_algoritmos(self, coleccion, nombre_dataset):
        try:
            registros= self.obtener_registros_metricas_recientes(coleccion, nombre_dataset)
            #print(f'Registros: {registros}')
            return registros 
        except Exception as e:
            print(f'Error al obtener los mejores algoritmos: {str(e)}')
            return None
    

    def obtener_nombres_dataset(self, coleccion):
        try:
            nombres_dataset = []
            self.collection = self.db[coleccion]
            for documento in self.collection.find():
                nombre_dataset = documento.get("nombre_dataset")
                if nombre_dataset and nombre_dataset not in nombres_dataset:
                    nombres_dataset.append(nombre_dataset)
            return nombres_dataset
        except Exception as e:
            print("Error al obtener los nombres del dataset")
            return None