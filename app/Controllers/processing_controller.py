from app.services.processing_service import ProcessingService
from app.utils.utils import Utils
from app.services.file_service import FileService
from fastapi.responses import JSONResponse


'''
clase proporciona los metodos para interactuar con el servicio de procesamiento de datos 

'''
class ProcessingController:

    def __init__(self):
        self.processing_service = ProcessingService()
        self.utils = Utils()
        self.file_service = FileService()

 # Realiza la imputación de datos y devuelve el mensaje procesado con un código de estado
    def imputation_data(self, nombre_dataset):
        try: 
            msg= self.processing_service.imputacion_datos(nombre_dataset)
            if msg:
                return self.utils.prueba(msg=msg), 200
            return self.utils.prueba(msg='No'), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al procesar el archivo: {str(e)}'), 500
        
 # Obtiene los tipos de datos y devuelve los tipos de datos procesados con un código de estado
    def get_types(self, nombre_dataset):
        try: 
            datos = self.processing_service.obtener_tipo_datos(nombre_dataset=nombre_dataset)
            
            return (self.utils.prueba(msg=datos)), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los tipos de datos: {str(e)}'), 500
        
 # Realiza el descarte de datos y devuelve el mensaje procesado con un código de estado            
    def descarte(self, nombre_dataset):
        try: 
            msg = self.processing_service.descarte_datos(nombre_dataset)
            if msg:
                return (self.utils.prueba(msg=msg)), 200
            return (self.utils.prueba(msg='No hay tipos de datos')), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los tipos de datos: {str(e)}'), 500

 # Genera imágenes de análisis y devuelve un mensaje de éxito con las ubicaciones de las imágenes y un código de estado
    def generar_imagenes_analisis(self, nombre_dataset):
        try: 
            ubicaciones = self.processing_service.generar_img_analisis(nombre_dataset)
            return ubicaciones
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los tipos de datos: {str(e)}'), 500
        
# Obtiene una imagen de histograma  y devuelve la imagen obtenida.
    async def obtener_histograma(self, nombre_dataset):
            return await self.processing_service.obtener_imagen_histograma(nombre_dataset)

  # Obtiene una imagen de matriz de correlación y devuelve la imagen obtenida.       
    async def obtener_matriz_correlacion(self, nombre_dataset):
            return self.processing_service.obtener_imagen_matriz(nombre_dataset)
  # Obtiene la última matriz de confusión de un algoritmo utilizando el nombre del modelo y devuelve la matriz obtenida.
    async def obtener_matriz_confusion(self, nombre_modelo, nombre_dataset):
        return self.processing_service.obtener_ultima_matriz_confusion_algoritmo(nombre_modelo, nombre_dataset)

# Obtiene las metricas de un algoritmo entrenado
    def metricas_algoritmos_entrenados(self, nombre_dataset):

        return self.processing_service.obtener_metricas_algoritmos(nombre_dataset=nombre_dataset)

        