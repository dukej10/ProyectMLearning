from app.services.processing_service import ProcessingService
from app.utils.utils import Utils
from app.services.file_service import FileService
from fastapi.responses import JSONResponse



class ProcessingController:

    def __init__(self):
        self.processing_service = ProcessingService()
        self.utils = Utils()
        self.file_service = FileService()

    def imputation_data(self):
        try: 
            msg= self.processing_service.imputacion_datos()
            if msg:
                return self.utils.prueba(msg=msg), 200
            return self.utils.prueba(msg='No'), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al procesar el archivo: {str(e)}'), 500

    def get_types(self):
        try: 
            datos = self.processing_service.obtener_tipo_datos()
            
            return (self.utils.prueba(msg=datos)), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los tipos de datos: {str(e)}'), 500
            
    def descarte(self):
        try: 
            msg = self.processing_service.descarte_datos()
            if msg:
                return (self.utils.prueba(msg=msg)), 200
            return (self.utils.prueba(msg='No hay tipos de datos')), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los tipos de datos: {str(e)}'), 500

    def generar_imagenes_analisis(self):
        try: 
            ubicaciones = self.processing_service.generar_img_analisis()
            if ubicaciones:
                return  (self.utils.prueba(msg='Se generaron las imagenes de análisis', datos= ubicaciones)), 200
            return (self.utils.prueba(msg='No hay tipos de datos')), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los tipos de datos: {str(e)}'), 500
        
    async def obtener_histograma(self):
            return await self.processing_service.obtener_imagen_histograma()

        
    async def obtener_matriz_correlacion(self):
            return self.processing_service.obtener_imagen_matriz()
    
    async def obtener_matriz_confusion(self, nombre_modelo):
        return self.processing_service.obtener_ultima_matriz_confusion_algoritmo(nombre_modelo)
    
    def metricas_algoritmos_entrenados(self):

        return self.processing_service.obtener_metricas_algoritmos()

        