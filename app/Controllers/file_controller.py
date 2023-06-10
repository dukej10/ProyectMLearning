from app.services.file_service import FileService
from flask import jsonify, request
from datetime import datetime
from app.utils.utils import Utils

'''
clase para controlar conexiones de servicios para manejo de archivos 
'''

class FileController:
    
    def __init__(self):
       self.file_service = FileService()
       self.utils = Utils()

#metodo asincrono para cargar el archivo desde el explorador

    async def upload_file(self, file):
        if not file:
                return self.utils.prueba(msg='No se recibió ningún archivo.'), 400
        
        if file.filename == '':
                return self.utils.prueba(msg='No se seleccionó ningún archivo.'), 400

        try:
                if await self.file_service.guardar_archivo(file, "original",'./app/files/sets/'):
                    return self.utils.prueba(msg='Archivo guardado correctamente'), 201
        except Exception as e:
                return self.utils.prueba(msg=f'Error al guardar el archivo: {str(e)}'), 500
            
#metodo para devolver el nombre de los archivos 
    def get_all_names_files(self):
        try:
            return (self.utils.prueba(msg='Lista de nombres de archivos', datos= self.file_service.obtener_nombres_archivos())), 200
        except Exception as e:
            return self.utils.prueba(msg=f'Error al obtener los nombres de los archivos: {str(e)}'), 500
