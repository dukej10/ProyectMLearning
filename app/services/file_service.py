from datetime import datetime
import os
import glob
import pandas as pd
from fastapi.responses import FileResponse

from app.services.mongodb_service import MongoDBService
from app.utils.utils import Utils

class FileService:


    def __init__(self):
        self.mongo_service = MongoDBService()
        self.utils = Utils()

    def welcome(self):
        return 'Bienvenido a la API de ML.'

    async def guardar_archivo(self, file, version, ubicacion):
        try:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + file.filename
            os.makedirs(ubicacion, exist_ok=True)  # Crear la carpeta si no existe
            directory = os.path.join(ubicacion, filename)
            # print(directory)
            with open(directory, 'wb') as f:
                f.write(await file.read())
            self.guardar_json(version, filename, 'sets')
            return 'Archivo guardado correctamente.'
        except FileNotFoundError as e:
            return f'Error al guardar el archivo: {str(e)}'
        
    def obtener_nombres_archivos(self):
        ruta_carpeta = os.path.join('app', 'files')
        nombres_archivos = []
        
        try:
            for nombre_archivo in os.listdir(ruta_carpeta):
                if os.path.isfile(os.path.join(ruta_carpeta, nombre_archivo)):
                    nombres_archivos.append(nombre_archivo)
            
            return nombres_archivos
        except Exception as e:
            print(f'Error al obtener los nombres de los archivos: {str(e)}')
            return []
        
    def obtener_ultimo_archivo(self, ubicacion):
        try: 
            ruta_carpeta = os.path.join('app','files', ubicacion)
            if 'imgs' in ubicacion:
                archivo = glob.glob(os.path.join(ruta_carpeta, '*.png')) + glob.glob(os.path.join(ruta_carpeta, '*.jpg'))
            else:
                archivo = glob.glob(os.path.join(ruta_carpeta, '*.xlsx')) + glob.glob(os.path.join(ruta_carpeta, '*.csv'))
            if archivo:
                archivo_mas_reciente = max(archivo, key=os.path.getctime)
                print(archivo_mas_reciente)
                archivo_mas_reciente = archivo_mas_reciente.replace('/', '\\')
                return archivo_mas_reciente
            else:
                return None
        except FileNotFoundError as e:
            print(f'Error al obtener el ultimo archivo: {str(e)}')
            return None

    def convertir_archivo_a_json(self, version, filename, ubicacion):
        try:
            ultimo_archivo = self.obtener_ultimo_archivo(ubicacion)
            if ultimo_archivo:
                ruta_archivo = os.path.join(ultimo_archivo)
                # print(ultimo_archivo)
                # print(ruta_archivo)
                if ultimo_archivo.endswith('.xlsx'):
                    df = pd.read_excel(ultimo_archivo, engine='openpyxl')
                elif ultimo_archivo.endswith('.csv'):
                    df = pd.read_csv(ultimo_archivo, sep=';')
                titulos = list(df.columns)
                valores = df.to_dict(orient='records')
                # print(titulos)
                # print(valores)
                datos_json = {'nombreDoc':filename,'version': version ,'titulos': titulos, 'valores': valores}
                return datos_json
            else:
                print('No se encontrón archivos Excel')
                return None
        except Exception as e:
            print(f'Error al convertir el archivo Excel a JSON: {str(e)}')
            return None
        
    def guardar_json(self, version, filename, ubicacion):
        try:
            json = self.convertir_archivo_a_json(version, filename, ubicacion)
            if json:
                self.mongo_service.guardar_json(json )
                # print("ME CONECTÉ")
                print('Archivo JSON guardado correctamente.')
            else: 
                print("NO HAY JSON")
                return None
            
        except Exception as e:
            print(f'Error al guardar el archivo JSON: {str(e)}')

    

    def _copia_excel(self, df, nombre_archivo, accion):
        try:
            ruta_archivo = './app/files/cleanData/'+accion+os.path.basename(nombre_archivo)
            df.to_excel(ruta_archivo, index=False)
            return "Se ha guardado una copia del DataFrame modificado en un archivo Excel, en " + ruta_archivo
        except Exception as e:
            return f'Error al guardar el documento: {str(e)}'

    