import os
from flask import send_file
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from datetime import datetime
from app.services.file_service import FileService
from app.utils.utils import Utils
from fastapi.responses import FileResponse

from app.services.mongodb_service import MongoDBService


class ProcessingService:

    def __init__(self):
        self.mongo_service = MongoDBService()
        self.file_service = FileService()
        self.utils = Utils()

    def descarte_datos(self):
        try:
            ruta_archivo = self.file_service.obtener_ultimo_archivo("sets")
            # Lee el archivo Excel o utiliza tu DataFrame existente
            nombre_archivo = os.path.basename(ruta_archivo)
            print(ruta_archivo)
            if ruta_archivo:
                 # Lee el archivo Excel o utiliza tu DataFrame existente
                df = pd.read_excel(ruta_archivo)

                # Verifica si hay valores nulos en el DataFrame
                columnas_con_nulos = df.columns[df.isnull().any()].tolist()
                print(columnas_con_nulos)
                if len(columnas_con_nulos) > 0:
                    # Elimina las columnas con valores nulos
                    df = df.dropna(subset=columnas_con_nulos)
                    # Verifica nuevamente si hay valores nulos después del descarte
                    if df.isnull().values.any():
                        print("Aún hay valores nulos en el DataFrame después del descarte.")
                    else:
                        print("Se han descartado las columnas con valores nulos.")
                        # Guarda una copia del DataFrame modificado en un archivo Excel
                    msg= self.file_service._copia_excel(df, ruta_archivo, "descarte-")
                    self.file_service.guardar_json("descarte", nombre_archivo, 'cleanData')
                    return self.utils.prueba( msg=msg)
                else:
                    print("No hay columnas con valores nulos.")
                    return "No hay columnas con valores nulos."
        except Exception as e:
            print(f'Error al tratar los datos: {str(e)}')
            return None
       
    def imputacion_datos(self):
        try:
            ruta_archivo = self.file_service.obtener_ultimo_archivo("sets")
            # Lee el archivo Excel o utiliza tu DataFrame existente
            nombre_archivo = os.path.basename(ruta_archivo)
            if ruta_archivo:
                df = pd.read_excel(ruta_archivo)
                # Encuentra las columnas con valores nulos
                columnas_con_nulos = df.columns[df.isnull().any()].tolist()
    
                # Itera sobre las columnas con valores nulos
                for columna in columnas_con_nulos:
                    if df[columna].dtype == 'object':
                        # Imputación por moda en columnas categóricas (texto)
                        moda_columna = df[columna].mode()[0]
                        df[columna].fillna(moda_columna, inplace=True)
                    else:
                        # Imputación por media en columnas numéricas
                        media_columna = df[columna].mean()
                        df[columna].fillna(media_columna, inplace=True)
                # Verifica si se han realizado cambios debido a la imputación de valores nulos
                if df.isnull().values.any():
                    return "Aún hay valores nulos en el DataFrame."
                else:
                    print("Se han imputado los valores nulos con la media.")
                    self.file_service.guardar_json("imputacion", nombre_archivo, 'cleanData')
                    # Guarda una copia del DataFrame modificado en un archivo Excel
                    return self.file_service._copia_excel(df, ruta_archivo, "imputacion-")
            else:
                return "No hay archivo"
        except Exception as e:
            print(f'Error al tratar los datos: {str(e)}')
            return None
        
    def obtener_tipo_datos(self):
        
        try:
            ruta_archivo = self.file_service.obtener_ultimo_archivo('sets')
            if ruta_archivo:
                if ruta_archivo.endswith('.xlsx'):
                    df = pd.read_excel(ruta_archivo, engine='openpyxl')
                elif ruta_archivo.endswith('.csv'):
                    df = pd.read_csv(ruta_archivo, sep=';')
                tipos_datos = df.dtypes
                columnas_clasificadas = {
                    "numerico": [],
                    "texto": [],
                    "booleano": []
                }


                for columna, tipo in tipos_datos.items():
                    if tipo == 'int64' or tipo == 'float64':
                        columnas_clasificadas["numerico"].append(columna)
                    elif tipo == 'object':
                        columnas_clasificadas["texto"].append(columna)
                    elif tipo == 'bool':
                        columnas_clasificadas["booleano"].append(columna)
                print(columnas_clasificadas)
                return columnas_clasificadas
            else:
                print("NADA")
                return None
        except Exception as e:
            print(f'Error al obtener los tipos de datos: {str(e)}')
            return None

    def generar_img_analisis(self):
        try:
            ruta_archivo = self.file_service.obtener_ultimo_archivo("cleanData")
            # Lee el archivo Excel o utiliza tu DataFrame existente
               
            if ruta_archivo:
                df = pd.read_excel(ruta_archivo)

                # Obtener las columnas numéricas del DataFrame
                numerico = df.select_dtypes(np.number)
                self._generate_histograma(numerico)
                
                ubicacion_histograma = self.file_service.obtener_ultimo_archivo("imgs/histogramas")
                ubicacion_matriz = self.file_service.obtener_ultimo_archivo("imgs/matriz_correlacion")
                ubicacion_histograma = ubicacion_histograma.replace("/", "\\")
                ubicacion_matriz = ubicacion_matriz.replace("/", "\\")
                retorno = f'Ubicación del histograma: {ubicacion_histograma} Ubicación de la matriz de correlación: {ubicacion_matriz}'
                
                return retorno
            else:
                return 'No se encontró ningún archivo Excel.'
        except Exception as e:
            return f'Error ubicar el archivo: {str(e)}'


    def _generate_histograma(self, columnas_numericas):
         # Configurar el tamaño y el estilo de la figura
                plt.rcParams['figure.figsize'] = (16, 9)
                plt.style.use('ggplot')
                filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + 'histogramas.png'
                # Generar los histogramas
                columnas_numericas.hist()
                plt.savefig('app/files/imgs/histogramas/' + filename)
                self._generate_matriz_correlacion(columnas_numericas, plt)

    def _generate_matriz_correlacion(self, columnas_numericas, plt):
        colormap = plt.cm.coolwarm
        plt.figure(figsize=(12,12))
        plt.title('Bank', y=1.05, size=15)
        sb.heatmap(columnas_numericas.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + 'matriz_correlacion.png'
        plt.savefig('app/files/imgs/matriz_correlacion/' + filename)
        plt.close()

    def _obtener_imagen(self, ubicacion):
        try: 
            ubicacion_img = self.file_service.obtener_ultimo_archivo(ubicacion)
            if ubicacion_img:
                path_img = os.path.join(ubicacion_img)
                path_img = os.path.abspath(path_img)
                
                if os.path.exists(path_img):
                    return send_file(path_img, mimetype='image/png'), 200
            
                else:
                    return 'La imagen no existe.', 404
            else:
                return None
        except FileNotFoundError as e:
            return f'No se encontró la imagen {str(e)}', 404
    
    def obtener_imagen_matriz(self):
        try:
            img = self._obtener_imagen("imgs/matriz_correlacion")
            if img:
                return img
            else:
                return {'mensaje':'No se encontré la imagen de la matriz de correlación.'}, 404
        except FileNotFoundError as e:
            return {'mensaje':f'Error al obtener la imagen de la matriz de correlación: {str(e)}'}, 404
        
    def obtener_imagen_histograma(self):
        try:
            img = self._obtener_imagen("imgs/histogramas")
            imagen_path = "files/img/histogramas/2023-05-22_19-28-49_histogramas.png"
            if img:
                return FileResponse(imagen_path)
            else:
                return {'mensaje':'No se encontró la imagen del histograma.'}, 404
        except FileNotFoundError as e:
            return {'mensaje':f'Error al obtener la imagen del histograma: {str(e)}'}, 404