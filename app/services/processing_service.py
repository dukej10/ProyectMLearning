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
            datos = self.mongo_service.obtener_ultimo_registro("Dataset") 
            ruta_archivo = self.file_service.obtener_ultimo_archivo("sets")
            # Lee el archivo Excel o utiliza tu DataFrame existente
            nombre_archivo = os.path.basename(ruta_archivo)
            # print(ruta_archivo)
            if datos:
               titulos = datos["titulos"]
               valores = datos['valores']
               df = pd.DataFrame(valores, columns=titulos)

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
                    self.file_service.guardar_archivo_json("descarte", nombre_archivo, 'cleanData', 'Dataset')
                    return self.utils.prueba( msg=msg)
               else:
                    print("No hay columnas con valores nulos.")
                    return "No hay columnas con valores nulos."
            else:
                return "No hay datos"
        except Exception as e:
            print(f'Error al tratar los datos: {str(e)}')
            return None
       
    def imputacion_datos(self):
        try:
            ruta_archivo = self.file_service.obtener_ultimo_archivo("sets")
            datos = self.mongo_service.obtener_ultimo_registro("Dataset") 
            # Lee el archivo Excel o utiliza tu DataFrame existente
            nombre_archivo = os.path.basename(ruta_archivo)
            if datos:
                titulos = datos["titulos"]
                valores = datos['valores']
                df = pd.DataFrame(valores, columns=titulos)
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
                    self.file_service.guardar_archivo_json("imputacion", nombre_archivo, 'cleanData','Dataset')
                    # Guarda una copia del DataFrame modificado en un archivo Excel
                    return self.file_service._copia_excel(df, ruta_archivo, "imputacion-")
            else:
                return "No hay datos"
        except Exception as e:
            print(f'Error al tratar los datos: {str(e)}')
            return None
        
    def obtener_tipo_datos(self):
        
        try:
            datos = self.mongo_service.obtener_ultimo_registro()    
            if datos:
                titulos = datos["titulos"]
                valores = datos['valores']
                df = pd.DataFrame(valores, columns=titulos)
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
                return columnas_clasificadas
            else:
                return "No se encuentra un registro en la base de datos."
        except Exception as e:
            print(f'Error al obtener el registro: {str(e)}')
            return None

    def generar_img_analisis(self):
        try:
            # Lee el archivo Excel o utiliza tu DataFrame existente
            datos = self.mongo_service.obtener_ultimo_registro()    
            # print(dataframe)
            if datos:
                titulos = datos["titulos"]
                valores = datos['valores']
                df = pd.DataFrame(valores, columns=titulos)

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
                return 'No se encontró ningún registro.'
        except Exception as e:
            return f'Error buscar el registro: {str(e)}'


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
            # print("IMAGEN")
            ubicacion_img = self.file_service.obtener_ultimo_archivo(ubicacion)
            if ubicacion_img:
                path_img = os.path.join(ubicacion_img)
                path_img = os.path.abspath(path_img)
                
                if os.path.exists(path_img):
                    return path_img
            
                else:
                    return 'La imagen no existe.', 404
            else:
                return None
        except FileNotFoundError as e:
            return f'No se encontró la imagen {str(e)}', 404
    
    def obtener_imagen_matriz(self):
        try:
            img = self._obtener_imagen("imgs/matriz_correlacion")
            # print("TENGO IMAGEN" + img)
            if img:
                return img
            else:
                return {'mensaje':f'No se encontrón las imagenes.'}, 404
        except FileNotFoundError as e:
            return {'mensaje':f'Error al obtener la imagen de la matriz de correlación: {str(e)}'}, 404
        
    async def obtener_imagen_histograma(self):
        try:
            # print("HISTOGRAMA service")
            img = self._obtener_imagen("imgs/histogramas")
            # print(f'IMAGEN: {img}')
            return img
        except FileNotFoundError as e:
            return {'mensaje':f'Error al obtener la imagen del histograma: {str(e)}'}, 404