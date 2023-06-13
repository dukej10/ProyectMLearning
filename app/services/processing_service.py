import glob
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

'''
clase que define los servicios de las funcionalidades generales que usa la api para entrenar y predecir datos
'''
class ProcessingService:

    def __init__(self):
        self.mongo_service = MongoDBService()
        self.file_service = FileService()
        self.utils = Utils()
    '''
    Este método obtiene el último registro de un conjunto de datos desde MongoDB y el último archivo de la carpeta "sets". 
    Luego, crea un DataFrame a partir de los datos obtenidos y verifica si hay valores nulos en el DataFrame. 
    Si encuentra valores nulos, elimina las columnas correspondientes y guarda una copia del DataFrame modificado en un archivo Excel.
    '''
    def descarte_datos(self, nombre_dataset):
        try:
            if self.file_service.verificar_dataset(nombre_dataset) is False:
                return f"No existe el dataset {nombre_dataset}"
            datos = self.mongo_service.obtener_ultimo_registro("Dataset",nombre_dataset=nombre_dataset ) 
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
               #print(columnas_con_nulos)
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
                return f"No hay datos del dataset {nombre_dataset}"
        except Exception as e:
            print(f'Error al tratar los datos: {str(e)}')
            return None
    '''
     Este método obtiene el último archivo de la carpeta "sets" y el último registro de un conjunto de datos desde MongoDB. 
     A continuación, crea un DataFrame a partir de los datos obtenidos y realiza la imputación de valores nulos en las 
     columnas correspondientes. Si se realizan cambios debido a la imputación, guarda una copia del DataFrame modificado en un archivo Excel.
    '''  
    def imputacion_datos(self, nombre_dataset):
        try:
            if self.file_service.verificar_dataset(nombre_dataset) is False:
                return f"No existe el dataset {nombre_dataset}"
            ruta_archivo = self.file_service.obtener_ultimo_archivo("sets")
            datos = self.mongo_service.obtener_ultimo_registro_por_nombre("Dataset", nombre_dataset) 
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
                return f"No hay datos del dataset {nombre_dataset}"
        except Exception as e:
            print(f'Error al tratar los datos: {str(e)}')
            return None
    '''
    Este método obtiene el último registro de un conjunto de datos desde MongoDB y devuelve los títulos de las columnas disponibles.
    '''   
    def columnas_disponibles_dataset(self, nombre_dataset):
        try:
            datos = self.mongo_service.obtener_ultimo_registro_por_nombre("Dataset", nombre_dataset)
            if datos:
                #print(datos["titulos"])
                titulos = datos["titulos"]
                # print("OE " ,titulos)
                return titulos
        except Exception as e:
            print(f'Error al obtener los datos: {str(e)}')
            return None
    '''
    Este método obtiene el último registro de un conjunto de datos desde MongoDB y crea un DataFrame a partir de los datos obtenidos. 
    Luego, clasifica las columnas del DataFrame en tres categorías: numérico, texto y booleano, y devuelve un diccionario con las columnas clasificadas.
    '''
    def obtener_tipo_datos(self, nombre_dataset):
        
        try:
            if self.file_service.verificar_dataset(nombre_dataset) is False:
                return f"No existe el dataset {nombre_dataset}"
            datos = self.mongo_service.obtener_ultimo_registro_por_nombre('Dataset', nombre_dataset)    
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
    '''
    Este método obtiene el último registro de un conjunto de datos desde MongoDB y crea un DataFrame a partir de los datos obtenidos.
    Luego, genera un histograma y una matriz de correlación para las columnas numéricas del DataFrame.
    Guarda las imágenes generadas en las carpetas correspondientes y devuelve las ubicaciones de las imágenes.
    '''
    def generar_img_analisis(self, nombre_dataset):
        try:
            if self.file_service.verificar_dataset(nombre_dataset) is False:
                return f"No existe el dataset {nombre_dataset}"
            # Lee el archivo Excel o utiliza tu DataFrame existente
            datos = self.mongo_service.obtener_ultimo_registro_por_nombre('Dataset', nombre_dataset)    
            # print(dataframe)
            if datos:
                titulos = datos["titulos"]
                valores = datos['valores']
                df = pd.DataFrame(valores, columns=titulos)

                # Obtener las columnas numéricas del DataFrame
                numerico = df.select_dtypes(np.number)
                self._generate_histograma(numerico, nombre_dataset)
                
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

    '''
    Estos son métodos privados utilizados por el método "generar_img_analisis" para generar el histograma y la matriz de correlación, respectivamente.
    '''
    def _generate_histograma(self, columnas_numericas, nombre_dataset):
         # Configurar el tamaño y el estilo de la figura
                plt.rcParams['figure.figsize'] = (16, 9)
                plt.style.use('ggplot')
                filename = nombre_dataset+ '-' + 'histogramas.png'
                # Generar los histogramas
                columnas_numericas.hist()
                plt.savefig('app/files/imgs/histogramas/' + filename)
                self._generate_matriz_correlacion(columnas_numericas, plt, nombre_dataset)

    def _generate_matriz_correlacion(self, columnas_numericas, plt, nombre_dataset):
        colormap = plt.cm.coolwarm
        plt.figure(figsize=(12,12))
        plt.title(nombre_dataset, y=1.05, size=15)
        sb.heatmap(columnas_numericas.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
        filename = nombre_dataset + '-' + 'matriz_correlacion.png'
        plt.savefig('app/files/imgs/matriz_correlacion/' + filename)
        plt.close()
    '''
    Este es un método privado utilizado para obtener la ubicación de una imagen en función de la ubicación proporcionada.
    '''
    def _obtener_imagen(self, ubicacion):
        try: 
            # print("UBICACION  ", ubicacion)
            ubicacion_img = self.file_service.obtener_ultimo_archivo(ubicacion)
            print("UBICACION  ", ubicacion_img)
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
   
    '''
     Este método recibe el nombre de un algoritmo y busca la última matriz de confusión generada para ese algoritmo. Devuelve la ubicación de la última imagen generada.
    '''
    def obtener_ultima_matriz_confusion_algoritmo(self,nombre:str, nombre_dataset):
        try:
            if  self.utils.arreglar_nombre(nombre) == 'REGRESIONLOGISTICA':
                 nombre = 'reglog'
            elif self.utils.arreglar_nombre(nombre) == 'KNN':
                 nombre = 'knn'
            elif self.utils.arreglar_nombre(nombre) == 'NAIVEBAYES':
                 nombre = 'naive_bayes'
            elif self.utils.arreglar_nombre(nombre) == 'ARBOLDEDECISION':
                 nombre = 'arbol_decision'
            elif self.utils.arreglar_nombre(nombre) == 'SVM':
                 nombre = 'svm'
            else:
                return {'mensaje':f"No se encontr+o matriz de confusion para el algoritmo {nombre} del dataset {nombre_dataset}"}, 404
            # Obtener la lista de archivos que coinciden con el nombre proporcionado
            ruta = "app/files/imgs/modelos/matrices-confusion/"

            archivos = glob.glob(f"app/files/imgs/modelos/matrices-confusion/{nombre_dataset}*{nombre}*.png")
            aux = []
            #print("ARCHIVOS ", archivos)
            for archivo in archivos:
                archivo = archivo.replace('\\', '/')
                aux.append(archivo)
            archivos = aux
            #print("ARCHIVOS ", archivos)
            # Ordenar la lista de archivos por fecha de modificación descendente
            archivos.sort(key=os.path.getmtime, reverse=True)

            # Verificar si se encontraron archivos
            if archivos:
                print(archivo[0])
                # Devolver la ruta de la última imagen generada
                return archivos[0]
            else:
                return {'mensaje':f"No se encontr+o matriz de confusion para el algoritmo {nombre} del dataset {nombre_dataset}"}, 404
        except FileNotFoundError as e:
                return {'mensaje':f"Error al obtener la matriz de confusion del algoritmo {nombre}."}, 500
    
    '''
    Estos métodos obtienen la ubicación de la imagen de la matriz de correlación y el histograma, respectivamente.
    '''
    def obtener_imagen_matriz(self, nombre_dataset):
        try:
            nombre_dataset = nombre_dataset.lower()
            img = self._obtener_imagen(f"imgs/matriz_correlacion/{nombre_dataset}-matriz_correlacion")
            # print("TENGO IMAGEN" + img)
            if img:
                return img
            else:
                return {'mensaje':f'No se encontrón las imagenes.'}, 404
        except FileNotFoundError as e:
            return {'mensaje':f'Error al obtener la imagen de la matriz de correlación: {str(e)}'}, 404
        
    async def obtener_imagen_histograma(self, nombre_dataset):
        try:
            nombre_dataset = nombre_dataset.lower()
            img = self._obtener_imagen(f"imgs/histogramas/{nombre_dataset}-histogramas")
            # print(f'IMAGEN: {img}')
            return img
        except FileNotFoundError as e:
            return {'mensaje':f'Error al obtener la imagen del histograma: {str(e)}'}, 404
    
    '''
    Este método obtiene las métricas de los modelos de algoritmos más recientes almacenados en MongoDB. 
    Devuelve una lista de diccionarios que contienen el nombre del algoritmo, la normalización, la técnica utilizada y las métricas.
    '''
    def obtener_metricas_algoritmos(self, nombre_dataset):
            metricas =self.mongo_service.obtener_registros_metricas_recientes('InformacionModelos', nombre_dataset)
            datos = []
            #print(metricas)
            if metricas is not None:
                if len(metricas) > 0:
                    for  data in metricas:
                        datos.append({'nombre_algoritmo': data['nombre_algoritmo'], 'normalizacion': data['normalizacion'], 'tecnica': data['tecnica'], 'metricas': data['metricas']})
                        #print(datos)
                    return self.utils.prueba(msg="Se encontraron metricas", datos=datos)
                else:
                    return self.utils.prueba(msg="No se encontraron metricas"), 200
            else:
                return self.utils.prueba(msg="Error al obtener las metricas"), 404