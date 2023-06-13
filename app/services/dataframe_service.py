import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from app.models.entrenamiento_model import InfoEntrenamiento

from app.services.mongodb_service import MongoDBService

'''
clase que aporta la funcionalidad para normalizar y codificar los datos del Dataframe
'''

class DataframeService:

    def __init__(self):
         self.mongo_service = MongoDBService()

    '''
    metodo que Codifica los valores categóricos en el dataframe utilizando LabelEncoder. Almacena los valores originales y codificados en listas separadas 
    (listY y listX). Luego, guarda esta información en un objeto JSON utilizando el servicio de MongoDB si listY contiene elementos.
    '''  

    def codificar_valores_cat(self, dataframe, entrenamiento: InfoEntrenamiento, nombre_dataset):
        encoder=LabelEncoder()
        #print(dataframe.head())
        # Obtener todas las columnas de tipo object
        #print("TIPO ", dataframe.dtypes)
        cat_colsAll = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        colsAll = [col for col in dataframe.columns]
        # Obtener columnas a encodificar
        listY = []
        listX = []
        aux = []
        auxX = []
        auxX2 = []
        valores_originales = dataframe[colsAll].copy()
        #print("COLUMNAS ", cat_colsAll)
        for col in cat_colsAll:
                # print("COL ", col)
                # if col in cat_colsAll:
                dataframe[col] = encoder.fit_transform(dataframe[col])
                # print("CODIFICADO ", col)
                # Obtener los valores originales correspondientes a cada valor codificado
                valores_originales[col] = encoder.inverse_transform(dataframe[col])
                # else:
                #     valores_originales[col] = dataframe[col]
                
                
        print("COLUMNAS ", cat_colsAll)
# # Obtener los valor originales y codificado de cada columna elemento de las columnas
        for col in cat_colsAll:
                #print("COL ", col)
                for valor_codificado, valor_original in zip(dataframe[col], valores_originales[col]):
                    if col == entrenamiento.objetivo_y:
                        if valor_codificado not in aux:
                            listY.append({ "valor_codificado": valor_codificado, "valor_original": valor_original})
                            aux.append(valor_codificado)
                        else: 
                            continue
                    else:
                        if valor_codificado not in auxX2:
                            auxX.append({ "valor_codificado": valor_codificado, "valor_original": valor_original})
                            auxX2.append(valor_codificado)
                        else:
                            continue
                if col != entrenamiento.objetivo_y:
                    listX.append({col:auxX})
                auxX = []
                auxX2 = []

        if len(listY) > 0:
             print("guardar json")
             id = self.mongo_service.guardar_json({"nombre_dataset": nombre_dataset,"nombre_algoritmo":entrenamiento.nombre_algoritmo, "datosY":listY, "datosX":listX,'x':entrenamiento.columnas_x, 'y':entrenamiento.objetivo_y  }, "RepresentacionCodificacion")
        return dataframe
    
    '''
    Selecciona las columnas numéricas del dataframe y aplica una técnica de normalización según el tipo especificado 
      Devuelve el dataframe con los datos numéricos normalizados.
    '''

    def  normalizar_informacion(self, dataframe, tipo, objetivo_y):
         #print("dataframe")
         dataNumerica = dataframe.select_dtypes(np.number)
     #     columnas_no_y = dataNumerica.columns[dataNumerica.columns != objetivo_y]
     #     dataNumerica = dataNumerica[columnas_no_y]
         if tipo == "min-max":
          escalador=MinMaxScaler()
          dataNumerica=pd.DataFrame(escalador.fit_transform(dataNumerica), columns = dataNumerica.columns)
          print("min-max")
          #print(dataNumerica)
          return dataNumerica
         elif tipo == "standarscaler":
          escalador=StandardScaler()
          dataNumerica=pd.DataFrame(escalador.fit_transform(dataNumerica), columns = dataNumerica.columns)
          print("standarscaler")
          #print(dataNumerica)
          return dataNumerica
         
    '''
    Redondea los valores numéricos en el dataframe a enteros y devuelve el dataframe resultante.
    '''
    def redondear_datos(self, dataNumerica):
         #print(dataNumerica)
         dataNumerica = dataNumerica.astype(int)
         print("Redondeado")
         #print(dataNumerica)
         return dataNumerica
    
    '''
    Concatena los datos numéricos (almacenados en dataNumerica) con la columna objetivo objetivo_y del dataframe original (dataframe).
      Devuelve el dataframe final que contiene tanto los datos numéricos como la columna objetivo.
    '''
    def concatenar_datos(self, dataNumerica, dataframe, objetivo_y):
        print("dATA")
        #print(dataNumerica.info())
        
        dataFinal = pd.concat([dataNumerica, dataframe[objetivo_y]], axis=1)
        #print("DATAFINAL")
        #print(dataFinal.info())
        #print("--------------------")
        return dataFinal
    