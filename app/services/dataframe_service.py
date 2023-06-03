import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler



class DataframeService:

    def __init__(self):
         pass

    def codificar_valores_cat(self, dataframe):
        #print(dataframe)
        encoder=LabelEncoder()
        cat_colsAll = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        # Obtener columnas a encodificar
        for col in cat_colsAll:
                dataframe[col] = encoder.fit_transform(dataframe[col])
        #print(dataframe)

        return dataframe
    
    def  normalizar_informacion(self, dataframe, tipo):
         #print("dataframe")
         dataNumerica = dataframe.select_dtypes(np.number)
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
         
    
    def redondear_datos(self, dataNumerica):
         #print(dataNumerica)
         dataNumerica = dataNumerica.astype(int)
         print("Redondeado")
         #print(dataNumerica)
         return dataNumerica
    
    def concatenar_datos(self, dataNumerica, dataframe):
        dataCategorica = dataframe.select_dtypes(object)
        dataFinal = pd.concat([dataNumerica, dataCategorica], axis=1)
        return dataFinal
    