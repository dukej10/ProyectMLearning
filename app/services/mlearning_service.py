import numpy as np
import pandas as pd
from app.services.mongodb_service import MongoDBService
from app.utils.utils import Utils
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Normalizar datos
from sklearn.preprocessing import MinMaxScaler


class MLearningService:

    def __init__(self):
        self.mongo_service = MongoDBService()
        self.utils = Utils()

    def knn(self):
        try:
            datos = self.mongo_service.obtener_ultimo_registro()
            print(datos)
            if datos:
                titulos = datos["titulos"]
                valores = datos['valores']
                df = pd.DataFrame(valores, columns=titulos)
                #print(df)
                print("-------------------------------------------")
                #print("CONCATENAR")
                df = self.codificar_valores_cat(df)
                print("NORMALIZAR")
                dataNumerica = self.normalizar_informacion(df)
                print("REDONDEAR")
                dataNumerica = self.redondear_datos(dataNumerica)
                print("CONCATENAR")
                df = self.concatenar_datos(dataNumerica, df)
                print("AQUI ESTA EL DATAFRAME")
                print(df)
        except:
            return  "No hay datos"
        
    def codificar_valores_cat(self, dataframe):
        #print(dataframe)
        encoder=LabelEncoder()
        cat_colsAll = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        # Obtener columnas a encodificar
        for col in cat_colsAll:
                dataframe[col] = encoder.fit_transform(dataframe[col])
        #print(dataframe)

        return dataframe
    
    def  normalizar_informacion(self, dataframe):
         #print("dataframe")
         dataNumerica = dataframe.select_dtypes(np.number)
         escalador=StandardScaler()
         dataNumerica=pd.DataFrame(escalador.fit_transform(dataNumerica), columns = dataNumerica.columns)
         #print("Normalizado")
         #print(dataNumerica)
         return dataNumerica
    
    def redondear_datos(self, dataNumerica):
         #print(dataNumerica)
         dataNumerica = dataNumerica.astype(int)
         #print("Redondeado")
         #print(dataNumerica)
         return dataNumerica
    
    def concatenar_datos(self, dataNumerica, dataframe):
        dataCategorica = dataframe.select_dtypes(object)
        dataFinal = pd.concat([dataNumerica, dataCategorica], axis=1)
        dataFinal.head()
        return dataFinal