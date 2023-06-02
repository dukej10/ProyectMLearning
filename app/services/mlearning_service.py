import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from app.services.mongodb_service import MongoDBService
from app.utils.utils import Utils
from app.services.dataframe_service import DataframeService
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#Importación knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
import seaborn as sb


#Normalizar datos
from sklearn.preprocessing import MinMaxScaler


class MLearningService:

    def __init__(self):
        self.mongo_service = MongoDBService()
        self.utils = Utils()
        self.XTrainKNN = None
        self.yTrainKNN = None
        self.XTestKNN = None
        self.yTestKNN = None
        self.x = None
        self.y = None
        self.modeloKNN = None
        self.NameY = None
        self.yPredictKNN = None
        self.dataframe_service = DataframeService()

    def knn(self):
         dataframe = self.preparacion_dataframe()
         self.particion_dataset(dataframe)
         self.modeloKNN = KNeighborsClassifier(n_neighbors=self.mejor_k())
         self.modeloKNN.fit(self.XTrainKNN,self.yTrainKNN)
         self.yPredictKNN = self.modeloKNN.predict(self.XTestKNN)
         self.identificar_overffing_underffing()
         if self.obtener_matriz_confusion() is None:
              return "error"
         return "ok"
         
    def preparacion_dataframe(self):
        try:
            datos = self.mongo_service.obtener_ultimo_registro()
            #print(datos)
            if datos:
                titulos = datos["titulos"]
                self.NameY = titulos[-1]
                
                print("TITULOS ", self.NameY)
                valores = datos['valores']
                df = pd.DataFrame(valores, columns=titulos)
                #print(df)
                print("-------------------------------------------")
                #print("CONCATENAR")
                df = self.dataframe_service.codificar_valores_cat(df)
                print("NORMALIZAR")
                dataNumerica = self.dataframe_service.normalizar_informacion(df)
                #print("REDONDEAR")
                dataNumerica = self.dataframe_service.redondear_datos(dataNumerica)
                #print("CONCATENAR")
                df = self.dataframe_service.concatenar_datos(dataNumerica, df)
                #print("AQUI ESTA EL DATAFRAME")
                #print(df)
                return df
        except:
            return  "No hay datos"

    def particion_dataset(self, dataframe):
        self.x=dataframe.drop([self.NameY],axis=1) # obtener valores de x
        self.y=dataframe[self.NameY] 
        self.XTrainKNN,self.XTestKNN,self.yTrainKNN,self.yTestKNN=train_test_split(self.x,self.y,test_size=0.2, random_state=2)

    def mejor_k(self):
         # Posibles valores que puede tomar
        kvalores = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17]}

        # obtener el mejor valor por validación cruzada
        grid = GridSearchCV(KNeighborsClassifier(), kvalores)

        # Ajustar a los datos de entrenamiento
        grid.fit(self.XTrainKNN, self.yTrainKNN)

        # valor óptimo de k
        print("Mejor valor de k: ", grid.best_params_['n_neighbors'])
        return grid.best_params_['n_neighbors']
    
    def obtener_matriz_confusion(self):
        try:
            print("Matriz de confusion")
            matrizKNN = confusion_matrix(self.yTestKNN, self.yPredictKNN)
            print(matrizKNN)
            sb.heatmap(matrizKNN, annot=True, cmap="Blues")
            ruta_guardado = "app/files/imgs/modelos/matrices_correlacion/"
            os.makedirs(ruta_guardado, exist_ok=True)
            plt.savefig(os.path.join(ruta_guardado, "knn-matriz_confusion.png"))
            plt.close()
        except Exception as e:
            print("Ow")

    def identificar_overffing_underffing(self):
         scores = cross_val_score(self.modeloKNN, self.x, self.y, cv=5)

         # calcular la media y la desviación estándar de las puntuaciones de precisión
         meanScore = np.mean(scores)
         stdScore = np.std(scores)

         # hacer predicciones en los datos de entrenamiento y prueba
         yPredTrain = self.modeloKNN.predict(self.XTrainKNN)
         yPredTest = self.modeloKNN.predict(self.XTestKNN)

         # calcular la precisión en los datos de entrenamiento y prueba
         accuracyTrain = accuracy_score(self.yTrainKNN, yPredTrain)
         accuracyTest = accuracy_score(self.yTestKNN, yPredTest)

         print("Precisión en los datos de entrenamiento:", accuracyTrain)
         print("Precisión en los datos de prueba:", accuracyTest)

         # imprimir las puntuaciones de precisión y sus estadísticas
         print('Puntuaciones de precisión:', scores.mean())
         print('Precisión media:', meanScore)
         print('Desviación estándar de la precisión:', stdScore)

    
        
    
                