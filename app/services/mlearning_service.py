import datetime
import glob
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.linear_model import LogisticRegression
from app.models.prediccion_model import PrediccionModel
from app.services.file_service import FileService
from app.services.mongodb_service import MongoDBService
from app.services.processing_service import ProcessingService
from app.utils.utils import Utils
from app.services.dataframe_service import DataframeService
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
#Importación knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score, make_scorer
import seaborn as sb
# PARA REVISAR DISTRIBUCIÓN DE LA NORMALIDAD
from scipy.stats import normaltest

#Arboles
from sklearn.tree import DecisionTreeClassifier

#Regresion Lineal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


#Normalizar datos
from sklearn.preprocessing import MinMaxScaler
from app.models.entrenamiento_model import InfoEntrenamiento

'''
CLASE QUE REUNE LA FUNCIONALIDAD DE MLEARNING
Autores:
        - JUAN DIEGO DUQUE LOPEZ
        - DANIEL ANDRES CUARTAS ARANGO
        - ALEJANDRO TRUJILLO 

    Versión:
        1.0.0
'''

class MLearningService:

    def __init__(self):
        self.mongo_service = MongoDBService()
        self.utils = Utils()
        self.XTrain = None
        self.yTrain = None
        self.XTest = None
        self.yTest = None
        self.x = None
        self.y = None
        self.modeloKNN = None
        self.modeloRegLog = None
        self.yPredict = None
        self.dataframe_service = DataframeService()
        self.mongo_service = MongoDBService()
        self.infoEntrenamiento = None
        self.processing_service = ProcessingService()
        self.disponibles = None
        self.n_dataset = ""
        self.file_service = FileService()


    '''
    Este método realiza varias validaciones en los datos de entrenamiento proporcionados.
    Verifica la existencia de columnas y valida diferentes parámetros relacionados con la técnica de entrenamiento y
    la normalización de los datos. Retorna mensajes de error específicos si alguna validación falla.
    '''
    def validaciones(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        if self.file_service.verificar_dataset(nombre_dataset) is False:
            return f"No existe el dataset {nombre_dataset}"
        validar_columnas = self.validar_columnas(entrenamiento.columnas_x, nombre_dataset)
        if validar_columnas:
            validacion2 = self.validar_columnas(entrenamiento.objetivo_y, nombre_dataset)
            if validacion2:
                if entrenamiento.tecnica == 'hold-out' or entrenamiento.tecnica == "cross-validation":
                    if entrenamiento.normalizacion == 'min-max' or entrenamiento.normalizacion == 'standardscaler':
                        if ((entrenamiento.cantidad >= 0 and entrenamiento.cantidad <= 100) and entrenamiento.tecnica == 'hold-out'):
                            return True
                        elif (entrenamiento.cantidad > 0 and entrenamiento.tecnica == 'cross-validation'):
                            return True
                        elif (entrenamiento.cantidad <= 0 and entrenamiento.tecnica == 'cross-validation'):
                            return "La cantidad de particiones debe ser mayor a 0"
                        elif (entrenamiento.cantidad > 100 and entrenamiento.tecnica == 'hold-out'):
                            return "La cantidad de particiones debe ser menor a 100"
                        elif (entrenamiento.cantidad < 0 and entrenamiento.tecnica == 'hold-out'):
                            return "La cantidad de particiones debe ser mayor a 0"
                        else: 
                            return f'La técnica seleccionada es {entrenamiento.tecnica} y la cantidad de particiones es {entrenamiento.cantidad}, no es posible realizar la validación'
                    else: 
                        return "Se indicó una técnica de normalización no disponible, las opciones son: min-max o standardscaler"
                else:
                    return "Se indicó una técnica de validación no disponible, las opciones son: hold_out o cross-validation"
            elif validacion2 is False:
                return f"Se indicó columna objetivo que no existen en el dataset, las columnas disponibles son: {self.disponibles}"
            else:
                return "Error en la validación de columnas"
        elif validar_columnas is False:
            return f"Se indicaron columnas que no existen en el dataset, las columnas disponibles son: {self.disponibles}"
        else:
            return "Error en la validación de columnas"
    
    '''
    Este método realiza el entrenamiento y la evaluación de un modelo K-Nearest Neighbors (KNN) utilizando los datos de entrenamiento proporcionados. 
    Aplica validaciones previas y utiliza la técnica de hold-out o cross-validation según 
    la configuración especificada en entrenamiento. Retorna las métricas de evaluación del modelo.
    '''
    def knn(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
            validaciones = self.validaciones(entrenamiento, nombre_dataset)
            if validaciones is True:
                dataframe = self.preparacion_dataframe(entrenamiento, nombre_dataset)
                if dataframe is not None:
                    self.determinar_x_y(dataframe, entrenamiento.columnas_x, entrenamiento.objetivo_y)
                    if entrenamiento.tecnica == "hold-out":
                        # print("entro a hold out")
                        self.particion_dataset(dataframe, entrenamiento.cantidad)
                        self.modeloKNN = KNeighborsClassifier(n_neighbors=self.mejor_k())
                        self.modeloKNN.fit(self.XTrain,self.yTrain)
                        self.guardar_modelo(self.modeloKNN, 'knn')
                        self.yPredict = self.modeloKNN.predict(self.XTest)
                        print("-----------------yPredict-------------------")
                        #self.identificar_overffing_underffing(self.modeloKNN)
                        matriz = self.obtener_matriz_confusion('knn')
                        if  matriz is None:
                            return "Error al obtener la matriz de confusión"
                        metricas =self.metricas_hold_out()
                        self.guardar_info_modelos( 'KNN', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                        return metricas
                    elif entrenamiento.tecnica == "cross-validation":
                        #print("cantidad", entrenamiento.cantidad)
                        self.modeloKNN = KNeighborsClassifier(n_neighbors=self.find_best_n_neighbors())
                        metricas, matriz = self.validacion_cruzada(self.modeloKNN, entrenamiento.cantidad, 'knn')
                        self.guardar_info_modelos('KNN', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                        return metricas
                else:
                    return "Error con la preparación del dataframe"
            else:
                return validaciones
    
    '''
    Este método realiza el entrenamiento y la evaluación de un modelo de Regresión Logística utilizando los datos de entrenamiento proporcionados.
    Aplica validaciones previas y utiliza la técnica de hold-out o cross-validation según la 
    configuración especificada en entrenamiento. Retorna las métricas de evaluación del modelo.
    '''
    def regresion_logistica(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        validaciones = self.validaciones(entrenamiento, nombre_dataset)
        if validaciones is True:
            dataframe = self.preparacion_dataframe(entrenamiento, nombre_dataset)
            if dataframe is not None:
                self.determinar_x_y(dataframe, entrenamiento.columnas_x, entrenamiento.objetivo_y)
                if entrenamiento.tecnica == "hold-out":
                    self.particion_dataset(dataframe, entrenamiento.cantidad)
                    #mejores_parametros = self.mejores_parametros_regresion_log()
                    self.modeloRegLog = LogisticRegression(solver='newton-cg')
                    self.modeloRegLog.fit(self.XTrain, self.yTrain)
                    self.guardar_modelo(self.modeloRegLog, 'reglog')
                    #print("Entreno")
                    self.yPredict = self.modeloRegLog.predict(self.XTest)
                    #self.identificar_overffing_underffing(self.modeloRegLog)
                    matriz = self.obtener_matriz_confusion('reglog')
                    if  matriz is None:
                            return "Error al obtener la matriz de confusión"
                    metricas =self.metricas_hold_out()
                    self.guardar_info_modelos('Regresión Logística', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
                elif entrenamiento.tecnica == "cross-validation":
                    self.modeloRegLog = LogisticRegression()
                    metricas,matriz = self.validacion_cruzada(self.modeloRegLog, entrenamiento.cantidad, 'reglog')
                    self.guardar_info_modelos('Regresión Logística', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
        return "ok"
    

    '''
    Este método realiza el entrenamiento y la evaluación de un modelo Naive Bayes utilizando los datos de entrenamiento proporcionados. 
    Aplica validaciones previas y utiliza la técnica de hold-out o cross-validation según la 
    configuración especificada en entrenamiento. Retorna las métricas de evaluación del modelo.
    '''
    def naive_bayes(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        validaciones = self.validaciones(entrenamiento, nombre_dataset)
        if validaciones is True:
            dataframe = self.preparacion_dataframe(entrenamiento, nombre_dataset)
            if dataframe is not None:
                self.determinar_x_y(dataframe, entrenamiento.columnas_x, entrenamiento.objetivo_y)
                if entrenamiento.tecnica == "hold-out":
                    self.particion_dataset(dataframe, entrenamiento.cantidad)
                    modelo = GaussianNB()
                    modelo.fit(self.XTrain, self.yTrain)
                    self.guardar_modelo(modelo, 'naive_bayes')
                    self.yPredict = modelo.predict(self.XTest)
                    matriz = self.obtener_matriz_confusion('naive_bayes')
                    if matriz is None:
                        return "error"
                    metricas = self.metricas_hold_out()
                    self.guardar_info_modelos('Naive Bayes', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
                elif entrenamiento.tecnica == "cross-validation":
                    modelo = GaussianNB()
                    metricas, matriz = self.validacion_cruzada(modelo, entrenamiento.cantidad, 'naive_bayes')
                    self.guardar_info_modelos('Naive Bayes', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
            else:
                return "Error con la preparación del dataframe"

        else:
            return validaciones
    '''
    Este método utiliza GridSearchCV para encontrar el mejor valor de n_neighbors (número de vecinos) para un modelo KNN. Retorna el mejor valor encontrado.
    '''
    def find_best_n_neighbors(self):
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17]}
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid)
        grid_search.fit(self.x, self.y)
        best_n_neighbors = grid_search.best_params_['n_neighbors']
        return best_n_neighbors
    
    '''
    Este método realiza el entrenamiento y la evaluación de un modelo de Árbol de Decisión utilizando los datos de entrenamiento proporcionados y el nombre del dataset. 
    Aplica validaciones previas y utiliza la técnica de hold-out o cross-validation
    según la configuración especificada en entrenamiento. Retorna las métricas de evaluación del modelo.
    '''
    def arbol_decision(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        validaciones = self.validaciones(entrenamiento, nombre_dataset)
        if validaciones is True:
            dataframe = self.preparacion_dataframe(entrenamiento, nombre_dataset)
            if dataframe is not None:
                self.determinar_x_y(dataframe, entrenamiento.columnas_x, entrenamiento.objetivo_y)
                modelo = DecisionTreeClassifier()
                if entrenamiento.tecnica == "hold-out":
                    self.particion_dataset(dataframe, entrenamiento.cantidad)
                    modelo.fit(self.XTrain, self.yTrain)
                    self.guardar_modelo(modelo, 'arbol_decision')
                    self.yPredict = modelo.predict(self.XTest)
                    matriz = self.obtener_matriz_confusion('arbol_decision')
                    if  matriz is None:
                            return "Error al obtener la matriz de confusión"
                    metricas =  self.metricas_hold_out()
                    self.guardar_info_modelos('Árbol de decisión', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
                elif entrenamiento.tecnica == "cross-validation":
                    metricas, matriz = self.validacion_cruzada(modelo, entrenamiento.cantidad, 'arbol_decision')
                    self.guardar_info_modelos('Árbol de decisión', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
            else:
                return "Error con la preparación del dataframe"

        else:
            return validaciones
    '''
    Este método realiza el entrenamiento y la evaluación de un modelo de Regresión Lineal utilizando los datos de entrenamiento proporcionados y el nombre del dataset. 
    Aplica validaciones previas y utiliza la técnica de hold-out o cross-validation 
    según la configuración especificada en entrenamiento. Retorna las métricas de evaluación del modelo.
    '''  
    def regresion_lineal(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        validaciones = self.validaciones(entrenamiento, nombre_dataset)
        if validaciones is True:
            dataframe = self.preparacion_dataframe(entrenamiento, nombre_dataset)
            if dataframe is not None:
                self.determinar_x_y(dataframe, entrenamiento.columnas_x, entrenamiento.objetivo_y)
                if entrenamiento.tecnica == "hold-out":
                    modelo = LinearRegression()
                    self.particion_dataset(dataframe, entrenamiento.cantidad)
                    modelo.fit(self.XTrain, self.yTrain)
                    self.guardar_modelo(modelo, 'regresion_lineal')
                    self.yPredict = modelo.predict(self.XTest)
                    #if self.obtener_matriz_confusion('regresion_lineal') is None:
                        #return "error"
                    metricas = self.metricas_hold_out_regresionLineal()
                    self.guardar_info_modelos('Regresión Líneal', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, None)
                    return metricas
                elif entrenamiento.tecnica == "cross-validation":
                    modelo = LinearRegression()
                    metricas = self.validacion_cruzada_regresionLineal(modelo, entrenamiento.cantidad)
                    self.guardar_info_modelos('Regresión Líneal', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, None)
                    return metricas
            else:
                return "Error con la preparación del dataframe"
        #else:
            #return validaciones
    '''
    Este método calcula las métricas de evaluación (MAE, MSE y R2) para un modelo de Regresión Lineal utilizando la técnica de hold-out.
    '''
    def metricas_hold_out_regresionLineal(self):
        mae = mean_absolute_error(self.yTest,self.yPredict)
        mse = mean_squared_error(self.yTest,self.yPredict)
        r2 = r2_score(self.yTest,self.yPredict)

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared (R2): {r2}")

        return {'Mean Absolute Error (MAE)': mae, 'Mean Squared Error (MSE)': mse, 'R-squared (R2)': r2}
    
    '''
    Este método realiza la validación cruzada de un modelo de Regresión Lineal utilizando la técnica de cross-validation. 
    Calcula las métricas de evaluación (MSE, RMSE y R2) para cada iteración de validación y retorna los resultados.
    '''
    def validacion_cruzada_regresionLineal(self, modelo, cv):
        scores = cross_val_score(modelo, self.x, self.y, cv=cv, scoring='neg_mean_squared_error')
        mse_scores = -scores  # convertir a positivo
        rmse_scores = np.sqrt(mse_scores)
        r2_scores = cross_val_score(modelo, self.x, self.y, cv=cv, scoring='r2')

        print(f"Mean Squared Error (MSE): {mse_scores}")
        print(f"Root Mean Squared Error (RMSE): {rmse_scores}")
        print(f"R-squared (R2): {r2_scores}")

        modelo.fit(self.x, self.y)

        # Guardar el modelo entrenado
        self.guardar_modelo(modelo, 'regresion_lineal')

        return {'Mean Squared Error (MSE)': mse_scores.mean(), 
                'Root Mean Squared Error (RMSE)': rmse_scores.mean(), 
                'R-squared (R2)': r2_scores.mean()}

        
 

    '''
    recibe un objeto entrenamiento, el dataset indicado y realiza varias operaciones en un DataFrame. Filtra las columnas y valores según las columnas 
    especificadas en entrenamiento.columnas_x y entrenamiento.objetivo_y. Luego, crea un DataFrame con los valores seleccionados y lo retorna.
    '''
    def preparacion_dataframe(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        try:
            datos = self.mongo_service.obtener_ultimo_registro_por_nombre('Dataset', nombre_dataset)
            self.n_dataset = datos['nombre_dataset']
            # print('version',datos['version'])
            #print(datos)
            if datos:
                union = entrenamiento.columnas_x + [entrenamiento.objetivo_y]
                titulos = list(datos['titulos'])
                valores = list(datos['valores'])
                #print("TITULOS ", union)
                #print("VALORES ", valores)
                # Filtrar los títulos y valores según las columnas especificadas
                titulos_seleccionados = {c  for c in union if c in titulos}
                valores_seleccionados = []
                valorAux = {}
                for elemento in valores:
                    for c in elemento:
                        if c in titulos_seleccionados:
                            valorAux[c] = elemento[c]
                    valores_seleccionados.append(valorAux)
                    valorAux = {}
                #print("TITULOS SELECCIONADOS ", titulos_seleccionados)
                #print("VALORES SELECCIONADOS ", valores_seleccionados)
                df = pd.DataFrame(valores_seleccionados, columns=list(titulos_seleccionados))
                #print("DATAFRAME ", df)
                #print(df)
                print("DISTRIBUCION NORMAL")
                self.distribucion_normal(df)
                #print(df)
                print("-------------------------------------------")
                #print("CODIFICAR")
                df = self.dataframe_service.codificar_valores_cat(df, entrenamiento, self.n_dataset)
                print("NORMALIZAR")
                dataNumerica = self.dataframe_service.normalizar_informacion(dataframe=df, tipo=entrenamiento.normalizacion, objetivo_y= entrenamiento.objetivo_y)
                #print(df)
                print("REDONDEAR")
                dataNumerica = self.dataframe_service.redondear_datos(dataNumerica)
                #print(df)
                print("AQUI ESTA EL DATAFRAME")
                #print(df)
                return df
        except Exception as e:
            print("Error en la preparación del dataframe: ", e)
            return  None

    '''
    realiza validaciones,usa el dataset indicado llama al método preparacion_dataframe y realiza diferentes operaciones dependiendo de la 
    técnica especificada en entrenamiento.tecnica. Si la técnica es "hold-out", realiza una partición del dataset y entrena un modelo de SVM. 
    Si la técnica es "cross-validation", entrena un modelo de SVM con validación cruzada. Guarda la información del modelo y retorna las métricas.
    '''
    def svm(self, entrenamiento: InfoEntrenamiento, nombre_dataset):
        validaciones = self.validaciones(entrenamiento, nombre_dataset)
        if validaciones is True:
            dataframe = self.preparacion_dataframe(entrenamiento, nombre_dataset)
            if dataframe is not None:
                self.determinar_x_y(dataframe, entrenamiento.columnas_x, entrenamiento.objetivo_y)
                if entrenamiento.tecnica == "hold-out":
                    self.particion_dataset(dataframe, entrenamiento.cantidad)
                    scaler = StandardScaler()
                    self.XTrain = scaler.fit_transform(self.XTrain)
                    self.XTest = scaler.transform(self.XTest)
                    modelo = SVC(C=1.0, kernel='rbf', gamma='scale')
                    modelo.fit(self.XTrain, self.yTrain)
                    self.guardar_modelo(modelo, 'svm')
                    self.yPredict = modelo.predict(self.XTest)
                    matriz = self.obtener_matriz_confusion('svm')
                    if matriz is None:
                            return "Error al obtener la matriz de confusión"
                    metricas = self.metricas_hold_out()
                    self.guardar_info_modelos('Máquinas de soporte vectorial (SVM)', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
                elif entrenamiento.tecnica == "cross-validation":
                    modelo = SVC(C=1.0, kernel='rbf', gamma='scale')
                    metricas, matriz = self.validacion_cruzada(modelo, entrenamiento.cantidad, 'svm')
                    self.guardar_info_modelos('Máquinas de soporte vectorial (SVM)', entrenamiento.normalizacion, entrenamiento.tecnica, metricas, matriz.tolist())
                    return metricas
            else:
                return "Error con la preparación del dataframe"
        else:
            return validaciones

    '''
     guarda la información de las métricas de un modelo en una base de datos.
    '''
    def guardar_info_modelos(self, nombre_modelo, normalizacion, tecnica, metricas, matriz):
        fecha_actual = datetime.datetime.now().strftime('%d-%m-%Y')
        if matriz is not None:
            info = {'fecha':fecha_actual,"nombre_dataset":self.n_dataset, 'nombre_algoritmo': nombre_modelo, 'normalizacion': normalizacion, 'tecnica': tecnica,'metricas': metricas, 'matriz_confusion': matriz}
        else:    
            info = {'fecha':fecha_actual,"nombre_dataset":self.n_dataset, 'nombre_algoritmo': nombre_modelo, 'normalizacion': normalizacion, 'tecnica': tecnica,'metricas': metricas}
        id = self.mongo_service.guardar_json_metricas(info, 'InformacionModelos')
        print("ID ", id)

    '''
     analiza la distribución normal de las columnas numéricas en un DataFrame.
    '''
    def distribucion_normal(self, dataframe):
        #print(dataframe.info())
        numerico = dataframe.select_dtypes(np.number)
        #print("Numerico ", numerico)
        normal=[]
        noNormal=[]
        for i in numerico:
            datosColumna = numerico[i]
            stat,p=normaltest(datosColumna)
            if p > 0.5:
                normal.append(i)
            else:
                noNormal.append(i)

        print("Con distribucion normal: ",normal)
        print("Sin distribucion normal: ",noNormal)

    ''''
     realiza una partición del dataset en conjuntos de entrenamiento y prueba.
    '''
    def particion_dataset(self, dataframe, porcentaje):
        if porcentaje >1:
            porcentaje = porcentaje/100
        self.XTrain,self.XTest,self.yTrain,self.yTest=train_test_split(self.x,self.y,test_size=porcentaje, random_state=2)


    '''
    asigna las columnas columnas y el objetivo objetivo a los atributos self.x y self.y respectivamente.
    '''
    def determinar_x_y(self, dataframe, columnas, objetivo):
        if not isinstance(dataframe, pd.DataFrame):
             raise TypeError("El parámetro 'dataframe' debe ser un DataFrame de pandas.")
        else: 
            self.x=dataframe.loc[:,columnas] # obtener valores de x
            #print("X ", self.x)
            self.y=dataframe[objetivo]
            #print("Y ", self.y)

    '''
     realiza una búsqueda del mejor valor de k para el algoritmo KNN utilizando validación cruzada.
    '''
    def mejor_k(self):
         # Posibles valores que puede tomar
        kvalores = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17]}

        # obtener el mejor valor por validación cruzada
        grid = GridSearchCV(KNeighborsClassifier(), kvalores)

        # Ajustar a los datos de entrenamiento
        grid.fit(self.XTrain, self.yTrain)

        # valor óptimo de k
        print("Mejor valor de k: ", grid.best_params_['n_neighbors'])
        return grid.best_params_['n_neighbors']
    
    '''
    calcula y guarda la matriz de confusión para un modelo dado.
    '''
    def obtener_matriz_confusion(self, nombre_modelo):
        try:
            # print("Matriz de confusion")
            # print(self.yTest)
            # print("/////////////")
            # print(self.yPredict)
            # Convertir self.yTestKNN a una matriz numpy
            y_test = np.array(self.yTest.values)

            # Asegurarse de que self.yPredictKNN sea una matriz numpy
            y_pred = np.array(self.yPredict)
            matrizKNN = confusion_matrix(y_test, y_pred)
            #print(matrizKNN)
            sb.heatmap(matrizKNN, annot=True, cmap="Blues")
            plt.title(f"Matriz de Confusión {nombre_modelo} - Hold-Out")
            ruta_guardado = "app/files/imgs/modelos/matrices-confusion/"
            os.makedirs(ruta_guardado, exist_ok=True)
            plt.savefig(os.path.join(ruta_guardado, f"{self.n_dataset}-{nombre_modelo}-ho-matriz_confusion.png"))
            plt.close()
            return matrizKNN
        except Exception as e:
            return None

    '''
    evalúa el overfitting y underfitting de un modelo.
    '''
    def identificar_overffing_underffing(self, modelo):
        #  print("///////////777")
        #  print("IDENTIFICAR OVERFITTING Y UNDERFITTING")
         scores = cross_val_score(modelo, self.x, self.y, cv=5)

         # calcular la media y la desviación estándar de las puntuaciones de precisión
         meanScore = np.mean(scores)
         stdScore = np.std(scores)

         # hacer predicciones en los datos de entrenamiento y prueba
         yPredTrain = modelo.predict(self.XTrain)
         yPredTest = modelo.predict(self.XTest)

         # calcular la precisión en los datos de entrenamiento y prueba
         accuracyTrain = accuracy_score(self.yTrain, yPredTrain)
         accuracyTest = accuracy_score(self.yTest, yPredTest)

         print("Precisión en los datos de entrenamiento:", accuracyTrain)
         print("Precisión en los datos de prueba:", accuracyTest)

         # imprimir las puntuaciones de precisión y sus estadísticas
         print('Puntuaciones de precisión:', scores.mean())
         print('Precisión media:', meanScore)
         print('Desviación estándar de la precisión:', stdScore)


    
    ''''
    
    calcula y muestra las métricas de evaluación para un modelo utilizando la técnica hold-out.
    '''
    def metricas_hold_out(self):
        accuracy = accuracy_score(self.yTest,self.yPredict) # proporción de predicciones correctas del modelo
        precision = precision_score(self.yTest,self.yPredict, average = 'weighted', zero_division=1) # proporción de predicciones positivas que fueron correctas
        recall = recall_score(self.yTest, self.yPredict, average = 'weighted') # proporción de positivos reales que se identificaron correctamente
        f1 = f1_score(self.yTest, self.yPredict, average = 'weighted') # medida armónica de precision y recall
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    

    '''
    verifica si las columnas especificadas son válidas en el dataset indicado.
    '''
    def validar_columnas(self, columnas, nombre_dataset):
        self.disponibles= self.processing_service.columnas_disponibles_dataset(nombre_dataset)
        self.disponibles = self.reemplazar_caracteres_especiales(self.disponibles)
        columnas = self.reemplazar_caracteres_especiales(columnas)
        # print("Columnas disponibles: ", self.disponibles)
        # print("columnas que llegaron ", columnas)
        if self.disponibles != None:
            if set(columnas) <= set(self.disponibles) or columnas in self.disponibles:
                return True
            else:
                return False
        else:
            return None
        
    def reemplazar_caracteres_especiales(self, columnas):
        auxCol = []
        print(len(columnas))
        print(columnas)
        if isinstance(columnas, list):
            for txt in columnas:
                auxCol.append(txt.upper().replace("Ñ","N"))
            return auxCol
        else:
            columnas = columnas.upper().replace("Ñ","N")
            return columnas
    '''
    realiza validación cruzada con un modelo dado y calcula las métricas promedio y la matriz de confusión.
    '''
    def validacion_cruzada(self, modelo, cv, nombre_modelo):
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=1, average='macro'),
            'recall': 'recall_macro',
            'f1': 'f1_macro'
        }
        print("Tipo de dato: ",type(self.x))
        # Obtener las métricas para cada pliegue
        resultados = cross_validate(modelo, self.x, self.y, cv=cv, scoring=scoring)

       # Obtener las métricas promedio
        accuracy_media = resultados['test_accuracy'].mean()
        precision_media = resultados['test_precision'].mean()
        recall_media = resultados['test_recall'].mean()
        f1_media = resultados['test_f1'].mean()

        print("Accuracy media:", accuracy_media)
        print("Precisión media:", precision_media)
        print("Exhaustividad media:", recall_media)
        print("Puntuación F1 media:", f1_media)

        # Obtener la matriz de confusión promedio
        y_pred = cross_val_predict(modelo, self.x, self.y, cv=cv)
        matriz_confusion = confusion_matrix(self.y, y_pred)

        modelo.fit(self.x, self.y)

        # Guardar el modelo entrenado
        self.guardar_modelo(modelo, 'knn')

        #Guardar modelo

        # Visualizar la matriz de confusión
        #plt.figure(figsize=(8, 6))
        sb.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusión {nombre_modelo} - Validación Cruzada")
        ruta_guardado = "app/files/imgs/modelos/matrices-confusion/"
        os.makedirs(ruta_guardado, exist_ok=True)
        plt.savefig(os.path.join(ruta_guardado, f"{self.n_dataset}-{nombre_modelo}-cv-matriz_confusion.png"))
        plt.close()
        # plt.xlabel("Predicciones")
        # plt.ylabel("Etiquetas Verdaderas")


        return {'accuracy': accuracy_media, 'precision': precision_media, 'recall': recall_media, 'f1': f1_media}, matriz_confusion
    
    '''
     Esta función guarda un modelo en un archivo pickle. Toma el modelo, el nombre del modelo y del dataset como argumentos y utiliza la biblioteca pickle para guardar el modelo en un archivo.
    '''
    def guardar_modelo(self, modelo, nombre_modelo):
        #print(self.XTest.columns)
        try:
           ruta_directorio = f'app/files/modelos'
           if not os.path.exists(ruta_directorio):
            os.makedirs(ruta_directorio)
           ruta_modelo = os.path.join(ruta_directorio, f'{self.n_dataset}-{nombre_modelo}.pkl')
           with open(ruta_modelo, 'wb') as archivo:
                pickle.dump(modelo, archivo)
        except Exception as e:
            return None

    '''
    Esta función realiza una predicción utilizando un modelo previamente entrenado para el dataset indicado. 
    Primero, obtiene los datos necesarios para realizar la predicción del servicio Mongo 
    utilizando el algoritmo proporcionado. Luego, verifica si los valores de predicción 
    coinciden con los campos con los que se entrenó el modelo. Si coinciden, codifica los valores 
    de predicción para que coincidan con los valores del modelo y carga el modelo desde el archivo. 
    A continuación, crea un DataFrame con los datos codificados y realiza la predicción utilizando el modelo cargado. 
    Finalmente, decodifica el resultado de la predicción y lo devuelve.
    ''' 
    def prediccion(self, prediccion: PrediccionModel, nombre_dataset):
        try:
            if self.file_service.verificar_dataset(nombre_dataset) is False:
                return self.utils.prueba(msg=f"No existe el dataset {nombre_dataset}")
            datos = self.mongo_service.obtener_datos_algoritmo('RepresentacionCodificacion', prediccion.algoritmo,nombre_dataset )
            
            
            #print(columns_predecir)
            # Verificar si el diccionario solo contiene las claves presentes en la lista
            if datos:
                columns_x = [valor.lower() for valor in datos['x']]
                columns_predecir = [valor.lower() for valor in prediccion.valores_predecir.keys()]
                clavesReemplazadas = []
                # Validar que la información para predecir corresponda con los campos con los que se entrenó el modelo
                cant = 0
                for clave in columns_predecir:
                    if clave in columns_x:
                        cant = cant + 1
                if cant == len(columns_x) and len(columns_x) == len(columns_predecir):
                    # Se codifican los valores recibidos para que coincidan con los valores con los que se entrenó el modelo
                    prediccion = self.__codificar_valores_recibidos(datos, columns_x, prediccion, clavesReemplazadas)
                    # Cargar el modelo desde el archivo
                    ruta_modelo = 'app/files/modelos/'
                    modelo = self.__obtener_modelo(ruta_modelo, prediccion.algoritmo, nombre_dataset)
                    # Crear un dataframe con los datos codificados
                    ejemplo_prueba = pd.DataFrame(prediccion.valores_predecir, index=[0])
                    if modelo:
                    #Realizar predicción utilizando el modelo cargado y el ejemplo de prueba
                        prediccion = modelo.predict(ejemplo_prueba)
                        for data in datos["datosY"]:
                            #print("ENTRO")
                            if data["valor_codificado"] == round(prediccion[0]):
                                    prediccion = data["valor_original"]
                                    break            
                        return self.utils.prueba(msg=f"La predicción es: {prediccion}")
                    else:
                        return self.utils.prueba(msg=f"No se encuentra el modelo para el algoritmo {prediccion.algoritmo} del dataset {nombre_dataset}")    
                    
                else:
                    return self.utils.prueba(msg=f"Los valores a predecir debe tener las columnas que se especificaron como x para entrenar el algoritmo {datos['x']}")

            else:
                return self.utils.prueba(msg=f"No se encuentra el modelo para el algoritmo {prediccion.algoritmo}")
        except FileNotFoundError as e:
            return self.utils.prueba(msg=f"No se encuentra el modelo para el algoritmo {prediccion.algoritmo}")
    

    ''''
    Esta función se utiliza en el método prediccion para codificar los valores de predicción según los valores utilizados durante el entrenamiento del modelo. 
    Recorre los campos utilizados para entrenar el modelo y compara los valores de predicción con los valores originales para reemplazarlos por sus equivalentes codificados.
    '''
    def __codificar_valores_recibidos(self, datos, columns_x, prediccion, clavesReemplazadas):
        ya = False
        for clave in columns_x:
                        for claveO in datos['x']:
                            for valor in datos["datosX"]:
                                for claveP in prediccion.valores_predecir.keys():
                                    if claveP not in clavesReemplazadas and not isinstance(prediccion.valores_predecir[claveP], (int, float, complex)):
                                        if self.utils.arreglar_nombre(clave) == self.utils.arreglar_nombre(claveO):
                                            for i in valor:
                                                #pdata = self.utils.arreglar_nombre(data[info][i]['valor_original'])
                                                # print(claveO)
                                                # print(clave)
                                                for j in valor[i]:
                                                    # print(j)
                                                    if not isinstance(j['valor_original'], (int, float, complex)):
                                                        if self.utils.arreglar_nombre(j['valor_original']) == self.utils.arreglar_nombre(prediccion.valores_predecir[claveP]):
                                                            clavesReemplazadas.append(claveP)
                                                            prediccion.valores_predecir[claveP] = j['valor_codificado']
                                                            ya = True
                                                            break
                                                if ya:
                                                    break
        return prediccion
    
    '''
    Esta función se utiliza en el método prediccion para obtener el modelo almacenado en un archivo pickle. 
    Construye la ruta del archivo en función del nombre del algoritmo proporcionado y carga el modelo desde el archivo utilizando la biblioteca pickle.
    '''
    def __obtener_modelo(self, ruta_modelo, nombre_algoritmo, nombre_dataset):
            nombre_algoritmo = self.utils.arreglar_nombre(nombre_algoritmo).lower()
            patron = f'{nombre_dataset}*{nombre_algoritmo}.pkl'
            if nombre_algoritmo == 'knn':
                patron = f'{nombre_dataset}*knn.pkl'
                ruta= os.path.join(ruta_modelo, patron)
            elif nombre_algoritmo == 'svm':
                patron = f'{nombre_dataset}*svm.pkl'
                ruta= os.path.join(ruta_modelo, patron)
            elif nombre_algoritmo == 'naivebayes':
                patron = f'{nombre_dataset}*naive_bayes.pkl'
                ruta= os.path.join(ruta_modelo, patron)
            elif nombre_algoritmo == 'regresionlogistica':
                patron = f'{nombre_dataset}*reglog.pkl'
                ruta= os.path.join(ruta_modelo, patron)
            elif nombre_algoritmo == 'arboldedecision':
                patron = f'{nombre_dataset}*arbol_decision.pkl'
                ruta= os.path.join(ruta_modelo, patron)
            elif nombre_algoritmo == 'regresionlineal':
                patron = f'{nombre_dataset}*regresion_lineal.pkl'
                ruta= os.path.join(ruta_modelo, patron)
            if ruta:
                archivos = glob.glob(ruta)
                with open(archivos[0], 'rb') as archivo:
                    modelo_cargado = pickle.load(archivo)
                return modelo_cargado
            else:
                return None

    '''
    Esta función obtiene los tres mejores algoritmos entrenados para el dataset indicado en función de la métrica de precisión. Utiliza el servicio Mongo para obtener las métricas de los modelos y 
    las ordena en orden descendente según la precisión. Luego, selecciona los tres primeros elementos y los devuelve como resultado.
    '''
    def obtener_top3_algoritmos(self, nombre_dataset):
        if self.file_service.verificar_dataset(nombre_dataset) is False:
                return f"No existe el dataset {nombre_dataset}"
        metricas= self.mongo_service.obtener_mejores_algoritmos('InformacionModelos', nombre_dataset=nombre_dataset)
        datos = []
        if metricas:
            if len(metricas) > 0:
                # Ordenar la lista en función de la precisión en orden descendente
                dic_ordenado = sorted(metricas, key=lambda x: x['metricas'].get('accuracy', -1), reverse=True)

                # Obtener los tres primeros elementos
                mejores_tres = dic_ordenado[:3]
                for  data in mejores_tres:
                        datos.append({'nombre_algoritmo': data['nombre_algoritmo'], 'normalizacion': data['normalizacion'], 'tecnica': data['tecnica'], 'metricas': data['metricas']})
                        #print(datos)
                return self.utils.prueba(msg=f"Top 3 mejores algoritmos entrenados con el dataset {nombre_dataset}", datos=datos)
       
            return self.utils.error(msg="No se encontraron métricas")