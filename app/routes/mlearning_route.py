from fastapi import APIRouter, File, UploadFile
from app.Controllers.mlearning_controller import MLearningController
from app.Controllers.processing_controller import ProcessingController
from app.Controllers.file_controller import FileController
from fastapi import Body
from app.models.prediccion_model import PrediccionModel

from app.models.entrenamiento_model import InfoEntrenamiento
from bson import ObjectId
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

todo_api_router = APIRouter()
file_controller = FileController()

processing_controller = ProcessingController()
mlearning_controller = MLearningController()


'''
clase que maneja todas las rutas de nuestra API
'''


#ruta para el index de la api
@todo_api_router.get("/")
async def get_todos():
    return FileResponse("app/templates/index.html")

@todo_api_router.get("/datasets", description="Obtiene los datasets cargados en la aplicación")
async def get_datasets():
    return file_controller.obtener_datasets()

#ruta que me trae el servicio de histograma
@todo_api_router.get("/histograma", description="Obtiene el histograma de las variables numéricas del dataset")
async def get_histogram(nombre_dataset: str):
    img_path =  await processing_controller.obtener_histograma(nombre_dataset)
    if isinstance(img_path, str):
        return FileResponse(img_path)
    else:
       return img_path

#ruta que usa el servicio de matriz de correlaccion
@todo_api_router.get("/matriz-correlacion", description="Obtiene la matriz de correlacción de las variables numéricas del dataset")
async def get_matriz_correlacion(nombre_dataset:str):
    img_path =  await processing_controller.obtener_matriz_correlacion(nombre_dataset)
    if isinstance(img_path, str):
        return FileResponse(img_path)
    else:
       return img_path
    
#metodo que impelementa el servicio de descarte
@todo_api_router.get('/descarte', description="Utilizar descarte como técnica de tratamiento de datos faltantes del dataset " )
def descarte(nombre_dataset: str):
    return processing_controller.descarte(nombre_dataset)

#ruta que maneja el servicio de imputacion
@todo_api_router.get('/imputacion',description="Utilizar imputación como técnica de tratamiento de datos faltantes del dataset ")
def imputacion(nombre_dataset: str):
    return processing_controller.imputation_data(nombre_dataset)

#metodo que inciar el servicio de generar imagener
@todo_api_router.get('/generar-imagenes', description="Genera las imagenes de analisis de los datos histograma y matriz")
def generar_imagenes(nombre_dataset:str):
    return processing_controller.generar_imagenes_analisis(nombre_dataset)

# Uruta oara la cargar de archivos
@todo_api_router.post("/upload", description="Cargar set de datos")
async def upload_file(file: UploadFile = File(...)):
    # Process the uploaded file
    return await file_controller.upload_file(file)

#ruta para obtener el tipo de datos
@todo_api_router.get("/tipos-datos", description="Obtiene los tipos de datos de los datos del dataset")
async def get_types(nombre_dataset):
    return processing_controller.get_types(nombre_dataset)
#ruta para el servicio general de entrenamiento
@todo_api_router.post("/entrenamiento", description="Entrenamiento de algoritmo \n  nombre_algoritmo: nombre del algoritmo a utilizar, columnas_x: variables a usar para x, objetivo_y: columna objetivo, tecnica: hold-out o cross-validation, cantidad: % partición dataset o número de folds para cv, normalizacion: min-max o standarscaler")
async def entrenamiento_algoritmo(nombre_dataset:str,entrenamiento: InfoEntrenamiento= Body(..., example={
  "nombre_algoritmo": "knn",
  "columnas_x": [
    "Area", "Categoria", "genero", "agrupa","valor","ano","mes"
  ],
  "objetivo_y": "indicador",
  "tecnica": "hold-out",
  "cantidad": 20,
  "normalizacion": "min-max"
})):
   print(entrenamiento)
   return mlearning_controller.algoritmos(entrenamiento, nombre_dataset)

#ruta para el servicio de matriz de confusion
@todo_api_router.get("/matriz-confusion-algoritmo={nombre_algoritmo}", description="Permite obtener la matriz de confusión de un algoritmo en específico, se debe enviar el nombre del algoritmo como parámetro")
async def matriz_confusional(nombre_dataset: str,nombre_algoritmo: str):
    img_path = await processing_controller.obtener_matriz_confusion(nombre_algoritmo, nombre_dataset)
    if isinstance(img_path, str):
        return FileResponse(img_path)
    else:
       return img_path
#ruta para obtener la description de los algoritmos entrenados
@todo_api_router.get("/metricas-algoritmos-entrenados", description="Permite obtener las métricas que obtuvieron los algoritmos entrenados")
async def metricas_algoritmos_entrenados(nombre_dataset: str):
    return processing_controller.metricas_algoritmos_entrenados(nombre_dataset=nombre_dataset)

#ruta del servicio de prediccion 
@todo_api_router.post("/prediccion", description="Predicción de algoritmo \n  nombre_algoritmo: nombre del algoritmo a utilizar, columnas_x: variables a usar para x, objetivo_y: columna objetivo, tecnica: hold-out o cross-validation, cantidad: % partición dataset o número de folds para cv, normalizacion: min-max o standarscaler")
async def prediccion_algoritmo(nombre_dataset:str, prediccion: PrediccionModel= Body(..., example={
            'algoritmo': 'knn',
            'valores_predecir': {
                "Area": "Manizales",
                "Categoria": "arma blanca / cortopunzante",
                "genero": "masculino",
                "agrupa": "adultos",
                "valor": 20000,
                "ano": 2015,
                "mes": "Enero"             }
        })):
    return mlearning_controller.prediccion(prediccion, nombre_dataset)

#ruta para obtener el top de los algoritmos que se han entrenado
@todo_api_router.get("/top3-algoritmos", description="Permite obtener los 3 mejores algoritmos entrenados")
async def top3_algoritmos(nombre_dataset: str):
    return mlearning_controller.obtener_mejores_algoritmos(nombre_dataset)
    
