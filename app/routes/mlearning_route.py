from fastapi import APIRouter, File, UploadFile
from app.Controllers.mlearning_controller import MLearningController
from app.Controllers.processing_controller import ProcessingController
from app.Controllers.file_controller import FileController

from app.models.todos_model import Todo
from bson import ObjectId
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

todo_api_router = APIRouter()
file_controller = FileController()

processing_controller = ProcessingController()
mlearning_controller = MLearningController()

# retrieve
@todo_api_router.get("/")
async def get_todos():
    return {"msg":"todos"}

# @todo_api_router.get("/{id}")
# async def get_todo(id: str):
#     return {"msg":"obtener"}

@todo_api_router.get("/histograma")
async def get_histogram():
    img_path =  await processing_controller.obtener_histograma()
    return FileResponse(img_path)

@todo_api_router.get("/matriz-correlacion")
async def get_matriz_correlacion():
    img_path =  await processing_controller.obtener_matriz_correlacion()
    if isinstance(img_path, str):
        return FileResponse(img_path)
    else:
       return img_path
    

@todo_api_router.get('/descarte')
def descarte():
    return processing_controller.descarte()

@todo_api_router.get('/imputacion')
def imputacion():
    return processing_controller.imputation_data()

@todo_api_router.get('/generar-imagenes')
def generar_imagenes():
    return processing_controller.generar_imagenes_analisis()

# Upload File
@todo_api_router.post("/upload", description="Cargar set de datos")
async def upload_file(file: UploadFile = File(...)):
    # Process the uploaded file
    return await file_controller.upload_file(file)

# post
@todo_api_router.post("/")
async def create_todo(todo: Todo):
    return {"msg":"crear"}

@todo_api_router.get("/tipos-datos")
async def get_types():
    return processing_controller.get_types()
# update
@todo_api_router.put("/{id}")
async def update_todo(id: str, todo: Todo):
    return {"msg":"actualizar"}
# delete
@todo_api_router.delete("/{id}")
async def delete_todo(id: str):
    
    return {"status": "ok"}

@todo_api_router.get("/knn")
async def knn():
    return mlearning_controller.knn()

@todo_api_router.get("/regresion-logistica")
async def regresion_logistica():
    return mlearning_controller.reg_logistica()