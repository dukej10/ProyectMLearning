from fastapi import APIRouter, File, UploadFile
from app.Controllers.processing_controller import ProcessingController
from app.Controllers.file_controller import FileController

from bson import ObjectId
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

todo_api_router = APIRouter()
file_controller = FileController()

processing_controller = ProcessingController()

# retrieve
@todo_api_router.get("/")
async def get_todos():
    return {"msg":"todos"}

# @todo_api_router.get("/{id}")
# async def get_todo(id: str):
#     return {"msg":"obtener"}

@todo_api_router.get("/histograma")
async def get_histogram():
     imagen_path = "app/files/imgs/histogramas/2023-05-22_19-28-49_histogramas.png"
     return FileResponse(imagen_path)
# Upload File
@todo_api_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Process the uploaded file
    return await file_controller.upload_file(file)
    file_info = {
        "filename": file.filename,
        "content_type": file.content_type,
    }
    return {"file_info": file_info}




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