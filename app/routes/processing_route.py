from fastapi import APIRouter, File, UploadFile

from app.Controllers.processing_controller import ProcessingController

processing_api_router = APIRouter()

processing_controller = ProcessingController()

@processing_api_router.get("/tipos-datos")
async def get_types():
    return processing_controller.get_types()

@processing_api_router.get("/histograma")
async def get_histogram():
    return processing_controller.obtener_histograma()