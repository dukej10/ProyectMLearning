from fastapi import FastAPI
from app.routes.mlearning_route import todo_api_router
from app.routes.processing_route import processing_api_router

app = FastAPI()

app.include_router(todo_api_router)

