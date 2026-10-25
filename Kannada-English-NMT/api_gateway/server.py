"""
API Gateway & Orchestrator Server
Microservice Architecture for Kannada-English Domain NMT.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from services.dataset_service.data_loader import DatasetManager
from services.translation_service.translator import TranslationEngine
from services.evaluation_service.evaluator import EvaluationService

from api_gateway.routes.translation_routes import init_routes as init_translation_routes
from api_gateway.routes.dataset_routes import init_routes as init_dataset_routes
from api_gateway.routes.evaluation_routes import init_routes as init_evaluation_routes
from api_gateway.routes.health_routes import init_routes as init_health_routes

def create_app() -> FastAPI:
    """Factory creating the FastAPI Gateway."""
    app = FastAPI(
        title="Kannada-English Domain NMT System",
        description="Neural Machine Translation Suite for Agriculture Domain-Specific Text with Attention Alignment & Microservices Architecture",
        version="2.0.0"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Microservices
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "dataset.csv")

    dataset_manager = DatasetManager(dataset_path)
    dataset_manager.load_data()

    translation_engine = TranslationEngine(dataset_manager)
    evaluation_service = EvaluationService()

    # Register Microservice Routes
    app.include_router(init_translation_routes(translation_engine))
    app.include_router(init_dataset_routes(dataset_manager))
    app.include_router(init_evaluation_routes(evaluation_service, translation_engine, dataset_manager))
    app.include_router(init_health_routes(translation_engine, dataset_manager, evaluation_service))

    # Mount Static Web Assets
    static_dir = os.path.join(base_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Kannada-English NMT Microservices Gateway is Running. Access /docs for API documentation."}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_gateway.server:app", host="127.0.0.1", port=8000, reload=True)
