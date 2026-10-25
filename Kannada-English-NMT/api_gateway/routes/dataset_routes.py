"""
API Gateway: Dataset Routes
Endpoints for exploring the domain dataset, vocabulary metrics, and adding custom pairs.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/api/v1/dataset", tags=["Dataset"])

class AddSampleRequest(BaseModel):
    kannada: str = Field(..., example="ಹನಿ ನೀರಾವರಿ ಪದ್ಧತಿ ಉಪಯುಕ್ತವಾಗಿದೆ.")
    english: str = Field(..., example="Drip irrigation system is useful.")
    topic: Optional[str] = Field("Irrigation", example="Irrigation")

def init_routes(dataset_manager):
    @router.get("/summary")
    async def get_summary():
        return dataset_manager.get_summary()

    @router.get("/samples")
    async def get_samples():
        return dataset_manager.get_all_samples()

    @router.post("/add")
    async def add_sample(req: AddSampleRequest):
        if not req.kannada.strip() or not req.english.strip():
            raise HTTPException(status_code=400, detail="Both Kannada and English texts are required.")
        summary = dataset_manager.add_custom_sentence(req.kannada, req.english, req.topic or "Custom")
        return {"status": "success", "message": "Sentence added successfully", "summary": summary}

    return router
