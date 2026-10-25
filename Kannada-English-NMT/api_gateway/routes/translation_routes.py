"""
API Gateway: Translation Routes
Handles translation requests, model training status, and attention weight matrix delivery.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/api/v1/translation", tags=["Translation"])

class TranslateRequest(BaseModel):
    text: str = Field(..., example="ರೈತರು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಬೇಕು.")
    direction: Optional[str] = Field("kn->en", example="kn->en")
    domain: Optional[str] = Field("Agriculture", example="Agriculture")

class TranslateResponse(BaseModel):
    source_text: str
    cleaned_source: str
    translated_text: str
    direction: str
    domain: str
    source_tokens: List[str]
    target_tokens: List[str]
    attention_matrix: List[List[float]]
    domain_terms: List[Dict[str, Any]]
    confidence: float
    method: str

def init_routes(translation_engine):
    @router.post("/translate", response_model=TranslateResponse)
    async def translate(req: TranslateRequest):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
        result = translation_engine.translate(
            text=req.text,
            direction=req.direction,
            domain=req.domain
        )
        return result

    @router.get("/training-history")
    async def get_training_history():
        return {
            "is_trained": translation_engine.is_trained,
            "history": translation_engine.training_history
        }

    return router
