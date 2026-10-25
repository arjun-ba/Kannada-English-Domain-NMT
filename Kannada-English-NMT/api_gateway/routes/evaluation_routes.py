"""
API Gateway: Evaluation Routes
Endpoints for computing BLEU metrics, precision breakdowns, and validation benchmarks.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/api/v1/evaluation", tags=["Evaluation"])

class BleuRequest(BaseModel):
    hypothesis: str = Field(..., example="Farmers should get the soil tested.")
    reference: str = Field(..., example="Farmers should test the soil.")

def init_routes(evaluation_service, translation_engine, dataset_manager):
    @router.post("/bleu")
    async def compute_bleu(req: BleuRequest):
        if not req.hypothesis.strip() or not req.reference.strip():
            raise HTTPException(status_code=400, detail="Hypothesis and reference must not be empty.")
        return evaluation_service.calculate_bleu(req.hypothesis, req.reference)

    @router.get("/benchmark")
    async def run_benchmark():
        # Evaluate model against validation set
        pairs = []
        for kn, ref_en, _ in dataset_manager.val_pairs:
            trans_res = translation_engine.translate(kn, direction="kn->en")
            pairs.append({
                "translated": trans_res["translated_text"],
                "reference": ref_en
            })
        
        benchmark_res = evaluation_service.evaluate_test_set(pairs)
        return benchmark_res

    return router
