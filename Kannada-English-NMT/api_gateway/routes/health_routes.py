"""
API Gateway: Health & Microservice Status Routes
Provides health checks and architecture component status for each microservice.
"""

import time
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/health", tags=["System Health"])

start_time = time.time()

def init_routes(translation_engine, dataset_manager, evaluation_service):
    @router.get("")
    async def get_health():
        uptime = round(time.time() - start_time, 2)
        return {
            "gateway_status": "ONLINE",
            "uptime_seconds": uptime,
            "microservices": {
                "preprocessing_service": {
                    "status": "HEALTHY",
                    "features": ["Unicode NFC Normalization", "Kannada Diacritic Cleaning", "Domain Lexicon Matching"]
                },
                "translation_service": {
                    "status": "HEALTHY" if translation_engine.is_trained else "INITIALIZING",
                    "architecture": "Bidirectional Seq2Seq GRU with Bahdanau Attention",
                    "device": str(translation_engine.device),
                    "parameters_trained": translation_engine.is_trained
                },
                "dataset_service": {
                    "status": "HEALTHY" if dataset_manager.loaded else "UNINITIALIZED",
                    "samples_loaded": len(dataset_manager.df),
                    "kn_vocab_size": dataset_manager.kannada_vocab.num_words,
                    "en_vocab_size": dataset_manager.english_vocab.num_words
                },
                "evaluation_service": {
                    "status": "HEALTHY",
                    "metrics_available": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "Cumulative BLEU", "Brevity Penalty"]
                }
            }
        }

    return router
