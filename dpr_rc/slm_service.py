"""
SLM (Small Language Model) Service for DPR-RC System

REFACTORED ARCHITECTURE (SOLID):
This file is now a THIN FACADE - it only handles:
- FastAPI endpoint setup
- HTTP request/response handling
- Delegation to domain services

All business logic is in:
- dpr_rc/domain/slm/ - Domain services (PromptBuilder, ResponseParser, InferenceEngine)
- dpr_rc/infrastructure/slm/ - Model backend, factory
"""

import os
from typing import Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .logging_utils import StructuredLogger
from .models import ComponentType
from .infrastructure.slm import SLMFactory

# Configuration
SLM_MODEL = os.getenv("SLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
SLM_PORT = int(os.getenv("PORT", os.getenv("SLM_PORT", "8080")))

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

# Global inference engine
_inference_engine: Optional[Any] = None


def get_inference_engine():
    """Lazy initialize inference engine from factory."""
    global _inference_engine
    if _inference_engine is None:
        logger.logger.info(f"Loading model: {SLM_MODEL}")
        _inference_engine = SLMFactory.create_from_env()
        logger.logger.info(f"Model loaded successfully on {_inference_engine.model_backend.device}")
    return _inference_engine


# Request/Response Models
class VerifyRequest(BaseModel):
    """Request for content verification"""
    query: str
    retrieved_content: str
    trace_id: Optional[str] = None
    shard_summary: Optional[str] = None
    epoch_summary: Optional[str] = None


class VerifyResponse(BaseModel):
    """Response from verification"""
    confidence: float
    reasoning: str
    supports_query: bool
    model_id: str
    inference_time_ms: float


class EnhanceQueryRequest(BaseModel):
    """Request for query enhancement"""
    query: str
    timestamp_context: Optional[str] = None
    trace_id: Optional[str] = None


class EnhanceQueryResponse(BaseModel):
    """Response from query enhancement"""
    original_query: str
    enhanced_query: str
    expansions: list[str]
    model_id: str
    inference_time_ms: float


class HallucinationCheckRequest(BaseModel):
    """Request for hallucination detection"""
    query: str
    system_response: str
    ground_truth: dict
    valid_terms: list[str]
    confidence: float
    trace_id: Optional[str] = None


class HallucinationCheckResponse(BaseModel):
    """Response from hallucination detection"""
    has_hallucination: bool
    hallucination_type: Optional[str] = None
    explanation: str
    severity: str
    flagged_content: list[str]
    model_id: str
    inference_time_ms: float


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - loads model on startup"""
    # Pre-load model
    get_inference_engine()
    logger.logger.info("SLM Service ready")
    yield
    logger.logger.info("SLM Service shutting down")


app = FastAPI(title="DPR-SLM-Service", lifespan=lifespan)


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "DPR-SLM-Service",
        "model": SLM_MODEL,
        "status": "ready",
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    engine = get_inference_engine()
    return {
        "status": "healthy",
        "model": engine.model_backend.get_model_id(),
        "device": engine.model_backend.device,
    }


@app.get("/readiness")
def readiness_check():
    """Readiness check - ensures model is loaded"""
    try:
        engine = get_inference_engine()
        return {"status": "ready", "model": engine.model_backend.get_model_id()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    """
    Verify if retrieved content answers the query.

    Uses SLM-based semantic verification (not token overlap).
    """
    engine = get_inference_engine()

    try:
        result = engine.verify_content(
            query=request.query,
            content=request.retrieved_content,
            shard_summary=request.shard_summary,
            epoch_summary=request.epoch_summary,
        )

        return VerifyResponse(
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            supports_query=result["supports_query"],
            model_id=result["model_id"],
            inference_time_ms=result["inference_time_ms"],
        )

    except Exception as e:
        logger.logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_verify")
def batch_verify(requests: list[VerifyRequest]):
    """Batch verification endpoint"""
    engine = get_inference_engine()
    results = []

    for req in requests:
        try:
            result = engine.verify_content(
                query=req.query,
                content=req.retrieved_content,
                shard_summary=req.shard_summary,
                epoch_summary=req.epoch_summary,
            )
            results.append(VerifyResponse(**result))
        except Exception as e:
            logger.logger.error(f"Batch verification error: {e}")
            results.append(None)

    return {"results": results, "count": len(results)}


@app.post("/enhance_query", response_model=EnhanceQueryResponse)
def enhance_query_endpoint(request: EnhanceQueryRequest):
    """
    Enhance query for better retrieval.

    Expands abbreviations, adds synonyms, clarifies terms.
    """
    engine = get_inference_engine()

    try:
        result = engine.enhance_query(
            query=request.query,
            timestamp_context=request.timestamp_context,
        )

        return EnhanceQueryResponse(
            original_query=result["original_query"],
            enhanced_query=result["enhanced_query"],
            expansions=result["expansions"],
            model_id=result["model_id"],
            inference_time_ms=result["inference_time_ms"],
        )

    except Exception as e:
        logger.logger.error(f"Query enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check_hallucination", response_model=HallucinationCheckResponse)
def check_hallucination_endpoint(request: HallucinationCheckRequest):
    """
    Check system response for hallucinations.

    Detects fabricated facts, invalid terms, and false certainty.
    """
    engine = get_inference_engine()

    try:
        result = engine.check_hallucination(
            query=request.query,
            system_response=request.system_response,
            ground_truth=request.ground_truth,
            valid_terms=request.valid_terms,
            confidence=request.confidence,
        )

        return HallucinationCheckResponse(
            has_hallucination=result["has_hallucination"],
            hallucination_type=result["hallucination_type"],
            explanation=result["explanation"],
            severity=result["severity"],
            flagged_content=result["flagged_content"],
            model_id=result["model_id"],
            inference_time_ms=result["inference_time_ms"],
        )

    except Exception as e:
        logger.logger.error(f"Hallucination check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_check_hallucination")
def batch_check_hallucination_endpoint(requests: list[HallucinationCheckRequest]):
    """Batch hallucination detection endpoint"""
    engine = get_inference_engine()
    results = []

    for req in requests:
        try:
            result = engine.check_hallucination(
                query=req.query,
                system_response=req.system_response,
                ground_truth=req.ground_truth,
                valid_terms=req.valid_terms,
                confidence=req.confidence,
            )
            results.append(HallucinationCheckResponse(**result))
        except Exception as e:
            logger.logger.error(f"Batch hallucination check error: {e}")
            results.append(None)

    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SLM_PORT)
