"""
SLM (Small Language Model) Service for DPR-RC System

Per Architecture Spec Section 2.1 - L2 Verification:
"Each Passive Agent uses an SLM to verify whether retrieved content
answers the original query from A*."

This service hosts Qwen2-0.5B-Instruct and provides verification endpoints
for Passive Agents to call during the consensus protocol.

Architecture:
- Single model instance shared across all Passive Workers
- Loaded once at startup, serves many requests
- Returns structured verification judgments
"""

import os
import time
import json
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Configuration
SLM_MODEL = os.getenv("SLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
SLM_PORT = int(os.getenv("SLM_PORT", 8081))
MAX_NEW_TOKENS = int(os.getenv("SLM_MAX_TOKENS", 150))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer
_model = None
_tokenizer = None


class VerifyRequest(BaseModel):
    """Request for content verification"""
    query: str
    retrieved_content: str
    trace_id: Optional[str] = None


class VerifyResponse(BaseModel):
    """Response from verification"""
    confidence: float
    reasoning: str
    supports_query: bool
    model_id: str
    inference_time_ms: float


def get_verification_prompt(query: str, content: str) -> str:
    """
    Construct the verification prompt for the SLM.

    The prompt asks the model to judge whether the retrieved content
    answers or supports the original query.
    """
    return f"""You are a verification assistant. Your task is to judge whether the retrieved content answers or supports the given query.

Query: {query}

Retrieved Content: {content}

Instructions:
1. Analyze if the content directly answers or provides relevant information for the query
2. Provide a confidence score from 0.0 to 1.0
3. Explain your reasoning briefly

Respond in this exact JSON format:
{{"confidence": <0.0-1.0>, "supports_query": <true/false>, "reasoning": "<brief explanation>"}}

Your response (JSON only):"""


def load_model():
    """Load the SLM model and tokenizer."""
    global _model, _tokenizer

    print(f"Loading SLM model: {SLM_MODEL}")
    print(f"Device: {DEVICE}")
    start_time = time.time()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            SLM_MODEL,
            trust_remote_code=True
        )

        # Set padding token if not set
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )

        if DEVICE == "cpu":
            _model = _model.to("cpu")

        _model.eval()

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        print(f"Model memory: {_model.get_memory_footprint() / 1e9:.2f} GB")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def parse_model_response(response_text: str) -> dict:
    """
    Parse the model's JSON response, handling common formatting issues.
    """
    # Try to extract JSON from the response
    text = response_text.strip()

    # Find JSON object in response
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx + 1]
        try:
            parsed = json.loads(json_str)
            return {
                "confidence": float(parsed.get("confidence", 0.5)),
                "supports_query": bool(parsed.get("supports_query", False)),
                "reasoning": str(parsed.get("reasoning", "No reasoning provided"))
            }
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract values heuristically
    confidence = 0.5
    supports = False
    reasoning = "Could not parse model response"

    text_lower = text.lower()

    # Look for confidence indicators
    if "high confidence" in text_lower or "strongly supports" in text_lower:
        confidence = 0.85
        supports = True
    elif "moderate" in text_lower or "partially" in text_lower:
        confidence = 0.6
        supports = True
    elif "low confidence" in text_lower or "does not support" in text_lower:
        confidence = 0.3
        supports = False
    elif "yes" in text_lower or "supports" in text_lower or "answers" in text_lower:
        confidence = 0.7
        supports = True
    elif "no" in text_lower or "does not" in text_lower:
        confidence = 0.3
        supports = False

    # Extract any reasoning text
    if len(text) > 10:
        reasoning = text[:200]

    return {
        "confidence": confidence,
        "supports_query": supports,
        "reasoning": reasoning
    }


def verify_content(query: str, content: str) -> dict:
    """
    Run verification inference using the SLM.

    Args:
        query: The original query from A*
        content: The retrieved content to verify

    Returns:
        dict with confidence, supports_query, reasoning, inference_time_ms
    """
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    start_time = time.time()

    # Construct prompt
    prompt = get_verification_prompt(query, content[:1000])  # Limit content length

    # Tokenize
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500
    )

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Deterministic for consistency
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id
        )

    # Decode response (only the new tokens)
    input_length = inputs["input_ids"].shape[1]
    response_text = _tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Parse response
    result = parse_model_response(response_text)
    result["inference_time_ms"] = inference_time
    result["model_id"] = SLM_MODEL

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    load_model()
    print(f"SLM Service ready on port {SLM_PORT}")
    yield
    print("SLM Service shutting down")


app = FastAPI(
    title="DPR-RC SLM Service",
    description="Small Language Model service for L2 verification",
    lifespan=lifespan
)


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": SLM_MODEL,
        "device": DEVICE,
        "model_loaded": _model is not None
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    """
    Verify if retrieved content answers/supports the query.

    This is the main endpoint called by Passive Agents during
    L2 verification in the consensus protocol.
    """
    try:
        result = verify_content(request.query, request.retrieved_content)

        return VerifyResponse(
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            supports_query=result["supports_query"],
            model_id=result["model_id"],
            inference_time_ms=result["inference_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_verify")
def batch_verify(requests: list[VerifyRequest]):
    """
    Batch verification for multiple content pieces.

    More efficient when verifying multiple retrievals at once.
    """
    results = []
    for req in requests:
        try:
            result = verify_content(req.query, req.retrieved_content)
            results.append({
                "trace_id": req.trace_id,
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "supports_query": result["supports_query"],
                "inference_time_ms": result["inference_time_ms"]
            })
        except Exception as e:
            results.append({
                "trace_id": req.trace_id,
                "error": str(e),
                "confidence": 0.0,
                "supports_query": False
            })

    return {"results": results, "model_id": SLM_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SLM_PORT)
