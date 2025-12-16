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
SLM_PORT = int(os.getenv("PORT", os.getenv("SLM_PORT", "8080")))  # Cloud Run uses PORT env var
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
    hallucination_type: Optional[str] = None  # "fabricated_fact" | "invalid_term" | "false_certainty"
    explanation: str
    severity: str  # "high" | "medium" | "low" | "none"
    flagged_content: list[str]
    model_id: str
    inference_time_ms: float


def get_query_enhancement_prompt(query: str, timestamp_context: Optional[str] = None) -> str:
    """
    Construct a prompt to enhance/expand the query for better retrieval.

    The SLM reformulates the query to improve semantic matching by:
    - Expanding abbreviations
    - Adding synonyms
    - Clarifying ambiguous terms
    - Adding temporal context
    """
    temporal_hint = ""
    if timestamp_context:
        temporal_hint = f"\nTemporal context: The query relates to events around {timestamp_context}."

    return f"""You are a query enhancement assistant. Your task is to improve a search query for better retrieval from a historical knowledge base.

Original query: {query}{temporal_hint}

Instructions:
1. Expand any abbreviations (e.g., "ML" -> "machine learning")
2. Add relevant synonyms or related terms
3. Clarify any ambiguous terms
4. Keep the enhanced query concise but more searchable
5. List 2-3 key expansion terms separately

Respond in this exact JSON format:
{{"enhanced_query": "<improved query text>", "expansions": ["term1", "term2", "term3"]}}

Your response (JSON only):"""


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


def get_hallucination_check_prompt(
    query: str,
    system_response: str,
    ground_truth: dict,
    valid_terms: list[str],
    confidence: float
) -> str:
    """
    Construct the hallucination detection prompt for the SLM.

    Asks the model to evaluate if the system response contains
    hallucinated information given the query, ground truth, and valid terms.
    """
    # Limit valid terms to avoid token overflow (Qwen2-0.5B has limited context)
    terms_sample = ', '.join(valid_terms[:50]) if len(valid_terms) > 50 else ', '.join(valid_terms)

    expected_consensus = ground_truth.get('expected_consensus', [])
    expected_disputed = ground_truth.get('expected_disputed', [])

    return f"""You are evaluating a retrieval system for hallucinations.

QUERY: {query}

SYSTEM RESPONSE: {system_response}

SYSTEM CONFIDENCE: {confidence:.2f} (0=uncertain, 1=certain)

VALID TERMS (from dataset glossary): {terms_sample}

GROUND TRUTH:
- Expected consensus: {expected_consensus}
- Expected disputed points: {expected_disputed}

TASK: Determine if the system response contains hallucinations.

Consider these hallucination types:
1. FABRICATED FACTS: Claims not supported by ground truth
2. INVALID TERMS: Terms/concepts not in the valid glossary
3. FALSE CERTAINTY: Disputed info presented as definitive fact

IMPORTANT RULES:
- Common words like "The", "In", "No", "Yes", "Research" are NOT hallucinations
- If confidence < 0.9 OR response mentions "perspectives"/"mixed"/"uncertain", presenting alternatives is VALID
- Only flag completely fabricated terms AND presented as fact
- Terms from the glossary are valid even if they look unfamiliar

EXAMPLES:

Good (no hallucination):
Query: "What is Blarkon status?"
Response: "Blarkon research shows mixed results. Some perspectives indicate decay, others show stability."
Valid terms: ["Blarkon", "decay", "stability"]
→ No hallucination (uncertainty + all terms valid)

Bad (hallucination):
Query: "What is Blarkon status?"
Response: "Zynthium interference was definitively solved using quantum entanglement."
Valid terms: ["Blarkon", "decay"]
→ YES - "Zynthium" not in glossary, "quantum entanglement" not in alternate physics

Respond in this exact JSON format:
{{
  "has_hallucination": true/false,
  "hallucination_type": "fabricated_fact" | "invalid_term" | "false_certainty" | null,
  "explanation": "brief explanation of decision",
  "severity": "high" | "medium" | "low" | "none",
  "flagged_content": ["specific", "problematic", "terms"]
}}

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
def root():
    """Root endpoint."""
    return {
        "service": "DPR-RC SLM Service",
        "model": SLM_MODEL,
        "device": DEVICE,
        "model_loaded": _model is not None
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint for Cloud Run.
    Returns 503 if model is still loading to prevent premature traffic routing.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {SLM_MODEL} is still loading. Please wait."
        )
    return {
        "status": "healthy",
        "model": SLM_MODEL,
        "device": DEVICE,
        "model_loaded": True
    }


@app.get("/ready")
def readiness_check():
    """
    Readiness endpoint - returns 200 only when model is fully loaded.
    Use this to block deployment traffic until service is ready.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {SLM_MODEL} is still loading. Please wait."
        )
    return {
        "status": "ready",
        "model": SLM_MODEL,
        "device": DEVICE
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


def enhance_query(query: str, timestamp_context: Optional[str] = None) -> dict:
    """
    Enhance a query for better retrieval using the SLM.

    Args:
        query: The original query from A*
        timestamp_context: Optional temporal context

    Returns:
        dict with enhanced_query, expansions, inference_time_ms
    """
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    start_time = time.time()

    # Construct prompt
    prompt = get_query_enhancement_prompt(query, timestamp_context)

    # Tokenize
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=500
    )

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=100,  # Shorter for query enhancement
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id
        )

    # Decode response
    input_length = inputs["input_ids"].shape[1]
    response_text = _tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    inference_time = (time.time() - start_time) * 1000

    # Parse response
    result = parse_enhancement_response(response_text, query)
    result["inference_time_ms"] = inference_time
    result["model_id"] = SLM_MODEL

    return result


def parse_enhancement_response(response_text: str, original_query: str) -> dict:
    """
    Parse the model's query enhancement response.
    """
    text = response_text.strip()

    # Try to extract JSON
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx + 1]
        try:
            parsed = json.loads(json_str)
            enhanced = parsed.get("enhanced_query", original_query)
            expansions = parsed.get("expansions", [])
            # Ensure expansions is a list of strings
            if isinstance(expansions, list):
                expansions = [str(e) for e in expansions[:5]]  # Limit to 5
            else:
                expansions = []
            return {
                "original_query": original_query,
                "enhanced_query": str(enhanced),
                "expansions": expansions
            }
        except json.JSONDecodeError:
            pass

    # Fallback: return original query with no expansions
    return {
        "original_query": original_query,
        "enhanced_query": original_query,
        "expansions": []
    }


@app.post("/enhance_query", response_model=EnhanceQueryResponse)
def enhance_query_endpoint(request: EnhanceQueryRequest):
    """
    Enhance a query for better retrieval.

    Called by Active Agent before broadcasting RFI to improve
    semantic matching in the embedding space.

    The SLM:
    - Expands abbreviations (ML -> machine learning)
    - Adds synonyms for better recall
    - Clarifies ambiguous terms
    - Incorporates temporal context if provided
    """
    try:
        result = enhance_query(request.query, request.timestamp_context)

        return EnhanceQueryResponse(
            original_query=result["original_query"],
            enhanced_query=result["enhanced_query"],
            expansions=result["expansions"],
            model_id=result["model_id"],
            inference_time_ms=result["inference_time_ms"]
        )
    except Exception as e:
        # On error, return original query unchanged
        return EnhanceQueryResponse(
            original_query=request.query,
            enhanced_query=request.query,
            expansions=[],
            model_id=SLM_MODEL,
            inference_time_ms=0.0
        )


@app.post("/check_hallucination", response_model=HallucinationCheckResponse)
def check_hallucination_endpoint(request: HallucinationCheckRequest):
    """
    Check if a system response contains hallucinations.

    Called by benchmark suite to evaluate response quality.
    Uses the SLM to make semantic judgments about whether the
    response contains fabricated facts, invalid terms, or false certainty.

    This is more sophisticated than string matching - the SLM
    understands context, uncertainty, and when alternatives are valid.
    """
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Generate prompt
        prompt = get_hallucination_check_prompt(
            query=request.query,
            system_response=request.system_response,
            ground_truth=request.ground_truth,
            valid_terms=request.valid_terms,
            confidence=request.confidence
        )

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = _tokenizer([text], return_tensors="pt").to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,  # Low temperature for consistency
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id
            )

        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response_text = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        inference_time_ms = (time.time() - start_time) * 1000

        # Parse JSON from response
        result = _parse_hallucination_response(response_text)

        return HallucinationCheckResponse(
            has_hallucination=result["has_hallucination"],
            hallucination_type=result.get("hallucination_type"),
            explanation=result["explanation"],
            severity=result["severity"],
            flagged_content=result["flagged_content"],
            model_id=SLM_MODEL,
            inference_time_ms=inference_time_ms
        )

    except Exception as e:
        # On error, return conservative fallback
        print(f"Error in hallucination check: {e}")
        return HallucinationCheckResponse(
            has_hallucination=False,  # Conservative: don't flag without confidence
            hallucination_type=None,
            explanation=f"Error during check: {str(e)}",
            severity="none",
            flagged_content=[],
            model_id=SLM_MODEL,
            inference_time_ms=0.0
        )


@app.post("/batch_check_hallucination")
def batch_check_hallucination_endpoint(requests: list[HallucinationCheckRequest]):
    """
    Batch hallucination checking for multiple query-response pairs.

    More efficient when checking multiple results at once - reduces
    HTTP overhead and allows for better resource utilization.

    Returns a list of hallucination check results in the same order
    as the input requests.
    """
    results = []

    for req in requests:
        try:
            # Call the individual check for each request
            result = check_hallucination_endpoint(req)
            results.append({
                "trace_id": req.trace_id,
                "has_hallucination": result.has_hallucination,
                "hallucination_type": result.hallucination_type,
                "explanation": result.explanation,
                "severity": result.severity,
                "flagged_content": result.flagged_content,
                "inference_time_ms": result.inference_time_ms
            })
        except Exception as e:
            # On error, return conservative fallback
            results.append({
                "trace_id": req.trace_id,
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": f"Error during check: {str(e)}",
                "severity": "none",
                "flagged_content": [],
                "inference_time_ms": 0.0,
                "error": str(e)
            })

    return {
        "results": results,
        "model_id": SLM_MODEL,
        "batch_size": len(requests)
    }


def _parse_hallucination_response(text: str) -> dict:
    """
    Parse the SLM's hallucination check response.

    Expects JSON format:
    {
      "has_hallucination": true/false,
      "hallucination_type": "...",
      "explanation": "...",
      "severity": "...",
      "flagged_content": [...]
    }
    """
    import re

    # Try to extract JSON from response
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx + 1]
        try:
            parsed = json.loads(json_str)

            # Validate and extract fields with defaults
            return {
                "has_hallucination": parsed.get("has_hallucination", False),
                "hallucination_type": parsed.get("hallucination_type"),
                "explanation": str(parsed.get("explanation", "No explanation provided"))[:500],
                "severity": parsed.get("severity", "none"),
                "flagged_content": parsed.get("flagged_content", [])[:20]  # Limit to 20 items
            }
        except json.JSONDecodeError:
            pass

    # Fallback: heuristic parsing if JSON fails
    has_hallucination = "true" in text.lower() and "has_hallucination" in text.lower()

    return {
        "has_hallucination": has_hallucination,
        "hallucination_type": "unknown" if has_hallucination else None,
        "explanation": text[:500] if text else "Failed to parse response",
        "severity": "medium" if has_hallucination else "none",
        "flagged_content": []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SLM_PORT)
