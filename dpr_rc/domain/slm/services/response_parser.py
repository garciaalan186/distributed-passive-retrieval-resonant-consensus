"""
Domain Service: Response Parser

Parses SLM responses and extracts structured data.
Handles both JSON and fallback heuristics.
"""

import json
import re
from typing import Dict, Optional


class ResponseParser:
    """
    Domain service for parsing SLM responses.

    Handles JSON extraction with fallback to heuristics when JSON parsing fails.
    """

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse verification response from SLM.

        Expected format: {"confidence": 0.0-1.0, "supports_query": true/false, "reasoning": "..."}

        Returns:
            Dict with confidence, supports_query, reasoning
        """
        # Try JSON parsing first
        try:
            data = self._extract_json(response_text)
            if data and "confidence" in data:
                return {
                    "confidence": float(data.get("confidence", 0.5)),
                    "supports_query": bool(data.get("supports_query", False)),
                    "reasoning": str(data.get("reasoning", "")),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback heuristics
        return self._fallback_verification_parse(response_text)

    def parse_enhancement_response(
        self, response_text: str, original_query: str
    ) -> Dict:
        """
        Parse query enhancement response from SLM.

        Expected format: {"enhanced_query": "...", "expansions": ["term1", "term2"]}

        Returns:
            Dict with enhanced_query, expansions
        """
        # Try JSON parsing
        try:
            data = self._extract_json(response_text)
            if data and "enhanced_query" in data:
                return {
                    "enhanced_query": str(data.get("enhanced_query", original_query)),
                    "expansions": list(data.get("expansions", [])),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: use original query
        return {
            "enhanced_query": original_query,
            "expansions": [],
        }

    def parse_hallucination_response(self, response_text: str) -> Dict:
        """
        Parse hallucination detection response from SLM.

        Expected format:
        {
          "has_hallucination": true/false,
          "hallucination_type": "...",
          "severity": "high/medium/low/none",
          "explanation": "...",
          "flagged_content": [...]
        }

        Returns:
            Dict with hallucination detection results
        """
        # Try JSON parsing
        try:
            data = self._extract_json(response_text)
            if data and "has_hallucination" in data:
                return {
                    "has_hallucination": bool(data.get("has_hallucination", False)),
                    "hallucination_type": data.get("hallucination_type"),
                    "severity": str(data.get("severity", "none")),
                    "explanation": str(data.get("explanation", "")),
                    "flagged_content": list(data.get("flagged_content", [])),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: no hallucination detected
        return {
            "has_hallucination": False,
            "hallucination_type": None,
            "severity": "none",
            "explanation": "Failed to parse SLM response",
            "flagged_content": [],
        }

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text, handling common formatting issues.

        Tries multiple strategies to find and parse JSON.
        """
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON between curly braces
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON after specific markers
        markers = ["JSON Response:", "Your response:", "Response:"]
        for marker in markers:
            if marker in text:
                after_marker = text.split(marker, 1)[1].strip()
                try:
                    return json.loads(after_marker)
                except json.JSONDecodeError:
                    pass

        return None

    def _fallback_verification_parse(self, text: str) -> Dict:
        """
        Fallback heuristics when JSON parsing fails.

        Uses keyword detection and sentiment analysis.
        """
        text_lower = text.lower()

        # Check for positive indicators
        positive_words = ["yes", "supports", "relevant", "answers", "correct"]
        negative_words = ["no", "doesn't", "irrelevant", "incorrect", "unrelated"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            confidence = min(0.7, 0.5 + (positive_count * 0.1))
            supports_query = True
        else:
            confidence = max(0.3, 0.5 - (negative_count * 0.1))
            supports_query = False

        return {
            "confidence": confidence,
            "supports_query": supports_query,
            "reasoning": "Parsed using fallback heuristics",
        }
