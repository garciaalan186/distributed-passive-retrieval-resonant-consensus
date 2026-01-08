"""
Hallucination Detection Module

Provides SLM-based and fallback hallucination detection for benchmark evaluation.
"""

import time
from typing import List, Dict, Any, Optional

import requests


class HallucinationDetector:
    """
    Detects hallucinations in model responses using SLM or rule-based fallback.
    """

    def __init__(
        self,
        slm_service_url: Optional[str] = None,
        slm_timeout: float = 30.0
    ):
        self.slm_service_url = slm_service_url
        self.slm_timeout = slm_timeout

    def detect_via_slm(
        self,
        query: str,
        ground_truth: Dict,
        system_response: str,
        glossary: Dict,
        confidence: float
    ) -> Dict:
        """
        Use SLM to determine if response contains hallucinations.

        Args:
            query: The original query
            ground_truth: Expected consensus/disputed claims from dataset
            system_response: What A* returned
            glossary: Valid phonotactic terms and their definitions
            confidence: How certain the system was (0-1)

        Returns:
            Dict with hallucination analysis results
        """
        try:
            valid_terms = self._extract_valid_terms(glossary)[:50]

            max_retries = 3
            base_delay = 1.0

            last_error = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.slm_service_url}/check_hallucination",
                        json={
                            "query": query,
                            "system_response": system_response,
                            "ground_truth": ground_truth,
                            "valid_terms": valid_terms,
                            "confidence": confidence
                        },
                        timeout=self.slm_timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "has_hallucination": result.get("has_hallucination", False),
                            "hallucination_type": result.get("hallucination_type"),
                            "explanation": result.get("explanation", ""),
                            "severity": result.get("severity", "none"),
                            "flagged_content": result.get("flagged_content", [])
                        }
                    elif response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"SLM service error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                    else:
                        print(f"SLM hallucination detection failed: HTTP {response.status_code}")
                        break

                except requests.Timeout:
                    last_error = "timeout"
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"SLM service timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    break
                except requests.ConnectionError as e:
                    last_error = f"connection error: {e}"
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"SLM connection error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    break

            print(f"SLM hallucination detection failed after {max_retries} attempts ({last_error})")
            return self.detect_fallback(system_response, glossary, confidence)

        except Exception as e:
            print(f"Error in SLM hallucination detection: {e}")
            return self.detect_fallback(system_response, glossary, confidence)

    def detect_fallback(
        self,
        response: str,
        glossary: Dict,
        confidence: float
    ) -> Dict:
        """
        Improved fallback when SLM is unavailable.
        More sophisticated than pure string matching.
        """
        if response is None:
            return {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "No response text to evaluate (raw semantic quadrant mode)",
                "severity": "none",
                "flagged_content": []
            }

        valid_terms = set()

        if 'physics' in glossary:
            valid_terms.update(glossary['physics'].get('particles', {}).keys())
            valid_terms.update(glossary['physics'].get('phenomena', {}).keys())
            valid_terms.update(glossary['physics'].get('constants', {}).keys())

        for domain_name, domain_data in glossary.get('domains', {}).items():
            valid_terms.update(domain_data.get('concepts', {}).keys())
            valid_terms.add(domain_name)
            for word in domain_name.split():
                valid_terms.add(word)

        common_words = {
            'No', 'Yes', 'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'With',
            'By', 'From', 'As', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being',
            'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Will', 'Would', 'Should',
            'Could', 'May', 'Might', 'Must', 'Can', 'This', 'That', 'These', 'Those',
            'Research', 'Study', 'Analysis', 'Results', 'Data', 'Findings', 'Progress',
            'Development', 'Breakthrough', 'Discovery', 'Experiment', 'Observation',
            'Review', 'Article', 'Team', 'New', 'Significant', 'Discussion',
            'Implications', 'Focusing', 'Through', 'Challenges', 'Remain',
            'Aligns', 'Driven', 'Conclusions', 'Drawn', 'Protocol', 'Predictions',
            'Status', 'Milestone', 'Progress', 'Achieved', 'Improved', 'Showing',
            'Improvement', 'Point', 'Area', 'Metrics', 'Domain'
        }

        words = response.split()
        suspicious_terms = []

        for word in words:
            clean_word = word.strip('.,!?;:()"\'')
            if not clean_word:
                continue

            if clean_word[0].isupper() and clean_word not in common_words:
                if clean_word not in valid_terms:
                    suspicious_terms.append(clean_word)

        is_uncertain = confidence < 0.7 or any(
            word in response.lower()
            for word in ['uncertain', 'mixed', 'perspectives', 'disputed', 'conflicting']
        )

        if suspicious_terms and (not is_uncertain or len(suspicious_terms) > 5):
            return {
                "has_hallucination": True,
                "hallucination_type": "invalid_term",
                "explanation": f"Found terms not in glossary: {', '.join(suspicious_terms[:5])}",
                "severity": "high" if not is_uncertain else "medium",
                "flagged_content": suspicious_terms
            }
        else:
            return {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "No significant hallucinations detected",
                "severity": "none",
                "flagged_content": []
            }

    def batch_detect(self, check_requests: List[Dict]) -> List[Dict]:
        """
        Batch hallucination detection - sends multiple requests in one HTTP call.

        Args:
            check_requests: List of dicts with keys:
                - query: str
                - system_response: str
                - ground_truth: dict
                - valid_terms: list[str]
                - confidence: float
                - trace_id: str (optional)

        Returns:
            List of hallucination detection results in same order as requests
        """
        if not check_requests:
            return []

        if not self.slm_service_url:
            print(f"Using fallback hallucination detection (No SLM URL configured)")
            return [
                self.detect_fallback(
                    req["system_response"],
                    {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                    req["confidence"]
                )
                for req in check_requests
            ]

        try:
            response = requests.post(
                f"{self.slm_service_url}/batch_check_hallucination",
                json=check_requests,
                timeout=self.slm_timeout * 2
            )

            if response.status_code == 200:
                batch_response = response.json()
                results = batch_response.get("results", [])

                if len(results) != len(check_requests):
                    print(f"Batch response count mismatch: got {len(results)}, expected {len(check_requests)}")
                    return [
                        self.detect_fallback(
                            req["system_response"],
                            {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                            req["confidence"]
                        )
                        for req in check_requests
                    ]

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "has_hallucination": result.get("has_hallucination", False),
                        "hallucination_type": result.get("hallucination_type"),
                        "explanation": result.get("explanation", ""),
                        "severity": result.get("severity", "none"),
                        "flagged_content": result.get("flagged_content", [])
                    })

                return formatted_results
            else:
                print(f"Batch hallucination detection failed: HTTP {response.status_code}")
                return [
                    self.detect_fallback(
                        req["system_response"],
                        {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                        req["confidence"]
                    )
                    for req in check_requests
                ]

        except Exception as e:
            print(f"Error in batch hallucination detection: {e}")
            return [
                self.detect_fallback(
                    req["system_response"],
                    {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                    req["confidence"]
                )
                for req in check_requests
            ]

    def _extract_valid_terms(self, glossary: Dict) -> List[str]:
        """Extract valid terms from glossary for hallucination detection."""
        valid_terms = []

        if 'physics' in glossary:
            valid_terms.extend(list(glossary.get('physics', {}).get('particles', {}).keys()))
            valid_terms.extend(list(glossary.get('physics', {}).get('phenomena', {}).keys()))

        for domain_name, domain_data in glossary.get('domains', {}).items():
            valid_terms.extend(list(domain_data.get('concepts', {}).keys()))
            valid_terms.extend(domain_name.split())

        return valid_terms
