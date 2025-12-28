"""
Domain Service: Prompt Builder

Constructs prompts for SLM inference.
Pure business logic with no infrastructure dependencies.
"""

from typing import Optional


class PromptBuilder:
    """
    Domain service for building SLM prompts.

    Handles prompt engineering for verification, enhancement, and hallucination detection.
    """

    def build_query_enhancement_prompt(
        self, query: str, timestamp_context: Optional[str] = None
    ) -> str:
        """
        Construct prompt for query enhancement.

        Expands abbreviations, adds synonyms, clarifies ambiguous terms.
        """
        prompt = f"""You are a query expansion assistant. Your task is to expand and clarify the user's query to improve information retrieval.

Original Query: "{query}"
"""

        if timestamp_context:
            prompt += f"Temporal Context: {timestamp_context}\n"

        prompt += """
Instructions:
1. Expand abbreviations (ML → machine learning)
2. Add relevant synonyms for better recall
3. Clarify ambiguous terms
4. Keep the core intent unchanged

Return ONLY the enhanced query, no explanations.

Enhanced Query:"""

        return prompt

    def build_verification_prompt(
        self,
        query: str,
        content: str,
        shard_summary: Optional[str] = None,
        epoch_summary: Optional[str] = None,
    ) -> str:
        """
        Construct prompt for content verification with hierarchical foveated context.

        Per Mathematical Model Section 5.2, Equation 9:
        C(r_p) = V(q, context_p) · (1 / (1 + i))
        """
        prompt = f"""You are a semantic verification judge. Evaluate whether the retrieved content answers the query.

Query: "{query}"

Retrieved Content:
{content}
"""

        # Add hierarchical foveated context
        if shard_summary:
            prompt += f"""
Shard Context (L1): {shard_summary}
"""

        if epoch_summary:
            prompt += f"""
Epoch Context (L2): {epoch_summary}
"""

        prompt += """
Instructions:
1. Determine if the content provides a relevant answer to the query
2. Consider semantic meaning, not just keyword overlap
3. Account for hierarchical context if provided
4. Return your assessment in JSON format

Required JSON format:
{
  "supports_query": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

JSON Response:"""

        return prompt

    def build_hallucination_check_prompt(
        self,
        query: str,
        system_response: str,
        ground_truth: dict,
        valid_terms: list[str],
        confidence: float,
    ) -> str:
        """
        Construct prompt for hallucination detection.

        Checks for fabricated facts, invalid terms, and false certainty.
        """
        prompt = f"""You are a hallucination detector. Check if the system response contains hallucinations.

Original Query: "{query}"

System Response:
{system_response}

Ground Truth Reference:
{ground_truth}

Valid Terms from Dataset: {', '.join(valid_terms[:20])}
System Confidence: {confidence:.2f}

Hallucination Types to Check:
1. **Fabricated Fact**: Response contains information not in ground truth
2. **Invalid Term**: Response uses terms not in the valid dataset vocabulary
3. **False Certainty**: High confidence (>0.9) with incorrect information

Instructions:
Analyze the system response and determine if it contains hallucinations.

Required JSON format:
{{
  "has_hallucination": true/false,
  "hallucination_type": "fabricated_fact" | "invalid_term" | "false_certainty" | null,
  "severity": "high" | "medium" | "low" | "none",
  "explanation": "detailed explanation",
  "flagged_content": ["specific phrases that are hallucinated"]
}}

JSON Response:"""

        return prompt
