"""
Pure evaluation service with no I/O dependencies.

This service contains only pure functions that evaluate correctness and
superposition awareness. All I/O (HTTP calls, file operations) must be
handled by the caller.
"""

from typing import List, Dict, Optional, Any
from benchmark.domain.value_objects import CorrectnessResult


class EvaluationService:
    """
    Pure evaluation logic for benchmark responses.

    This service is stateless and performs no I/O. All methods are pure
    functions that take inputs and return value objects.
    """

    @staticmethod
    def evaluate_correctness(
        response: str,
        expected_entities: List[str],
        recall_threshold: float = 0.5,
        min_response_length: int = 20
    ) -> CorrectnessResult:
        """
        Evaluate if response contains expected entities.

        This is the core correctness evaluation for simple (non-superposition)
        responses. For superposition-aware evaluation, use
        evaluate_superposition_correctness().

        Args:
            response: The system's response text
            expected_entities: List of entities (e.g., phonotactic terms) expected in response
            recall_threshold: Minimum fraction of entities that must be present (0.0 to 1.0)
            min_response_length: Minimum response length to be considered valid

        Returns:
            CorrectnessResult with evaluation details

        Implementation Notes:
            - Uses simple substring matching (entity in response)
            - This is intentionally simple to avoid confounding variables in benchmarks
            - More sophisticated semantic matching would introduce external dependencies
            - The recall_threshold is a tunable parameter for benchmark sensitivity
        """
        if not response or len(response) < min_response_length:
            return CorrectnessResult(
                is_correct=False,
                correctness_score=0.0,
                matched_entities=[],
                expected_entities=expected_entities,
                explanation=f"Response too short (min {min_response_length} chars)"
            )

        if not expected_entities:
            # No entities to check - ambiguous case
            # We consider this correct but with low score
            return CorrectnessResult(
                is_correct=True,
                correctness_score=0.5,
                matched_entities=[],
                expected_entities=[],
                explanation="No entities to evaluate (query has no expected entities)"
            )

        # Count entity matches
        matched = [entity for entity in expected_entities if entity in response]
        recall = len(matched) / len(expected_entities)

        is_correct = recall >= recall_threshold

        return CorrectnessResult(
            is_correct=is_correct,
            correctness_score=recall,
            matched_entities=matched,
            expected_entities=expected_entities,
            explanation=f"Matched {len(matched)}/{len(expected_entities)} entities (recall={recall:.2f})"
        )

    @staticmethod
    def evaluate_superposition_correctness(
        response: str,
        expected_entities: List[str],
        recall_threshold: float = 0.5,
        min_response_length: int = 20,
        superposition_data: Optional[Dict[str, Any]] = None
    ) -> CorrectnessResult:
        """
        Evaluate correctness for superposition-aware responses.

        This handles responses that may present multiple alternatives/perspectives.
        The response is correct if ANY option contains the expected entities.

        If superposition_data (raw JSON) is provided, it evaluates against the
        structured quadrants (consensus, polar, negative_consensus) instead of
        parsing text.

        Args:
            response: The system's response text (may contain multiple options)
            expected_entities: List of entities expected to appear
            recall_threshold: Minimum fraction of entities per option
            min_response_length: Minimum response length
            superposition_data: Optional raw superposition dictionary

        Returns:
            CorrectnessResult indicating if correct answer is present in any option
        """
        if not expected_entities:
            # No entities to check
            return CorrectnessResult(
                is_correct=True,
                correctness_score=0.5,
                matched_entities=[],
                expected_entities=[],
                explanation="No entities to evaluate (query has no expected entities)"
            )

        # Extract potential options
        options = []

        if superposition_data:
            # Handle both old and new superposition schemas:
            # Old schema: {"consensus": [...], "polar": [...], "negative_consensus": [...]}
            #   where items are dicts with 'content' field
            # New schema: {"consensus_facts": [...], "perspectival_claims": [...]}
            #   where items are strings directly

            # Try new schema first (consensus_facts, perspectival_claims)
            for key in ["consensus_facts", "perspectival_claims"]:
                if key in superposition_data and isinstance(superposition_data[key], list):
                    for item in superposition_data[key]:
                        if isinstance(item, str):
                            options.append(item)
                        elif isinstance(item, dict) and "content" in item:
                            options.append(item["content"])

            # Fallback to old schema (consensus, polar, negative_consensus)
            if not options:
                for key in ["consensus", "polar", "negative_consensus"]:
                    if key in superposition_data and isinstance(superposition_data[key], list):
                        for artifact in superposition_data[key]:
                            # Artifact is a dict with 'content'
                            if isinstance(artifact, dict) and "content" in artifact:
                                options.append(artifact["content"])
        else:
            # Parse text response
            options = EvaluationService._extract_options(response)

        # Check if any option contains the expected entities
        best_matched = []
        best_recall = 0.0

        for option in options:
            if not option:
                continue
            matched = [entity for entity in expected_entities if entity in option]
            recall = len(matched) / len(expected_entities)

            if recall > best_recall:
                best_recall = recall
                best_matched = matched

        is_correct = best_recall >= recall_threshold

        if is_correct:
            explanation = f"Found correct answer in one option (matched {len(best_matched)}/{len(expected_entities)} entities)"
        else:
            explanation = f"No option had sufficient entities (best: {len(best_matched)}/{len(expected_entities)})"

        return CorrectnessResult(
            is_correct=is_correct,
            correctness_score=best_recall,
            matched_entities=best_matched,
            expected_entities=expected_entities,
            explanation=explanation
        )

    @staticmethod
    def evaluate_superposition_awareness(
        response: str,
        min_options: int = 2,
        min_option_length: int = 10
    ) -> Dict[str, any]:
        """
        Check if response demonstrates superposition awareness.

        Superposition awareness means the system presents multiple perspectives
        or alternatives rather than a single definitive answer.

        Args:
            response: The system's response text
            min_options: Minimum number of distinct options to be considered superposition-aware
            min_option_length: Minimum length for an option to be considered valid

        Returns:
            Dict with:
                - has_multiple_perspectives: bool
                - option_count: int
                - options: List[str]
                - explanation: str

        Implementation Notes:
            - Uses _extract_options() to parse response structure
            - Filters options by minimum length to avoid counting noise
            - This is a structural check, not semantic
            - True superposition would require semantic understanding
        """
        options = EvaluationService._extract_options(response, min_length=min_option_length)

        has_multiple = len(options) >= min_options

        return {
            "has_multiple_perspectives": has_multiple,
            "option_count": len(options),
            "options": options,
            "explanation": (
                f"Response has {len(options)} distinct options (min {min_options} required)"
                if has_multiple
                else f"Response appears to present a single view ({len(options)} options detected)"
            )
        }

    @staticmethod
    def _extract_options(response: str, min_length: int = 10) -> List[str]:
        """
        Extract distinct options/alternatives from a response.

        This is a heuristic parser that looks for common structural patterns
        indicating multiple alternatives:
        - Bullet points (lines starting with "- ")
        - Numbered lists
        - Newline-separated blocks

        Args:
            response: Response text to parse
            min_length: Minimum length for a valid option

        Returns:
            List of option strings

        Implementation Notes:
            - This is intentionally simple to avoid introducing confounding variables
            - More sophisticated parsing (e.g., using NLP) would create dependencies
            - The heuristics match common LLM response patterns
            - Fallback: if no structure detected, return entire response as single option
        """
        if not response:
            return []

        # Try to split by bullets or dashes
        # Pattern: "- Option 1\n- Option 2" or "* Option 1\n* Option 2"
        options = []

        # Replace bullet markers with newlines to normalize
        normalized = response.replace("- ", "\n").replace("* ", "\n")

        # Split on newlines and filter
        parts = normalized.split("\n")
        options = [part.strip() for part in parts if len(part.strip()) >= min_length]

        # If we found multiple options, return them
        if len(options) >= 2:
            return options

        # Otherwise, check for numbered lists
        # Pattern: "1. Option\n2. Option"
        # Simple check: if response contains "1." and "2.", split on numbers
        if "1." in response and "2." in response:
            # Split on number patterns
            import re
            parts = re.split(r'\d+\.\s+', response)
            options = [part.strip() for part in parts if len(part.strip()) >= min_length]

            if len(options) >= 2:
                return options

        # Fallback: return entire response as single option
        return [response] if len(response) >= min_length else []

    @staticmethod
    def extract_entities_from_question(question: str) -> List[str]:
        """
        Extract entity candidates from a question.

        For phonotactic benchmarks, entities are typically capitalized terms
        that represent synthetic concepts (e.g., "Quarkonium", "Chronokinetics").

        Args:
            question: The question text

        Returns:
            List of capitalized terms (potential entities)

        Implementation Notes:
            - Simple heuristic: extract capitalized words
            - This works for phonotactic terms which are always capitalized
            - Real-world applications would use NER, but that introduces dependencies
            - The simplicity here ensures benchmark purity
        """
        if not question:
            return []

        words = question.split()
        # Extract words that start with uppercase and are not common words
        common_words = {
            'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'With',
            'By', 'From', 'As', 'Is', 'Are', 'Was', 'Were', 'What', 'When',
            'Where', 'Why', 'How', 'Which', 'Who', 'Did', 'Does', 'Do'
        }

        entities = []
        for word in words:
            # Clean word (remove punctuation)
            clean_word = word.strip('.,!?;:()"\'')
            if not clean_word:
                continue

            # Check if capitalized and not a common word
            if clean_word and clean_word[0].isupper() and clean_word not in common_words:
                entities.append(clean_word)

        return entities
