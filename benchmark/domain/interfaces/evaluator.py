"""
Evaluator Interface

This interface defines how benchmark results are evaluated against ground truth.
Evaluators MUST be pure functions to ensure reproducible benchmark results.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IEvaluator(ABC):
    """
    Interface for evaluating query results against ground truth.

    CRITICAL DESIGN PRINCIPLES:

    1. **Pure Evaluation**: Evaluators MUST be pure functions with no side effects.
       Same inputs always produce same outputs. This is critical for:
       - Reproducible benchmarks
       - Valid statistical analysis
       - Scientific rigor

    2. **No I/O**: Evaluators should not perform I/O operations (file access, HTTP calls).
       All necessary data should be passed as parameters.

    3. **Domain-Specific Logic**: Different evaluators implement different evaluation strategies:
       - SuperpositionAwareEvaluator: For DPR-RC (accepts multiple valid answers)
       - StandardRAGEvaluator: For traditional RAG (single ground truth)
       - CustomEvaluator: For domain-specific evaluation criteria

    Implementations:
    - SuperpositionAwareEvaluator: DPR-RC specific evaluation with superposition support
    - StandardRAGEvaluator: Traditional RAG evaluation (exact match, semantic similarity)
    - MockEvaluator: For testing
    """

    @abstractmethod
    def evaluate(
        self,
        expected: Dict[str, Any],
        actual: Any  # QueryExecutionResult from query_executor
    ) -> Dict[str, Any]:
        """
        Evaluate a single query result against ground truth.

        Args:
            expected: Ground truth data from the benchmark dataset
                     (structure depends on evaluator implementation)
            actual: QueryExecutionResult from query execution

        Returns:
            Dictionary containing evaluation results. MUST include:
            - 'is_correct': bool - whether the answer is correct
            - 'correctness_score': float - score from 0.0 to 1.0

            MAY include additional fields:
            - 'explanation': str - human-readable explanation
            - 'matched_entities': list - entities matched in response
            - 'superposition_aware': bool - whether multiple perspectives presented

        Note:
            This method MUST be deterministic. Same inputs always produce same outputs.
        """
        pass

    @abstractmethod
    def aggregate(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple evaluation results into summary metrics.

        Args:
            results: List of evaluation results from evaluate()

        Returns:
            Dictionary containing aggregated metrics. MUST include:
            - 'accuracy': float - overall accuracy (0.0 to 1.0)
            - 'total_evaluated': int - number of queries evaluated

            MAY include additional fields:
            - 'mean_correctness_score': float
            - 'median_correctness_score': float
            - 'std_correctness_score': float
            - 'correct_count': int
            - 'incorrect_count': int

        Note:
            This method MUST also be deterministic and handle edge cases
            (empty list, single result, etc.) gracefully.
        """
        pass
