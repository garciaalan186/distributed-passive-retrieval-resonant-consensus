"""
Domain Service: Response Synthesizer

Collapses superposition to final response with confidence scoring.
"""

from ..entities import (
    ConsensusResult,
    ConsensusResult,
    ResonanceMatrix,
    CollapsedResponse,
    ConsensusTier,
)


class ResponseSynthesizer:
    """
    Domain service for collapsing superposition to final answer.

    Maps consensus tiers to confidence levels and selects best response.
    """

    def __init__(
        self,
        consensus_confidence: float = 0.95,
        polar_confidence: float = 0.70,
        negative_consensus_confidence: float = 0.40,
    ):
        """
        Initialize synthesizer.

        Args:
            consensus_confidence: Confidence for CONSENSUS tier
            polar_confidence: Confidence for POLAR tier
            negative_consensus_confidence: Confidence for NEGATIVE_CONSENSUS tier
        """
        self.confidence_map = {
            ConsensusTier.CONSENSUS: consensus_confidence,
            ConsensusTier.POLAR: polar_confidence,
            ConsensusTier.NEGATIVE_CONSENSUS: negative_consensus_confidence,
        }

    def synthesize_response(self, consensus_result: ConsensusResult) -> CollapsedResponse:
        """
        Collapse superposition to final response.

        Strategy:
        1. Create resonance matrix from all artifacts
        2. If consensus exists, use best consensus artifact
        3. Otherwise use best polar or negative consensus
        4. Map tier to confidence level

        Args:
            consensus_result: Result from consensus calculation

        Returns:
            CollapsedResponse with final answer and confidence
        """
        # Create resonance matrix
        resonance_matrix = ResonanceMatrix.from_consensus_result(consensus_result)

        # Check if we have any results
        if not consensus_result.has_any_results():
            return CollapsedResponse.no_consensus(resonance_matrix)

        # Get best artifact (consensus > polar > negative consensus)
        try:
            best_artifact = consensus_result.get_best_artifact()
        except ValueError:
            return CollapsedResponse.no_consensus(resonance_matrix)

        # Get confidence from tier
        confidence = self.confidence_map.get(best_artifact.tier, 0.0)

        # Use source document IDs instead of content hash
        sources = best_artifact.source_document_ids if hasattr(best_artifact, 'source_document_ids') and best_artifact.source_document_ids else []

        # Create response
        return CollapsedResponse(
            final_answer=best_artifact.content,
            confidence=confidence,
            status="SUCCESS",
            sources=sources,
            resonance_matrix=resonance_matrix,
        )
