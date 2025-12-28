
import pytest
import uuid
import json
from dataclasses import asdict
from typing import List

from dpr_rc.domain.active_agent.services.consensus_calculator import ConsensusCalculator, Vote
from dpr_rc.domain.active_agent.services.response_synthesizer import ResponseSynthesizer
from dpr_rc.domain.active_agent.entities.consensus_state import ConsensusTier, ConsensusResult, ArtifactConsensus

def create_vote(
    content: str, 
    cluster_id: str, 
    binary_vote: int, 
    confidence: float = 1.0, 
    author_cluster: str = "C_RECENT"
) -> Vote:
    return Vote(
        content_hash=str(hash(content)),
        content_snippet=content,
        cluster_id=cluster_id,
        binary_vote=binary_vote,
        author_cluster=author_cluster,
        confidence_score=confidence,
        document_ids=[f"doc_{uuid.uuid4().hex[:8]}"]
    )

class TestConsensusSynthesisIntegration:
    """
    Integration test for Edge E12 -> E15 of the runtime sequence diagram.
    Tests data flow from Aggregated Votes -> ConsensusCalculator -> ResponseSynthesizer.
    """
    
    @pytest.fixture
    def calculator(self):
        # theta=0.5 (50% approval needed within cluster)
        # tau=0.5 (50% of clusters needed for consensus is effectively > 0.5 ratio, or >= tau)
        # Default is theta=0.5, tau=0.667. Using defaults.
        return ConsensusCalculator(theta=0.5, tau=0.6) # Lower tau slightly to make 2-cluster logic cleaner (1/2 = 0.5, 2/2 = 1.0)

    @pytest.fixture
    def synthesizer(self):
        return ResponseSynthesizer()

    def test_strong_consensus(self, calculator, synthesizer):
        """
        Scenario 1: Strong Consensus
        Both C_RECENT and C_OLDER clusters approve the same content.
        """
        print("\n=== SCENARIO 1: STRONG CONSENSUS ===")
        content = "The answer is A."
        votes = [
            create_vote(content, "C_RECENT", 1),
            create_vote(content, "C_RECENT", 1), # 100% approval in C_RECENT
            create_vote(content, "C_OLDER", 1),
            create_vote(content, "C_OLDER", 1),  # 100% approval in C_OLDER
        ]
        
        # E12 -> E13: Calculate
        consensus_result = calculator.calculate_consensus(votes)
        
        # Verify E14 Output
        print("\nConsensus Result (E14):")
        # Helper to print formatting cleanly
        self._print_consensus_result(consensus_result)
        
        assert consensus_result.has_consensus()
        assert len(consensus_result.consensus_artifacts) == 1
        artifact = consensus_result.consensus_artifacts[0]
        assert artifact.tier == ConsensusTier.CONSENSUS
        assert artifact.agreement_ratio == 1.0  # 2/2 clusters approved
        
        # E15: Synthesize
        response = synthesizer.synthesize_response(consensus_result)
        print("\nCollapsed Response (E16):")
        print(response)

        assert response.status == "SUCCESS"
        assert response.final_answer == content
        assert response.confidence >= 0.95
        assert hasattr(response, 'resonance_matrix')
        assert not response.resonance_matrix.is_empty()

    def test_strong_polarization(self, calculator, synthesizer):
        """
        Scenario 2: Strong Polarization
        C_RECENT approves "Answer A", C_OLDER approves "Answer B".
        Or C_RECENT approves A, C_OLDER rejects A. Let's do rejection for same content first.
        """
        print("\n=== SCENARIO 2: STRONG POLARIZATION ===")
        content = "The answer is Controversial."
        votes = [
            create_vote(content, "C_RECENT", 1),
            create_vote(content, "C_RECENT", 1), # 100% approval in C_RECENT (Cluster approves)
            
            create_vote(content, "C_OLDER", 0),
            create_vote(content, "C_OLDER", 0),  # 0% approval in C_OLDER (Cluster rejects)
        ]
        
        # E12 -> E13
        consensus_result = calculator.calculate_consensus(votes)
        
        print("\nConsensus Result (E14):")
        self._print_consensus_result(consensus_result)
        
        # With 2 clusters, 1 approves -> ratio = 0.5.
        # If tau = 0.6, then 1-tau = 0.4.
        # 0.4 < 0.5 < 0.6 -> POLAR tier.
        
        assert len(consensus_result.polar_artifacts) == 1
        artifact = consensus_result.polar_artifacts[0]
        assert artifact.tier == ConsensusTier.POLAR
        assert artifact.agreement_ratio == 0.5
        
        # E15
        response = synthesizer.synthesize_response(consensus_result)
        print("\nCollapsed Response (E16):")
        print(response)
        
        assert response.status == "SUCCESS"
        assert response.confidence < 0.95 and response.confidence >= 0.70

    def test_strong_negative_consensus(self, calculator, synthesizer):
        """
        Scenario 3: Strong Negative Consensus
        Both clusters reject the content.
        """
        print("\n=== SCENARIO 3: STRONG NEGATIVE CONSENSUS ===")
        content = "The answer is Wrong."
        votes = [
            create_vote(content, "C_RECENT", 0),
            create_vote(content, "C_RECENT", 0), # 0% approval -> Reject
            
            create_vote(content, "C_OLDER", 0),
            create_vote(content, "C_OLDER", 0),  # 0% approval -> Reject
        ]
        
        # E12 -> E13
        consensus_result = calculator.calculate_consensus(votes)
        
        print("\nConsensus Result (E14):")
        self._print_consensus_result(consensus_result)
        
        # 0 clusters approve -> ratio = 0.0.
        # 0.0 <= 1-tau (0.4) -> NEGATIVE_CONSENSUS.
        
        assert len(consensus_result.negative_consensus_artifacts) == 1
        artifact = consensus_result.negative_consensus_artifacts[0]
        assert artifact.tier == ConsensusTier.NEGATIVE_CONSENSUS
        assert artifact.agreement_ratio == 0.0
        
        # E15
        response = synthesizer.synthesize_response(consensus_result)
        print("\nCollapsed Response (E16):")
        print(response)
        
        assert response.confidence <= 0.40

    def test_mixed_resonance_matrix(self, calculator, synthesizer):
        """
        Scenario 4: Mixed Resonance Matrix
        Multiple artifacts with varying resonance vectors producing a fully populated matrix.
        - Artifact A: Consensus (v+, v+)
        - Artifact B: Polar (v+, v-)
        - Artifact C: Negative Consensus (v-, v-)
        """
        print("\n=== SCENARIO 4: MIXED RESONANCE MATRIX ===")
        
        # Artifact A: Consensus
        content_a = "Consensus Answer"
        votes_a = [
            create_vote(content_a, "C_RECENT", 1),
            create_vote(content_a, "C_OLDER", 1)
        ]

        # Artifact B: Polar
        content_b = "Polar Answer"
        votes_b = [
            create_vote(content_b, "C_RECENT", 1),
            create_vote(content_b, "C_OLDER", 0)
        ]

        # Artifact C: Negative Consensus
        content_c = "Rejected Answer"
        votes_c = [
            create_vote(content_c, "C_RECENT", 0),
            create_vote(content_c, "C_OLDER", 0)
        ]

        all_votes = votes_a + votes_b + votes_c

        # E12 -> E13
        consensus_result = calculator.calculate_consensus(all_votes)

        print("\nConsensus Result (E14):")
        self._print_consensus_result(consensus_result)

        # Verify all tiers are populated
        assert len(consensus_result.consensus_artifacts) == 1
        assert consensus_result.consensus_artifacts[0].content == content_a
        
        assert len(consensus_result.polar_artifacts) == 1
        assert consensus_result.polar_artifacts[0].content == content_b
        
        assert len(consensus_result.negative_consensus_artifacts) == 1
        assert consensus_result.negative_consensus_artifacts[0].content == content_c

        # E15
        response = synthesizer.synthesize_response(consensus_result)
        print("\nCollapsed Response (E16):")
        print(response)

        # Verify Resonance Matrix Population
        matrix = response.resonance_matrix
        assert len(matrix.consensus) == 1
        assert len(matrix.polar) == 1
        assert len(matrix.negative_consensus) == 1

        # Verify collapse logic (Best Artifact = Consensus)
        assert response.final_answer == content_a
        assert response.confidence == 0.95

    def _print_consensus_result(self, result: ConsensusResult):
        # Manual JSON dumping for clarity as some objects might not be directly serializable without custom encoder
        print(f"Total Votes: {result.total_votes}")
        print(f"Num Clusters: {result.num_clusters}")
        print("Consensus Artifacts:")
        for a in result.consensus_artifacts:
            print(json.dumps(a.to_dict(), indent=2))
        print("Polar Artifacts:")
        for a in result.polar_artifacts:
            print(json.dumps(a.to_dict(), indent=2))
        print("Negative Consensus Artifacts:")
        for a in result.negative_consensus_artifacts:
            print(json.dumps(a.to_dict(), indent=2))
