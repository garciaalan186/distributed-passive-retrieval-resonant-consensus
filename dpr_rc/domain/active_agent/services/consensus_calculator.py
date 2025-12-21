"""
Domain Service: Consensus Calculator

Implements RCP v4 (Resonant Consensus Protocol) equations for multi-cluster consensus.
Pure business logic with no infrastructure dependencies.
"""

from typing import List, Dict, Set
from ..entities import (
    ConsensusTier,
    ArtifactConsensus,
    ConsensusResult,
)


class Vote:
    """Simplified vote representation for consensus calculation."""

    def __init__(
        self,
        content_hash: str,
        content_snippet: str,
        cluster_id: str,
        binary_vote: int,
        author_cluster: str,
        confidence_score: float = 0.0,
        document_ids: List[str] = None,
    ):
        self.content_hash = content_hash
        self.content_snippet = content_snippet
        self.cluster_id = cluster_id
        self.binary_vote = binary_vote
        self.author_cluster = author_cluster
        self.confidence_score = confidence_score
        self.document_ids = document_ids or []


class ConsensusCalculator:
    """
    Domain service for RCP v4 consensus calculation.

    Implements Equations 1-6 from the RCP v4 specification.
    """

    def __init__(self, theta: float = 0.5, tau: float = 0.667):
        """
        Initialize calculator.

        Args:
            theta: Cluster approval threshold (RCP v4 θ_i)
            tau: Consensus tier threshold (RCP v4 τ)
        """
        self.theta = theta
        self.tau = tau

    def calculate_consensus(self, votes: List) -> ConsensusResult:
        """
        Calculate consensus from votes using RCP v4.

        Args:
            votes: List of votes (ConsensusVote objects or Vote objects)

        Returns:
            ConsensusResult with artifacts classified by tier
        """
        if not votes:
            return ConsensusResult(
                consensus_artifacts=[],
                polar_artifacts=[],
                negative_consensus_artifacts=[],
                total_votes=0,
                num_clusters=0,
            )

        # Group votes by content hash
        artifacts: Dict[str, Dict] = {}
        for v in votes:
            if v.content_hash not in artifacts:
                artifacts[v.content_hash] = {
                    "content": v.content_snippet,
                    "votes": [],
                }
            artifacts[v.content_hash]["votes"].append(v)

        # Determine number of clusters
        all_clusters = set(v.cluster_id for v in votes)
        num_clusters = len(all_clusters)

        # Process each artifact through RCP v4 pipeline
        consensus_artifacts = []
        polar_artifacts = []
        negative_consensus_artifacts = []

        for content_hash, data in artifacts.items():
            artifact_votes = data["votes"]

            # RCP v4 Eq. 2: Compute approval set S(ω)
            approval_set = self._compute_approval_set(artifact_votes, all_clusters)

            # RCP v4 Eq. 3: Compute agreement ratio ρ(ω)
            agreement_ratio = self._compute_agreement_ratio(approval_set, num_clusters)

            # RCP v4 Eq. 4: Classify artifact tier
            tier = self._classify_artifact(agreement_ratio)

            # RCP v4 Eq. 5: Compute artifact score
            score = self._compute_artifact_score(artifact_votes)

            # RCP v4: Compute resonance vector [v+, v-]
            resonance_vector = self._compute_resonance_vector(artifact_votes)

            # Aggregate source document IDs from all votes for this artifact
            source_document_ids = []
            for vote in artifact_votes:
                if hasattr(vote, 'document_ids') and vote.document_ids:
                    source_document_ids.extend(vote.document_ids)
            # Deduplicate while preserving order
            source_document_ids = list(dict.fromkeys(source_document_ids))

            # Create artifact consensus object
            artifact = ArtifactConsensus(
                content_hash=content_hash,
                content=data["content"],
                votes_count=len(artifact_votes),
                approval_set=approval_set,
                agreement_ratio=agreement_ratio,
                tier=tier,
                score=score,
                resonance_vector=resonance_vector,
                source_document_ids=source_document_ids,
            )

            # Classify into tiers
            if tier == ConsensusTier.CONSENSUS:
                consensus_artifacts.append(artifact)
            elif tier == ConsensusTier.POLAR:
                polar_artifacts.append(artifact)
            elif tier == ConsensusTier.NEGATIVE_CONSENSUS:
                negative_consensus_artifacts.append(artifact)

        return ConsensusResult(
            consensus_artifacts=consensus_artifacts,
            polar_artifacts=polar_artifacts,
            negative_consensus_artifacts=negative_consensus_artifacts,
            total_votes=len(votes),
            num_clusters=num_clusters,
        )

    def _compute_cluster_approval(
        self, votes_for_artifact: List, cluster_id: str
    ) -> bool:
        """
        RCP v4 Equation 1: Cluster Approval

        Approve_i(ω) = 1 if (1/|Ci|) * Σ v(ω,a) >= θ_i
        """
        cluster_votes = [v for v in votes_for_artifact if v.cluster_id == cluster_id]

        if not cluster_votes:
            return False

        # Count binary approvals in this cluster
        approvals = sum(v.binary_vote for v in cluster_votes)
        approval_rate = approvals / len(cluster_votes)

        return approval_rate >= self.theta

    def _compute_approval_set(
        self, votes_for_artifact: List, all_clusters: Set[str]
    ) -> Set[str]:
        """
        RCP v4 Equation 2: Approval Set

        S(ω) = {C_i ∈ C : Approve_i(ω) = 1}
        """
        approval_set = set()

        for cluster_id in all_clusters:
            if self._compute_cluster_approval(votes_for_artifact, cluster_id):
                approval_set.add(cluster_id)

        return approval_set

    def _compute_agreement_ratio(
        self, approval_set: Set[str], total_clusters: int
    ) -> float:
        """
        RCP v4 Equation 3: Agreement Ratio

        ρ(ω) = |S(ω)| / n
        """
        if total_clusters == 0:
            return 0.0
        return len(approval_set) / total_clusters

    def _classify_artifact(self, agreement_ratio: float) -> ConsensusTier:
        """
        RCP v4 Equation 4: Tier Classification

        Tier(ω) = { Consensus          if ρ(ω) >= τ
                  { Polar              if 1-τ < ρ(ω) < τ
                  { Negative_Consensus if ρ(ω) <= 1-τ
        """
        if agreement_ratio >= self.tau:
            return ConsensusTier.CONSENSUS
        elif agreement_ratio > (1 - self.tau):
            return ConsensusTier.POLAR
        else:
            return ConsensusTier.NEGATIVE_CONSENSUS

    def _compute_artifact_score(self, votes_for_artifact: List) -> float:
        """
        RCP v4 Equation 5: Artifact Score

        Score(ω) = (1 / |A|-1) * Σ v(ω,a)  for a ≠ author(ω)
        """
        if not votes_for_artifact:
            return 0.0

        # Get author cluster
        author_cluster = votes_for_artifact[0].author_cluster

        # Count approvals from non-author agents
        non_author_votes = [
            v for v in votes_for_artifact if v.cluster_id != author_cluster
        ]

        if not non_author_votes:
            # Only author cluster voted
            author_votes = [
                v for v in votes_for_artifact if v.cluster_id == author_cluster
            ]
            if not author_votes:
                return 0.0
            approvals = sum(v.binary_vote for v in author_votes)
            return approvals / len(author_votes)

        # Score based on non-author approvals
        approvals = sum(v.binary_vote for v in non_author_votes)
        return approvals / len(non_author_votes)

    def _compute_resonance_vector(self, votes_for_artifact: List) -> List[float]:
        """
        RCP v4: Resonance Vector for DPR-RC Temporal Clustering

        Returns [v+, v-] where:
        - v+ = approval rate from C_RECENT cluster
        - v- = approval rate from C_OLDER cluster
        """
        recent_votes = [v for v in votes_for_artifact if v.cluster_id == "C_RECENT"]
        older_votes = [v for v in votes_for_artifact if v.cluster_id == "C_OLDER"]

        # Compute approval rates
        v_plus = (
            (sum(v.binary_vote for v in recent_votes) / len(recent_votes))
            if recent_votes
            else 0.0
        )
        v_minus = (
            (sum(v.binary_vote for v in older_votes) / len(older_votes))
            if older_votes
            else 0.0
        )

        return [round(v_plus, 2), round(v_minus, 2)]
