"""
Shard Manifest.

Manages metadata about tempo-normalized shards including:
- Boundary signals and types
- Time ranges
- Token counts
- Causal context references
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class Shard:
    """Metadata about a time shard."""
    id: str
    filename: str
    time_range: Dict[str, str]  # {start, end}
    event_count: int
    token_count: int
    boundary_signals: Dict
    causal_context: Dict[str, List[str]]  # {ancestor_shards, descendant_shards}
    foveated_summaries: Dict[str, str]  # {L1_summary_id, L2_epoch_id, L3_domains}


class ShardManifest:
    """
    Manages shard boundary metadata and provides query interface.
    """

    def __init__(self):
        """Initialize empty manifest."""
        self.manifest_version = "2.0"
        self.algorithm = "tempo_normalized_density_constrained"
        self.parameters = {}
        self.shards: List[Shard] = []
        self.statistics = {}

    def add_shard(self, shard: Shard) -> None:
        """Add shard to manifest."""
        self.shards.append(shard)

    def get_shard_by_id(self, shard_id: str) -> Optional[Shard]:
        """Get shard by ID."""
        for shard in self.shards:
            if shard.id == shard_id:
                return shard
        return None

    def get_shards_for_timestamp(self, timestamp: str) -> List[Shard]:
        """
        Get shards containing a specific timestamp.

        Args:
            timestamp: ISO format timestamp

        Returns:
            List of shards (usually 1, could be 0 or multiple)
        """
        matching_shards = []
        for shard in self.shards:
            start = shard.time_range['start']
            end = shard.time_range['end']

            if start <= timestamp <= end:
                matching_shards.append(shard)

        return matching_shards

    def get_shards_for_time_range(self, start: str, end: str) -> List[Shard]:
        """Get all shards overlapping with time range."""
        matching_shards = []
        for shard in self.shards:
            shard_start = shard.time_range['start']
            shard_end = shard.time_range['end']

            # Check for overlap
            if not (shard_end < start or shard_start > end):
                matching_shards.append(shard)

        return matching_shards

    def compute_statistics(self) -> Dict:
        """Compute manifest statistics."""
        if not self.shards:
            return {}

        token_counts = [s.token_count for s in self.shards]
        event_counts = [s.event_count for s in self.shards]

        # Count boundary types
        boundary_types = {}
        for shard in self.shards:
            b_type = shard.boundary_signals.get('type', 'unknown')
            boundary_types[b_type] = boundary_types.get(b_type, 0) + 1

        self.statistics = {
            'total_events': sum(event_counts),
            'total_shards': len(self.shards),
            'avg_shard_tokens': sum(token_counts) / len(token_counts),
            'max_shard_tokens': max(token_counts),
            'min_shard_tokens': min(token_counts),
            'avg_events_per_shard': sum(event_counts) / len(event_counts),
            'boundary_types': boundary_types,
            'causal_chains_preserved': self._check_causal_preservation()
        }

        return self.statistics

    def _check_causal_preservation(self) -> bool:
        """Check if causal chains are preserved (no orphaned dependencies)."""
        # Simple check: every shard with ancestors should be reachable
        for shard in self.shards:
            ancestors = shard.causal_context.get('ancestor_shards', [])
            if ancestors:
                # Check that all ancestor shards exist
                for ancestor_id in ancestors:
                    if not self.get_shard_by_id(ancestor_id):
                        return False
        return True

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        from datetime import datetime

        return {
            'manifest_version': self.manifest_version,
            'generation_timestamp': datetime.utcnow().isoformat() + 'Z',
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'shards': [asdict(s) for s in self.shards],
            'global_statistics': self.statistics
        }

    def save_to_file(self, filepath: str) -> None:
        """Save manifest to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ShardManifest':
        """Load manifest from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        manifest = cls()
        manifest.manifest_version = data.get('manifest_version', '2.0')
        manifest.algorithm = data.get('algorithm', '')
        manifest.parameters = data.get('parameters', {})
        manifest.statistics = data.get('global_statistics', {})

        # Load shards
        for shard_data in data.get('shards', []):
            shard = Shard(**shard_data)
            manifest.shards.append(shard)

        return manifest
