"""
Causal Index.

Manages cross-shard causal dependencies to enable:
- Efficient shard ancestry queries
- L1 router expansion to causally-relevant shards
- Cross-shard causal link tracking
"""

from typing import Dict, List, Set
from collections import defaultdict
import json


class CausalIndex:
    """
    Index of causal dependencies across shard boundaries.
    """

    def __init__(self):
        """Initialize empty causal index."""
        self.index_version = "1.0"

        # Event ID -> Shard ID mapping
        self.event_to_shard: Dict[str, str] = {}

        # Cross-shard dependency links
        self.cross_shard_dependencies: List[Dict] = []

        # Shard ancestry (for quick lookup)
        self.shard_ancestry: Dict[str, Dict[str, List[str]]] = {}

    def build_from_shards_and_closure(self,
                                     shards: List[Dict],
                                     causal_closure) -> None:
        """
        Build causal index from shards and causal closure.

        Args:
            shards: List of shards with events
            causal_closure: CausalClosure instance
        """
        # Build event -> shard mapping
        for shard in shards:
            shard_id = shard.get('id', '')
            for event in shard.get('events', []):
                event_id = event.get('id', '')
                if event_id:
                    self.event_to_shard[event_id] = shard_id

        # Find cross-shard dependencies
        for shard in shards:
            shard_id = shard.get('id', '')

            for event in shard.get('events', []):
                event_id = event.get('id', '')
                if not event_id:
                    continue

                # Get ancestors from closure
                ancestors = causal_closure.get_ancestors(event_id)

                for ancestor_id in ancestors:
                    if ancestor_id in self.event_to_shard:
                        ancestor_shard = self.event_to_shard[ancestor_id]

                        # Check if cross-shard dependency
                        if ancestor_shard != shard_id:
                            self.cross_shard_dependencies.append({
                                'child_event': event_id,
                                'child_shard': shard_id,
                                'parent_event': ancestor_id,
                                'parent_shard': ancestor_shard,
                                'dependency_type': 'causal'
                            })

        # Build shard ancestry
        self._build_shard_ancestry()

    def _build_shard_ancestry(self) -> None:
        """Build shard-level ancestry from event dependencies."""
        shard_direct_ancestors = defaultdict(set)
        shard_all_ancestors = {}

        # Collect direct ancestors for each shard
        for dep in self.cross_shard_dependencies:
            child_shard = dep['child_shard']
            parent_shard = dep['parent_shard']
            shard_direct_ancestors[child_shard].add(parent_shard)

        # Compute transitive closure of shard ancestry
        def get_all_ancestors(shard_id: str, visited: Set[str]) -> Set[str]:
            if shard_id in shard_all_ancestors:
                return shard_all_ancestors[shard_id]

            if shard_id in visited:
                return set()

            visited.add(shard_id)
            ancestors = set(shard_direct_ancestors.get(shard_id, []))

            for parent in list(ancestors):
                ancestors.update(get_all_ancestors(parent, visited.copy()))

            return ancestors

        # Compute for all shards
        all_shard_ids = set(self.event_to_shard.values())
        for shard_id in all_shard_ids:
            shard_all_ancestors[shard_id] = get_all_ancestors(shard_id, set())

        # Store in ancestry structure
        for shard_id in all_shard_ids:
            self.shard_ancestry[shard_id] = {
                'direct_ancestors': sorted(list(shard_direct_ancestors.get(shard_id, []))),
                'transitive_ancestors': sorted(list(shard_all_ancestors.get(shard_id, [])))
            }

    def get_ancestor_shards(self, shard_id: str, transitive: bool = True) -> List[str]:
        """
        Get ancestor shards for a given shard.

        Args:
            shard_id: Shard to query
            transitive: If True, return all transitive ancestors; if False, only direct

        Returns:
            List of ancestor shard IDs
        """
        if shard_id not in self.shard_ancestry:
            return []

        key = 'transitive_ancestors' if transitive else 'direct_ancestors'
        return self.shard_ancestry[shard_id].get(key, [])

    def get_required_shards_for_query(self,
                                     primary_shard_id: str,
                                     max_depth: int = 2) -> List[str]:
        """
        Get all shards required to answer queries in primary shard.

        Includes primary shard plus ancestors up to max_depth.

        Args:
            primary_shard_id: Primary target shard
            max_depth: Maximum ancestry depth to include

        Returns:
            List of shard IDs including primary and ancestors
        """
        required = {primary_shard_id}

        # BFS to depth-limited ancestors
        from collections import deque
        queue = deque([(primary_shard_id, 0)])
        visited = set()

        while queue:
            shard_id, depth = queue.popleft()

            if shard_id in visited:
                continue
            visited.add(shard_id)

            if depth < max_depth:
                direct_ancestors = self.shard_ancestry.get(shard_id, {}).get('direct_ancestors', [])
                for ancestor in direct_ancestors:
                    required.add(ancestor)
                    queue.append((ancestor, depth + 1))

        return sorted(list(required))

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'index_version': self.index_version,
            'event_to_shard': self.event_to_shard,
            'cross_shard_dependencies': self.cross_shard_dependencies,
            'shard_ancestry': self.shard_ancestry
        }

    def save_to_file(self, filepath: str) -> None:
        """Save index to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CausalIndex':
        """Load index from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        index = cls()
        index.index_version = data.get('index_version', '1.0')
        index.event_to_shard = data.get('event_to_shard', {})
        index.cross_shard_dependencies = data.get('cross_shard_dependencies', [])
        index.shard_ancestry = data.get('shard_ancestry', {})

        return index
