"""
Causal Closure Computation.

Computes the transitive closure of the causal dependency graph to enable:
- Complete causal ancestry queries
- Prevention of causal chain splits across shard boundaries
- Efficient routing to all causally-relevant shards

For a causal graph G = (V, E), the transitive closure G* = (V, E*) where:
    E* = {(u, v) : there exists a path from u to v in G}

Uses DFS with memoization for O(V * E) complexity on sparse graphs.
"""

from typing import Dict, Set, List
from collections import defaultdict, deque
import json


class CausalClosure:
    """
    Computes and stores transitive closure of causal dependency graph.
    """

    def __init__(self):
        """Initialize causal closure computer."""
        # Direct causal graph: child_id -> [parent_ids]
        self.causal_graph: Dict[str, List[str]] = defaultdict(list)

        # Transitive closure: event_id -> {all_ancestor_ids}
        self.ancestors: Dict[str, Set[str]] = {}

        # Reverse: event_id -> {all_descendant_ids}
        self.descendants: Dict[str, Set[str]] = {}

        # Statistics
        self.max_chain_depth: int = 0
        self.total_edges: int = 0
        self.total_transitive_edges: int = 0

    def build_from_events(self, events: List[Dict]) -> None:
        """
        Build causal graph from events.

        Args:
            events: List of events with 'id' and 'causal_parents' fields
        """
        self.causal_graph.clear()
        self.total_edges = 0

        for event in events:
            event_id = event.get('id', '')
            causal_parents = event.get('causal_parents', [])

            if not event_id:
                continue

            # Build reverse graph (child -> parents)
            self.causal_graph[event_id] = list(causal_parents)
            self.total_edges += len(causal_parents)

        # Compute transitive closure
        self.compute_closure()

    def compute_closure(self) -> None:
        """
        Compute transitive closure using DFS with memoization.

        For each event, finds all ancestors (events it transitively depends on).
        """
        self.ancestors.clear()
        self.descendants.clear()

        # Compute ancestors for each node
        for event_id in self.causal_graph.keys():
            if event_id not in self.ancestors:
                self._compute_ancestors_dfs(event_id, set())

        # Compute descendants (reverse of ancestors)
        for event_id, ancestor_set in self.ancestors.items():
            for ancestor_id in ancestor_set:
                if ancestor_id not in self.descendants:
                    self.descendants[ancestor_id] = set()
                self.descendants[ancestor_id].add(event_id)

        # Compute statistics
        self._compute_statistics()

    def _compute_ancestors_dfs(self, event_id: str, visited: Set[str]) -> Set[str]:
        """
        Compute all ancestors of an event using DFS with memoization.

        Args:
            event_id: Event to compute ancestors for
            visited: Cycle detection set

        Returns:
            Set of all ancestor event IDs
        """
        # Check memoization cache
        if event_id in self.ancestors:
            return self.ancestors[event_id]

        # Cycle detection
        if event_id in visited:
            return set()

        visited.add(event_id)
        ancestors = set()

        # Get direct parents
        parents = self.causal_graph.get(event_id, [])

        for parent_id in parents:
            # Add parent
            ancestors.add(parent_id)

            # Recursively add parent's ancestors
            parent_ancestors = self._compute_ancestors_dfs(parent_id, visited.copy())
            ancestors.update(parent_ancestors)

        # Memoize
        self.ancestors[event_id] = ancestors
        return ancestors

    def get_ancestors(self, event_id: str) -> Set[str]:
        """
        Get all ancestor event IDs (transitive causal parents).

        Args:
            event_id: Event ID to query

        Returns:
            Set of ancestor event IDs
        """
        return self.ancestors.get(event_id, set())

    def get_descendants(self, event_id: str) -> Set[str]:
        """
        Get all descendant event IDs (transitive causal children).

        Args:
            event_id: Event ID to query

        Returns:
            Set of descendant event IDs
        """
        return self.descendants.get(event_id, set())

    def get_chain_depth(self, event_id: str) -> int:
        """
        Get causal chain depth for an event (max distance to root).

        Args:
            event_id: Event ID to query

        Returns:
            Maximum causal chain depth
        """
        if event_id not in self.ancestors:
            return 0

        # BFS to find maximum depth
        visited = set()
        queue = deque([(event_id, 0)])
        max_depth = 0

        while queue:
            curr_id, depth = queue.popleft()

            if curr_id in visited:
                continue
            visited.add(curr_id)

            max_depth = max(max_depth, depth)

            # Add parents with incremented depth
            parents = self.causal_graph.get(curr_id, [])
            for parent_id in parents:
                queue.append((parent_id, depth + 1))

        return max_depth

    def _compute_statistics(self) -> None:
        """Compute closure statistics."""
        self.total_transitive_edges = sum(
            len(ancestors) for ancestors in self.ancestors.values()
        )

        if self.ancestors:
            self.max_chain_depth = max(
                self.get_chain_depth(event_id)
                for event_id in self.ancestors.keys()
            )
        else:
            self.max_chain_depth = 0

    def to_dict(self) -> Dict:
        """
        Serialize to dictionary for JSON export.

        Returns:
            Dictionary representation
        """
        return {
            "closure_version": "1.0",
            "computation_method": "DFS with memoization",
            "event_ancestors": {
                event_id: list(ancestors)
                for event_id, ancestors in self.ancestors.items()
            },
            "event_descendants": {
                event_id: list(descendants)
                for event_id, descendants in self.descendants.items()
            },
            "statistics": {
                "max_chain_depth": self.max_chain_depth,
                "avg_chain_depth": (
                    sum(self.get_chain_depth(eid) for eid in self.ancestors.keys())
                    / len(self.ancestors)
                    if self.ancestors else 0.0
                ),
                "total_edges": self.total_edges,
                "total_transitive_edges": self.total_transitive_edges
            }
        }

    def save_to_file(self, filepath: str) -> None:
        """
        Save causal closure to JSON file.

        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CausalClosure':
        """
        Load causal closure from JSON file.

        Args:
            filepath: Input file path

        Returns:
            CausalClosure instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        closure = cls()

        # Load ancestors
        closure.ancestors = {
            event_id: set(ancestors)
            for event_id, ancestors in data.get("event_ancestors", {}).items()
        }

        # Load descendants
        closure.descendants = {
            event_id: set(descendants)
            for event_id, descendants in data.get("event_descendants", {}).items()
        }

        # Load statistics
        stats = data.get("statistics", {})
        closure.max_chain_depth = stats.get("max_chain_depth", 0)
        closure.total_edges = stats.get("total_edges", 0)
        closure.total_transitive_edges = stats.get("total_transitive_edges", 0)

        return closure
