"""
Causal-Aware Partitioner.

Refines tempo-normalized boundaries to respect causal dependencies.
Prevents splitting causal chains across shard boundaries.

Critical for DPR-RC architectural requirement that workers can retrieve
complete causal context for consensus formation.
"""

from typing import List, Dict, Set, Optional
from dataclasses import replace

from .tempo_normalizer import BoundaryCandidate
from .density_optimizer import Shard
from ..indices.causal_closure import CausalClosure


class CausalAwarePartitioner:
    """
    Refines shard boundaries to preserve causal chain integrity.
    """

    def __init__(self, max_shift: int = 50):
        """
        Initialize causal-aware partitioner.

        Args:
            max_shift: Maximum number of events a boundary can be shifted
        """
        self.max_shift = max_shift

    def refine_boundaries_for_causality(self,
                                       boundaries: List[BoundaryCandidate],
                                       events: List[Dict],
                                       causal_closure: CausalClosure) -> List[BoundaryCandidate]:
        """
        Refine boundaries to prevent splitting causal chains.

        Algorithm:
        1. For each boundary, identify events before/after split
        2. Check if events after boundary have ancestors before boundary
        3. If causal violation detected, attempt to shift boundary
        4. If shift exceeds max_shift, drop the boundary

        Args:
            boundaries: Candidate boundaries from tempo-normalization
            events: All events
            causal_closure: Computed transitive closure

        Returns:
            Refined boundaries that respect causality
        """
        refined_boundaries = []
        events_by_id = {e.get('id', ''): e for e in events}

        for boundary in boundaries:
            split_index = boundary.index

            # Events before boundary
            before_events = events[:split_index]
            before_ids = {e.get('id', '') for e in before_events}

            # Events after boundary
            after_events = events[split_index:]
            after_ids = {e.get('id', '') for e in after_events}

            # Check for causal violations
            violations = []

            for event in after_events:
                event_id = event.get('id', '')
                if not event_id:
                    continue

                # Get all ancestors of this event
                ancestors = causal_closure.get_ancestors(event_id)

                # Find ancestors that would be split across boundary
                split_ancestors = ancestors - before_ids - after_ids

                # Find ancestors in the "before" set (would cause violation)
                missing_ancestors = ancestors & before_ids

                if missing_ancestors:
                    violations.append({
                        'event_id': event_id,
                        'missing_ancestors': missing_ancestors
                    })

            if not violations:
                # No violations - boundary is valid
                refined_boundaries.append(boundary)
            else:
                # Attempt to shift boundary
                shifted_boundary = self._shift_boundary_for_causality(
                    boundary,
                    violations,
                    events,
                    events_by_id,
                    causal_closure
                )

                if shifted_boundary is not None:
                    refined_boundaries.append(shifted_boundary)
                # else: boundary dropped (cannot be shifted to respect causality)

        return refined_boundaries

    def _shift_boundary_for_causality(self,
                                     boundary: BoundaryCandidate,
                                     violations: List[Dict],
                                     events: List[Dict],
                                     events_by_id: Dict[str, Dict],
                                     causal_closure: CausalClosure) -> Optional[BoundaryCandidate]:
        """
        Attempt to shift boundary to include all necessary causal ancestors.

        Args:
            boundary: Original boundary
            violations: List of causal violations
            events: All events
            events_by_id: Event ID -> event mapping
            causal_closure: Causal closure

        Returns:
            Shifted boundary or None if shift exceeds max_shift
        """
        # Collect all required ancestors
        required_ancestors = set()
        for violation in violations:
            required_ancestors.update(violation['missing_ancestors'])

        # Find the latest timestamp among required ancestors
        latest_required_ts = None
        latest_required_idx = -1

        for ancestor_id in required_ancestors:
            if ancestor_id in events_by_id:
                ancestor_event = events_by_id[ancestor_id]
                ancestor_ts = ancestor_event.get('timestamp', '')

                # Find this event's index
                for idx, event in enumerate(events):
                    if event.get('id', '') == ancestor_id:
                        if idx > latest_required_idx:
                            latest_required_idx = idx
                            latest_required_ts = ancestor_ts
                        break

        if latest_required_idx < 0:
            # Could not find required ancestors in event list
            return None

        # New boundary must be AFTER all required ancestors
        new_index = latest_required_idx + 1

        # Check if shift is within tolerance
        shift_amount = abs(new_index - boundary.index)
        if shift_amount > self.max_shift:
            # Too much shift required, drop boundary
            return None

        # Create shifted boundary
        shifted_boundary = replace(
            boundary,
            index=new_index,
            timestamp=events[new_index].get('timestamp', '') if new_index < len(events) else boundary.timestamp,
            strength=boundary.strength * 0.8,  # Reduce strength for shifted boundaries
            shifted_from=boundary.index
        )

        return shifted_boundary

    def partition_into_shards(self,
                            events: List[Dict],
                            boundaries: List[BoundaryCandidate],
                            causal_closure: CausalClosure) -> List[List[Dict]]:
        """
        Partition events into shards using refined boundaries.

        Args:
            events: All events
            boundaries: Refined boundaries
            causal_closure: Causal closure

        Returns:
            List of event lists (one per shard)
        """
        if not boundaries:
            return [events]

        # Sort boundaries by index
        sorted_boundaries = sorted(boundaries, key=lambda b: b.index)

        shards = []
        start_idx = 0

        for boundary in sorted_boundaries:
            end_idx = boundary.index
            if start_idx < end_idx:
                shard_events = events[start_idx:end_idx]
                shards.append(shard_events)
            start_idx = end_idx

        # Add final shard
        if start_idx < len(events):
            shards.append(events[start_idx:])

        return shards

    def enrich_shards_with_causal_context(self,
                                         shards: List[Shard],
                                         causal_closure: CausalClosure,
                                         events_by_id: Dict[str, Dict]) -> None:
        """
        Enrich shards with causal ancestor/descendant information.

        Modifies shards in-place to add:
        - causal_ancestors: List of shard IDs containing causal parents
        - causal_descendants: List of shard IDs containing causal children

        Args:
            shards: List of shards to enrich
            causal_closure: Causal closure
            events_by_id: Event ID -> event mapping
        """
        # Build mapping: event_id -> shard_id
        event_to_shard = {}
        for shard in shards:
            for event in shard.events:
                event_id = event.get('id', '')
                if event_id:
                    event_to_shard[event_id] = shard.id

        # For each shard, find ancestor and descendant shards
        for shard in shards:
            ancestor_shards = set()
            descendant_shards = set()

            for event in shard.events:
                event_id = event.get('id', '')
                if not event_id:
                    continue

                # Find ancestors
                ancestors = causal_closure.get_ancestors(event_id)
                for ancestor_id in ancestors:
                    if ancestor_id in event_to_shard:
                        ancestor_shard_id = event_to_shard[ancestor_id]
                        if ancestor_shard_id != shard.id:
                            ancestor_shards.add(ancestor_shard_id)

                # Find descendants
                descendants = causal_closure.get_descendants(event_id)
                for descendant_id in descendants:
                    if descendant_id in event_to_shard:
                        descendant_shard_id = event_to_shard[descendant_id]
                        if descendant_shard_id != shard.id:
                            descendant_shards.add(descendant_shard_id)

            # Update shard with causal context
            shard.causal_ancestors = sorted(list(ancestor_shards))
            shard.causal_descendants = sorted(list(descendant_shards))

    def validate_no_splits(self,
                          shards: List[Shard],
                          causal_closure: CausalClosure) -> bool:
        """
        Validate that no causal chains are split across shards.

        Args:
            shards: List of shards
            causal_closure: Causal closure

        Returns:
            True if no causal chains are split, False otherwise
        """
        # Build event to shard mapping
        event_to_shard = {}
        for shard in shards:
            for event in shard.events:
                event_id = event.get('id', '')
                if event_id:
                    event_to_shard[event_id] = shard.id

        # Check each event
        for shard in shards:
            for event in shard.events:
                event_id = event.get('id', '')
                if not event_id:
                    continue

                # Get ancestors
                ancestors = causal_closure.get_ancestors(event_id)

                # Check if all ancestors are either:
                # 1. In the same shard, OR
                # 2. In an ancestor shard (listed in shard.causal_ancestors)
                event_shard_id = event_to_shard[event_id]
                allowed_shards = {event_shard_id} | set(shard.causal_ancestors)

                for ancestor_id in ancestors:
                    if ancestor_id in event_to_shard:
                        ancestor_shard_id = event_to_shard[ancestor_id]
                        if ancestor_shard_id not in allowed_shards:
                            # Causal chain is split!
                            print(f"WARNING: Event {event_id} in shard {event_shard_id} "
                                  f"has ancestor {ancestor_id} in shard {ancestor_shard_id} "
                                  f"which is not listed in causal_ancestors")
                            return False

        return True
