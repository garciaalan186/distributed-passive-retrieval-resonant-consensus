"""
Summary Index.

Manages foveated summary layers for efficient retrieval.
"""

from typing import Dict, List, Optional
import json


class SummaryIndex:
    """
    Index for multi-resolution foveated summaries.
    """

    def __init__(self):
        """Initialize empty summary index."""
        self.summaries_by_layer: Dict[str, Dict] = {
            'L1': {},  # shard_id -> summary
            'L2': {},  # epoch_id -> summary
            'L3': {}   # domain -> summary
        }

    def add_summary(self, layer: str, summary_id: str, summary_data: Dict) -> None:
        """Add summary to index."""
        if layer not in self.summaries_by_layer:
            raise ValueError(f"Invalid layer: {layer}")

        self.summaries_by_layer[layer][summary_id] = summary_data

    def get_summary(self, layer: str, summary_id: str) -> Optional[Dict]:
        """Get summary by layer and ID."""
        return self.summaries_by_layer.get(layer, {}).get(summary_id)

    def get_summaries_for_layer(self, layer: str) -> Dict:
        """Get all summaries for a layer."""
        return self.summaries_by_layer.get(layer, {})

    def save_to_file(self, filepath: str) -> None:
        """Save summary index to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.summaries_by_layer, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'SummaryIndex':
        """Load summary index from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        index = cls()
        index.summaries_by_layer = data

        return index
