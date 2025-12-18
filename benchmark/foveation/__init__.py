"""
Foveated summarization module for DPR-RC.

Implements multi-resolution hierarchical summarization:
- L_0: Raw events (20,000 tokens per shard)
- L_1: Shard summaries (4,000 tokens, 5:1 compression)
- L_2: Epoch summaries (2,000 tokens, 2:1 compression)
- L_3: Domain summaries (1,000 tokens, 2:1 compression)

Per DPR-RC Mathematical Model Section 3.
"""

from .layer_generator import LayerGenerator, COMPRESSION_RATIOS
from .summary_index import SummaryIndex

__all__ = [
    'LayerGenerator',
    'SummaryIndex',
    'COMPRESSION_RATIOS',
]
