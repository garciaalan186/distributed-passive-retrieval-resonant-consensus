"""
Index structures for DPR-RC time shard boundaries.

Manages:
- Shard manifests (boundary metadata)
- Causal indices (cross-shard dependencies)
- Transitive closure (full causal ancestry)
"""

from .shard_manifest import ShardManifest, Shard
from .causal_index import CausalIndex
from .causal_closure import CausalClosure

__all__ = [
    'ShardManifest',
    'Shard',
    'CausalIndex',
    'CausalClosure',
]
