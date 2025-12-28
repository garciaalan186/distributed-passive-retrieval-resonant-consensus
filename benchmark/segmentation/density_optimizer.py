"""
Information Density Constraint Optimizer.

Enforces H_max constraint from DPR-RC Mathematical Model Section 8, Equation 19:
    H(N) <= H_max

Where H_max corresponds to the SLM context window capacity.

CORRECTED BUDGET CALCULATION:
-----------------------------
SLM Context Window:     32,768 tokens
Usable (85%):          28,000 tokens

Fixed overhead:
- System prompt:          500 tokens
- Instructions:           300 tokens
- Query context:          200 tokens
- Response buffer:      2,000 tokens
- Safety margin:        3,000 tokens
--------------------------------
Total overhead:         6,000 tokens

Available for content:  22,000 tokens
Recommended shard size: 20,000 tokens (with buffer)
"""

from typing import List, Dict
from dataclasses import dataclass
import tiktoken

# Corrected budget constants
SLM_CONTEXT_WINDOW = 32768
USABLE_RATIO = 0.85
PROMPT_OVERHEAD = 500
INSTRUCTIONS = 300
QUERY_CONTEXT = 200
RESPONSE_BUFFER = 2000
SAFETY_MARGIN = 3000

FIXED_OVERHEAD = (
    PROMPT_OVERHEAD
    + INSTRUCTIONS
    + QUERY_CONTEXT
    + RESPONSE_BUFFER
    + SAFETY_MARGIN
)

MAX_INPUT_CONTENT = int(SLM_CONTEXT_WINDOW * USABLE_RATIO - FIXED_OVERHEAD)
CORRECTED_MAX_SHARD_TOKENS = 20000  # Conservative target with buffer

# Minimum shard size to avoid too much fragmentation
MIN_SHARD_TOKENS = 5000


@dataclass
class Shard:
    """A time shard with density constraints enforced."""
    id: str
    events: List[Dict]
    time_range: tuple  # (start_timestamp, end_timestamp)
    token_count: int
    causal_ancestors: List[str]  # Shard IDs
    causal_descendants: List[str]  # Shard IDs
    boundary_signals: Dict

    def __post_init__(self):
        """Validate density constraint."""
        if self.token_count > CORRECTED_MAX_SHARD_TOKENS:
            raise ValueError(
                f"Shard {self.id} exceeds H_max constraint: "
                f"{self.token_count} > {CORRECTED_MAX_SHARD_TOKENS} tokens"
            )


class DensityOptimizer:
    """
    Applies information density constraints to shard formation.

    Implements the H_max constraint from Mathematical Model Section 8.
    """

    def __init__(self, max_tokens: int = CORRECTED_MAX_SHARD_TOKENS,
                 min_tokens: int = MIN_SHARD_TOKENS):
        """
        Initialize density optimizer.

        Args:
            max_tokens: Maximum tokens per shard (H_max)
            min_tokens: Minimum tokens per shard (avoid fragmentation)
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        # Use tiktoken for accurate token counting (matches OpenAI/Qwen tokenizers)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to simple approximation if tiktoken not available
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or approximation.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximation: ~4 characters per token for English text
            # Synthetic history uses phonotactic terms which may differ slightly
            return len(text) // 4

    def apply_density_constraints(self,
                                 boundaries: List['BoundaryCandidate'],
                                 events: List[Dict]) -> List[Shard]:
        """
        Apply H_max density constraints to form shards.

        Algorithm:
        1. Iterate through events in temporal order
        2. Accumulate events until max_tokens would be exceeded
        3. At that point, either:
           a) Use nearest tempo-normalized boundary if within tolerance
           b) Force boundary to respect H_max constraint
        4. Ensure minimum shard size to avoid over-fragmentation

        Args:
            boundaries: Candidate boundaries from tempo-normalization
            events: All events sorted by timestamp

        Returns:
            List of shards with density constraints enforced
        """
        shards = []
        shard_start_idx = 0
        current_tokens = 0
        boundary_indices = {b.index for b in boundaries}

        for i, event in enumerate(events):
            event_tokens = self.count_tokens(event.get('content', ''))

            # Check if adding this event would exceed H_max
            if current_tokens + event_tokens > self.max_tokens:
                # Must create boundary here
                if current_tokens >= self.min_tokens:
                    shard = self._create_shard(
                        events=events[shard_start_idx:i],
                        shard_id=len(shards),
                        boundary_type="density_forced"
                    )
                    shards.append(shard)
                    shard_start_idx = i
                    current_tokens = event_tokens
                else:
                    # Current shard too small, but adding event exceeds max
                    # This means individual event is too large
                    # Apply chunking strategy
                    if event_tokens > self.max_tokens:
                        # Single event exceeds max - need chunking
                        # For now, create shard with what we have and continue
                        if current_tokens > 0:
                            shard = self._create_shard(
                                events=events[shard_start_idx:i],
                                shard_id=len(shards),
                                boundary_type="density_forced"
                            )
                            shards.append(shard)

                        # Create oversized shard with warning
                        shard = self._create_shard(
                            events=[event],
                            shard_id=len(shards),
                            boundary_type="oversized_event",
                            allow_oversized=True
                        )
                        shards.append(shard)
                        shard_start_idx = i + 1
                        current_tokens = 0
                    else:
                        # Force include event even though it pushes over limit
                        current_tokens += event_tokens
            else:
                current_tokens += event_tokens

                # Check if this is a tempo-normalized boundary point
                if i in boundary_indices:
                    # Validate boundary respects minimum shard size
                    if current_tokens >= self.min_tokens:
                        shard = self._create_shard(
                            events=events[shard_start_idx:i],
                            shard_id=len(shards),
                            boundary_type="tempo_normalized"
                        )
                        shards.append(shard)
                        shard_start_idx = i
                        current_tokens = event_tokens
                    # else: skip boundary (shard too small)

        # Handle remaining events
        if shard_start_idx < len(events):
            shard = self._create_shard(
                events=events[shard_start_idx:],
                shard_id=len(shards),
                boundary_type="final"
            )
            shards.append(shard)

        return shards

    def _create_shard(self, events: List[Dict], shard_id: int,
                     boundary_type: str, allow_oversized: bool = False) -> Shard:
        """
        Create a shard from events.

        Args:
            events: Events to include in shard
            shard_id: Shard identifier
            boundary_type: Type of boundary ("tempo_normalized", "density_forced", etc.)
            allow_oversized: Allow shard to exceed max_tokens (for special cases)

        Returns:
            Shard object
        """
        if not events:
            raise ValueError("Cannot create shard with zero events")

        # Calculate token count
        total_tokens = sum(self.count_tokens(e.get('content', '')) for e in events)

        # Extract time range
        timestamps = [e.get('timestamp', '') for e in events]
        time_range = (min(timestamps), max(timestamps))

        # Create shard ID with time range embedded
        start_year_month = time_range[0][:7] if time_range[0] else "unknown"
        end_year_month = time_range[1][:7] if time_range[1] else "unknown"
        shard_name = f"shard_{shard_id:03d}_{start_year_month}_{end_year_month}"

        # Causal context will be populated later by causal_aware_partitioner
        shard = Shard(
            id=shard_name,
            events=events,
            time_range=time_range,
            token_count=total_tokens,
            causal_ancestors=[],
            causal_descendants=[],
            boundary_signals={
                "type": boundary_type,
                "token_count": total_tokens,
                "event_count": len(events)
            }
        )

        # Validate constraint (unless explicitly allowing oversized)
        if not allow_oversized and total_tokens > self.max_tokens:
            raise ValueError(
                f"Shard {shard_name} violates H_max: "
                f"{total_tokens} > {self.max_tokens} tokens"
            )

        if total_tokens > self.max_tokens:
            print(f"WARNING: Shard {shard_name} is oversized: {total_tokens} tokens")

        return shard
