"""
Foveated Layer Generator.

Generates multi-resolution hierarchical summaries per DPR-RC Mathematical Model Section 3.

Compression ratios (corrected for 20K token budget):
- L_0 -> L_1: 5:1 compression (20,000 -> 4,000 tokens)
- L_1 -> L_2: 2:1 compression (4,000 -> 2,000 tokens)
- L_2 -> L_3: 2:1 compression (2,000 -> 1,000 tokens)

This ensures each layer fits within SLM context window when processing
multiple constituent nodes.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# Compression ratio targets
COMPRESSION_RATIOS = {
    'L0_to_L1': 5.0,  # 20,000 -> 4,000 tokens
    'L1_to_L2': 2.0,  # 4,000 -> 2,000 tokens
    'L2_to_L3': 2.0   # 2,000 -> 1,000 tokens
}

# Target token counts for each layer
TARGET_TOKENS = {
    'L0': 20000,  # Raw events (shard size)
    'L1': 4000,   # Shard summary
    'L2': 2000,   # Epoch summary
    'L3': 1000    # Domain summary
}


@dataclass
class FoveatedSummary:
    """A summary at a specific foveation layer."""
    layer: str  # L1, L2, or L3
    id: str
    summary: str
    token_count: int
    source_ids: List[str]  # IDs of source nodes (shards, epochs, etc.)
    time_range: Optional[tuple] = None
    metadata: Optional[Dict] = None


class LayerGenerator:
    """
    Generates foveated summarization layers using SLM.
    """

    def __init__(self, slm_service=None):
        """
        Initialize layer generator.

        Args:
            slm_service: SLM service client for generating summaries
        """
        self.slm_service = slm_service

    def generate_l1_summaries(self,
                             shards: List[Dict],
                             target_tokens: int = TARGET_TOKENS['L1']) -> Dict[str, FoveatedSummary]:
        """
        Generate L1 (shard-level) summaries.

        Each shard (20,000 tokens of L_0 events) is summarized to 4,000 tokens.

        Args:
            shards: List of shards with events
            target_tokens: Target token count for summaries

        Returns:
            Dict mapping shard_id -> FoveatedSummary
        """
        l1_summaries = {}

        for shard in shards:
            shard_id = shard.get('id', '')
            events = shard.get('events', [])

            if not events:
                continue

            # Concatenate event contents
            event_texts = []
            for event in events:
                content = event.get('content', '')
                timestamp = event.get('timestamp', '')
                event_texts.append(f"[{timestamp}] {content}")

            full_text = "\n\n".join(event_texts)

            # Generate summary via SLM
            summary = self._generate_summary(
                content=full_text,
                target_tokens=target_tokens,
                summary_type="shard",
                source_id=shard_id
            )

            # Extract time range
            timestamps = [e.get('timestamp', '') for e in events]
            time_range = (min(timestamps), max(timestamps)) if timestamps else None

            # Extract key claims
            key_claims = self._extract_key_claims(events)

            l1_summaries[shard_id] = FoveatedSummary(
                layer='L1',
                id=f"summary_L1_{shard_id}",
                summary=summary,
                token_count=self._count_tokens(summary),
                source_ids=[shard_id],
                time_range=time_range,
                metadata={
                    'event_count': len(events),
                    'key_claims': key_claims,
                    'compression_ratio': COMPRESSION_RATIOS['L0_to_L1']
                }
            )

        return l1_summaries

    def generate_l2_summaries(self,
                             l1_summaries: Dict[str, FoveatedSummary],
                             epoch_size: int = 5,
                             target_tokens: int = TARGET_TOKENS['L2']) -> Dict[str, FoveatedSummary]:
        """
        Generate L2 (epoch-level) summaries.

        Groups multiple L1 summaries (typically 5 shards) and compresses
        5 x 4,000 = 20,000 tokens -> 2,000 tokens.

        Args:
            l1_summaries: L1 summaries to group
            epoch_size: Number of L1 summaries per epoch
            target_tokens: Target token count for summaries

        Returns:
            Dict mapping epoch_id -> FoveatedSummary
        """
        l2_summaries = {}

        # Sort L1 summaries by time
        sorted_summaries = sorted(
            l1_summaries.values(),
            key=lambda s: s.time_range[0] if s.time_range else ''
        )

        # Group into epochs
        for i in range(0, len(sorted_summaries), epoch_size):
            epoch_summaries = sorted_summaries[i:i+epoch_size]
            epoch_id = f"epoch_{i//epoch_size:03d}"

            # Concatenate L1 summaries
            combined_text = "\n\n=== Next Period ===\n\n".join(
                s.summary for s in epoch_summaries
            )

            # Generate L2 summary
            summary = self._generate_summary(
                content=combined_text,
                target_tokens=target_tokens,
                summary_type="epoch",
                source_id=epoch_id
            )

            # Compute time range
            time_ranges = [s.time_range for s in epoch_summaries if s.time_range]
            if time_ranges:
                time_range = (
                    min(tr[0] for tr in time_ranges),
                    max(tr[1] for tr in time_ranges)
                )
            else:
                time_range = None

            l2_summaries[epoch_id] = FoveatedSummary(
                layer='L2',
                id=f"summary_L2_{epoch_id}",
                summary=summary,
                token_count=self._count_tokens(summary),
                source_ids=[s.id for s in epoch_summaries],
                time_range=time_range,
                metadata={
                    'shard_count': len(epoch_summaries),
                    'compression_ratio': COMPRESSION_RATIOS['L1_to_L2']
                }
            )

        return l2_summaries

    def generate_l3_summaries(self,
                             events: List[Dict],
                             target_tokens: int = TARGET_TOKENS['L3']) -> Dict[str, FoveatedSummary]:
        """
        Generate L3 (domain-level) summaries.

        Summarizes all events in a research domain to 1,000 tokens.

        Args:
            events: All events
            target_tokens: Target token count for summaries

        Returns:
            Dict mapping domain -> FoveatedSummary
        """
        l3_summaries = {}

        # Group events by domain
        events_by_domain = {}
        for event in events:
            domain = event.get('topic', 'unknown')
            if domain not in events_by_domain:
                events_by_domain[domain] = []
            events_by_domain[domain].append(event)

        # Summarize each domain
        for domain, domain_events in events_by_domain.items():
            # Sort by timestamp
            domain_events.sort(key=lambda e: e.get('timestamp', ''))

            # Extract key developments
            key_developments = self._extract_key_developments(domain_events)

            # Generate summary
            summary = self._generate_summary(
                content="\n".join(key_developments),
                target_tokens=target_tokens,
                summary_type="domain",
                source_id=domain
            )

            timestamps = [e.get('timestamp', '') for e in domain_events]
            time_range = (min(timestamps), max(timestamps)) if timestamps else None

            l3_summaries[domain] = FoveatedSummary(
                layer='L3',
                id=f"summary_L3_{domain}",
                summary=summary,
                token_count=self._count_tokens(summary),
                source_ids=[domain],
                time_range=time_range,
                metadata={
                    'event_count': len(domain_events),
                    'time_span_years': self._compute_time_span_years(time_range),
                    'compression_ratio': COMPRESSION_RATIOS['L2_to_L3']
                }
            )

        return l3_summaries

    def _generate_summary(self,
                         content: str,
                         target_tokens: int,
                         summary_type: str,
                         source_id: str) -> str:
        """
        Generate summary using SLM or fallback method.

        Args:
            content: Content to summarize
            target_tokens: Target token count
            summary_type: Type of summary (shard, epoch, domain)
            source_id: Source identifier

        Returns:
            Summary text
        """
        if self.slm_service:
            # Use SLM for summarization
            prompt = self._build_summarization_prompt(
                content, target_tokens, summary_type
            )

            try:
                response = self.slm_service.generate_summary(
                    prompt=prompt,
                    max_tokens=target_tokens
                )
                return response.get('summary', content[:target_tokens * 4])
            except Exception as e:
                print(f"Warning: SLM summarization failed: {e}")
                # Fall through to extractive fallback

        # Fallback: Extractive summarization (first N tokens)
        return self._extractive_summary(content, target_tokens)

    def _build_summarization_prompt(self,
                                   content: str,
                                   target_tokens: int,
                                   summary_type: str) -> str:
        """Build prompt for SLM summarization."""
        prompts = {
            'shard': f"Summarize the following historical research events in approximately {target_tokens} tokens, preserving key findings and causal relationships:\n\n{content}",
            'epoch': f"Synthesize the following period summaries into a coherent {target_tokens}-token overview:\n\n{content}",
            'domain': f"Provide a comprehensive {target_tokens}-token summary of this research domain's evolution:\n\n{content}"
        }

        return prompts.get(summary_type, content)

    def _extractive_summary(self, content: str, target_tokens: int) -> str:
        """Simple extractive summarization fallback."""
        # Approximate: 4 characters per token
        target_chars = target_tokens * 4
        return content[:target_chars]

    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        # Simple approximation: 4 characters per token
        return len(text) // 4

    def _extract_key_claims(self, events: List[Dict]) -> List[str]:
        """Extract key claims from events."""
        claims = []
        for event in events:
            # Claims are stored as IDs (strings) in events
            if 'claims' in event and event['claims']:
                # For now, use event content as proxy for claims
                # TODO: Load actual claim objects if needed
                content_snippet = event.get('content', '')[:200]
                if content_snippet and content_snippet not in claims:
                    claims.append(content_snippet)
                if len(claims) >= 10:
                    break

        return claims[:10]  # Top 10 claims

    def _extract_key_developments(self, events: List[Dict]) -> List[str]:
        """Extract key developments from domain events."""
        developments = []

        # Group by year
        by_year = {}
        for event in events:
            year = event.get('timestamp', '')[:4]
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(event)

        # Extract one key event per year
        for year in sorted(by_year.keys()):
            year_events = by_year[year]
            # Pick event with most causal descendants (most impactful)
            key_event = max(
                year_events,
                key=lambda e: len(e.get('causal_parents', []))
            )
            developments.append(f"[{year}] {key_event.get('content', '')[:200]}")

        return developments

    def _compute_time_span_years(self, time_range: Optional[tuple]) -> float:
        """Compute time span in years."""
        if not time_range:
            return 0.0

        from datetime import datetime
        try:
            start = datetime.fromisoformat(time_range[0].replace('Z', '+00:00'))
            end = datetime.fromisoformat(time_range[1].replace('Z', '+00:00'))
            delta = (end - start).days / 365.25
            return round(delta, 2)
        except Exception:
            return 0.0

    def save_summaries(self, summaries: Dict[str, FoveatedSummary], filepath: str) -> None:
        """Save summaries to JSON file."""
        data = {
            layer_id: {
                'layer': summary.layer,
                'id': summary.id,
                'summary': summary.summary,
                'token_count': summary.token_count,
                'source_ids': summary.source_ids,
                'time_range': summary.time_range,
                'metadata': summary.metadata
            }
            for layer_id, summary in summaries.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
