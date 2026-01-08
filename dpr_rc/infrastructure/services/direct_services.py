"""
Direct Service Implementations
Authentication: None (In-Process)
Transport: Method Calls (No HTTP)

These implementations are used for local benchmarking and testing where
starting separate microservices is not desired. They implement the same
interfaces as the HTTP services but call the logic directly.
"""

import os
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from dpr_rc.application.interfaces import ISLMService, IWorkerService
from dpr_rc.models import ConsensusVote
from dpr_rc.infrastructure.slm import SLMFactory
from dpr_rc.infrastructure.passive_agent import PassiveAgentFactory
from dpr_rc.application.passive_agent import ProcessRFIRequest

class DirectSLMService(ISLMService):
    """
    In-process implementation of SLM service.
    Calls the InferenceEngine directly.

    Supports mixed model mode:
    - Fast model (Qwen2-0.5B) for query enhancement (SLM_FAST_MODEL)
    - Accurate model (Phi-3) for verification (SLM_MODEL)

    In multi-GPU mode, each call gets the engine from the factory based on
    the current thread's GPU context (set by SLMFactory.set_gpu_context()).
    """
    def __init__(self):
        self._initialized = False
        self._fast_initialized = False

    @property
    def engine(self):
        """
        Get main inference engine from factory (for verification).

        The factory handles all caching - we don't cache here to support
        proper GPU context switching in multi-GPU mode.
        """
        if not self._initialized:
            print("Initializing DirectSLMService engine...")
            self._initialized = True
        return SLMFactory.get_engine()

    @property
    def fast_engine(self):
        """
        Get fast inference engine from factory (for enhancement).

        In mixed model mode, this returns a lightweight model optimized
        for speed. Falls back to main engine if no fast model configured.
        """
        if not self._fast_initialized:
            import os
            fast_model = os.getenv("SLM_FAST_MODEL")
            if fast_model:
                print(f"Initializing DirectSLMService fast engine ({fast_model})...")
            self._fast_initialized = True
        return SLMFactory.get_fast_engine()

    def enhance_query(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance query using fast engine (in mixed model mode).

        Uses SLM_FAST_MODEL for speed if configured, otherwise falls
        back to main model.
        """
        try:
            # Use fast engine for enhancement (or main engine if not configured)
            result = self.fast_engine.enhance_query(
                query=query_text,
                timestamp_context=timestamp_context
            )
            return {
                "original_query": result["original_query"],
                "enhanced_query": result["enhanced_query"],
                "expansions": result["expansions"],
                "enhancement_used": True,
                "inference_time_ms": result["inference_time_ms"]
            }
        except Exception as e:
            print(f"Direct SLM enhancement failed: {e}")
            return {
                "original_query": query_text,
                "enhanced_query": query_text,
                "expansions": [],
                "enhancement_used": False
            }

class DirectWorkerService(IWorkerService):
    """
    In-process implementation of Worker service.
    Calls Passive Agent Use Case directly, simulating correct sharding.
    """
    def __init__(self):
        # Cache of use cases keyed by cluster_id
        self._use_cases = {}

        # Tier 3B: Thread pool for multi-GPU parallel shard processing
        self._thread_pool = None
        self._pool_lock = threading.Lock()
        
    def _get_use_case(self, epoch_year: int):
        """Get or create a use case for the specific epoch."""

        # Determine strict sharding assignment
        if epoch_year >= 2020:
            cluster_id = "C_RECENT"
            worker_id = "worker-recent-01"
        else:
            cluster_id = "C_OLDER"
            worker_id = "worker-older-01"

        if cluster_id in self._use_cases:
            return self._use_cases[cluster_id]

        # Create use case via factory
        slm_url = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        use_case = PassiveAgentFactory.create_process_rfi_use_case(
            slm_url=slm_url,
            worker_id=worker_id,
            cluster_id=cluster_id,
            embedding_model=embedding_model,
            default_epoch_year=epoch_year
        )

        self._use_cases[cluster_id] = use_case
        return use_case

    def _get_use_case_for_shard(self, shard_id: str):
        """Get or create use case for a specific shard (used in thread pool)."""
        # Extract year from shard_id
        try:
            parts = shard_id.split("_")
            if len(parts) == 2:  # Simple format: shard_YYYY
                epoch_year = int(parts[1])
            elif len(parts) >= 3:  # Tempo-normalized: shard_XXX_YYYY-MM_YYYY-MM
                start_date = parts[-2]  # e.g., "2015-01"
                epoch_year = int(start_date.split("-")[0])
            else:
                epoch_year = 2020  # Default fallback
        except (ValueError, IndexError):
            epoch_year = 2020  # Default fallback on parsing error

        return self._get_use_case(epoch_year)

    def _get_or_create_thread_pool(self):
        """Lazy initialization of thread pool with thread safety."""
        if self._thread_pool is None:
            with self._pool_lock:
                if self._thread_pool is None:  # Double-check locking
                    num_workers = int(os.getenv("NUM_WORKER_THREADS", "6"))
                    self._thread_pool = ThreadPoolExecutor(
                        max_workers=num_workers,
                        thread_name_prefix="SLM-Worker"
                    )
        return self._thread_pool

    async def gather_votes(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str] = None
    ) -> List[ConsensusVote]:
        """
        Gather votes by calling local use case.
        Simulates distributed retrieval by dispatching to the correct worker based on shard.

        Tier 3B Optimization: ThreadPoolExecutor multi-GPU parallel shard processing.
        """
        import os

        # Tier 3B: Check if multi-GPU parallel processing is enabled
        enable_multi_gpu = os.getenv("ENABLE_MULTI_GPU_WORKERS", "false").lower() == "true"

        if enable_multi_gpu:
            return await self._gather_votes_parallel_multi_gpu(
                trace_id, query_text, original_query, target_shards, timestamp_context
            )
        else:
            # Sequential processing (baseline)
            return await self._gather_votes_sequential(
                trace_id, query_text, original_query, target_shards, timestamp_context
            )

    async def _gather_votes_sequential(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str]
    ) -> List[ConsensusVote]:
        """Sequential shard processing (original implementation for rollback)"""
        all_votes = []

        # Dispatch per shard to respect sharding/clustering architecture
        for shard_id in target_shards:
            # Extract year from shard_id
            # Handles: "shard_YYYY" or "shard_XXX_YYYY-MM_YYYY-MM" (tempo-normalized)
            try:
                parts = shard_id.split("_")
                if len(parts) == 2:  # Simple format: shard_YYYY
                    epoch_year = int(parts[1])
                elif len(parts) >= 3:  # Tempo-normalized: shard_XXX_YYYY-MM_YYYY-MM
                    # Extract start year from second-to-last part
                    start_date = parts[-2]  # e.g., "2015-01"
                    epoch_year = int(start_date.split("-")[0])
                else:
                    epoch_year = 2020  # Default fallback
            except (ValueError, IndexError):
                epoch_year = 2020  # Default fallback on parsing error

            use_case = self._get_use_case(epoch_year)

            # Create request DTO for this specific shard
            request = ProcessRFIRequest(
                trace_id=trace_id,
                query_text=query_text,
                original_query=original_query,
                target_shards=[shard_id], # Process one shard at a time with correct worker
                timestamp_context=timestamp_context or ""
            )

            try:
                # Execute use case in a separate thread to avoid blocking the event loop
                # This is critical for local benchmarks where SLM/retrieval are CPU intensive
                response = await asyncio.to_thread(use_case.execute, request)

                # Convert dicts back to ConsensusVote objects
                for v_data in response.votes:
                    try:
                        vote = ConsensusVote(**v_data)
                        all_votes.append(vote)
                    except Exception as e:
                        pass # Failed to parse vote

            except Exception as e:
                pass # Direct worker execution failed

        return all_votes

    async def _gather_votes_parallel_multi_gpu(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str]
    ) -> List[ConsensusVote]:
        """
        Process shards in parallel using ThreadPoolExecutor (Tier 3B).
        Distributes shard processing across multiple GPUs.
        """
        pool = self._get_or_create_thread_pool()
        loop = asyncio.get_event_loop()

        # Create tasks for each shard with GPU assignment
        tasks = []
        for idx, shard_id in enumerate(target_shards):
            gpu_id = idx % 2  # Alternate between GPU 0 and 1

            task = loop.run_in_executor(
                pool,
                self._process_shard_sync,
                shard_id,
                gpu_id,
                trace_id,
                query_text,
                original_query,
                timestamp_context
            )
            tasks.append(task)

        # Wait for all shards to complete
        results = await asyncio.gather(*tasks)

        # Flatten votes from all shards
        all_votes = []
        for votes in results:
            all_votes.extend(votes)

        return all_votes

    def _process_shard_sync(
        self,
        shard_id: str,
        gpu_id: int,
        trace_id: str,
        query_text: str,
        original_query: str,
        timestamp_context: Optional[str] = None
    ) -> List[ConsensusVote]:
        """
        Synchronous shard processing for thread pool execution.

        This method runs in a worker thread and processes a single shard
        using a GPU-pinned SLM instance from the GPU pool.
        """
        # Set GPU context for this thread - all SLMFactory.get_engine() calls
        # in this thread will use the engine for this GPU
        SLMFactory.set_gpu_context(gpu_id)

        # Get or create use case for this shard
        # The use case will create DirectSLMClient which will use
        # SLMFactory.get_engine(), which returns the GPU-specific engine
        use_case = self._get_use_case_for_shard(shard_id)

        # Create RFI request
        request = ProcessRFIRequest(
            query_text=query_text,
            original_query=original_query,
            target_shards=[shard_id],
            trace_id=trace_id,
            timestamp_context=timestamp_context or ""
        )

        try:
            # Execute (synchronous call in thread)
            response = use_case.execute(request)

            # Convert dicts to ConsensusVote objects
            votes = []
            for v_data in response.votes:
                try:
                    vote = ConsensusVote(**v_data)
                    votes.append(vote)
                except Exception as e:
                    pass  # Failed to parse vote

            return votes

        except Exception as e:
            return []  # Execution failed

    async def gather_responses(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Gather response artifacts from A_h agents.

        Each A_h generates a full text response based on their complete
        shard context. These become artifacts (ω) for cross-voting.
        """
        all_responses = []

        # Dispatch per shard
        for shard_id in target_shards:
            # Extract year for worker assignment
            try:
                parts = shard_id.split("_")
                if len(parts) == 2:
                    epoch_year = int(parts[1])
                elif len(parts) >= 3:
                    start_date = parts[-2]
                    epoch_year = int(start_date.split("-")[0])
                else:
                    epoch_year = 2020
            except (ValueError, IndexError):
                epoch_year = 2020

            use_case = self._get_use_case(epoch_year)

            # Create request for RESPONSE_GENERATION phase
            request = ProcessRFIRequest(
                trace_id=trace_id,
                query_text=query_text,
                original_query=original_query,
                target_shards=[shard_id],
                timestamp_context=timestamp_context or "",
                phase="RESPONSE_GENERATION"  # NEW: Phase indicator
            )

            try:
                # Execute in thread
                response = await asyncio.to_thread(use_case.execute, request)

                # Extract the artifact response from votes
                if response.votes:
                    vote_data = response.votes[0]  # Should be one vote per shard
                    artifact = {
                        "artifact_id": f"{shard_id}_{vote_data['content_hash']}",
                        "author_agent": vote_data['worker_id'],
                        "response_text": vote_data.get('artifact_response', vote_data['content_snippet']),
                        "source_shard": shard_id,
                        "content_hash": vote_data['content_hash'],
                        "author_cluster": vote_data.get('author_cluster', vote_data['cluster_id'])
                    }
                    all_responses.append(artifact)

            except Exception as e:
                pass  # Failed to get response from this shard

        return all_responses

    async def gather_votes_on_artifact(
        self,
        trace_id: str,
        artifact: Dict[str, Any],
        voting_shards: List[str],
        original_query: str
    ) -> List[ConsensusVote]:
        """
        Phase 2: Gather votes on a specific artifact from all A_h agents.

        Each A_h evaluates the artifact and votes on it. This implements
        the cross-voting matrix.
        """
        all_votes = []

        # Each shard votes on this artifact
        for shard_id in voting_shards:
            # Extract year for worker assignment
            try:
                parts = shard_id.split("_")
                if len(parts) == 2:
                    epoch_year = int(parts[1])
                elif len(parts) >= 3:
                    start_date = parts[-2]
                    epoch_year = int(start_date.split("-")[0])
                else:
                    epoch_year = 2020
            except (ValueError, IndexError):
                epoch_year = 2020

            use_case = self._get_use_case(epoch_year)

            # Create request for VOTING phase
            request = ProcessRFIRequest(
                trace_id=trace_id,
                query_text="",  # Not needed for voting
                original_query=original_query,
                target_shards=[shard_id],
                timestamp_context="",
                phase="VOTING",  # NEW: Voting phase
                artifact_to_vote_on=artifact  # NEW: The artifact to evaluate
            )

            try:
                # Execute in thread
                response = await asyncio.to_thread(use_case.execute, request)

                # Convert vote dicts to ConsensusVote objects
                for v_data in response.votes:
                    try:
                        vote = ConsensusVote(**v_data)
                        all_votes.append(vote)
                    except Exception as e:
                        pass

            except Exception as e:
                pass  # Failed to get vote from this shard

        return all_votes
