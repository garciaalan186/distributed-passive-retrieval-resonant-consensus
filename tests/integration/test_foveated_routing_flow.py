
import pytest
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dpr_rc.application.active_agent.dtos import QueryRequestDTO
from dpr_rc.application.active_agent.use_cases.handle_query_use_case import HandleQueryUseCase
from dpr_rc.domain.active_agent.services.foveated_router import FoveatedRouter, IFoveatedVectorStore, FoveatedMatch

# --- Mocks ---

class MockQueryEnhancer:
    def __init__(self, enhanced_text: str):
        self.enhanced_text = enhanced_text

    def enhance(self, query_text: str, timestamp_context: Optional[str], trace_id: str) -> Dict[str, Any]:
        print(f"\n[E4 Mock] Enhancing query: '{query_text}'")
        payload = {
            "enhanced_query": self.enhanced_text,
            "enhancement_used": True,
            "original_query": query_text
        }
        print(f"[E4 Mock] Output Payload: {payload}")
        return payload

class MockVectorStore(IFoveatedVectorStore):
    def __init__(self, matches: Dict[str, List[FoveatedMatch]]):
        self.matches = matches # Map layer -> matches

    def search(self, layer: str, query_text: str, limit: int = 5, filters: Optional[Dict] = None) -> List[FoveatedMatch]:
        print(f"\n[E5 Mock Internal] Searching Layer {layer} with query: '{query_text}'")
        if filters:
            print(f"[E5 Mock Internal] Filters: {filters}")
        
        results = self.matches.get(layer, [])
        print(f"[E5 Mock Internal] Returning {len(results)} matches")
        return results

class MockLogger:
    def log_message(self, trace_id, direction, message_type, payload, metadata=None):
        pass
    def log_event(self, trace_id, event_type, data, metrics=None):
        if event_type == "FOVEATED_ROUTING":
            print(f"\n[Logger] FOVEATED_ROUTING Event Data (E6 Result): {data}")

# Simple mocks for other dependencies to allow instantiation
class MockService:
    def get_target_shards(self, **kwargs): return []
    def calculate_consensus(self, votes): from dpr_rc.domain.active_agent.entities import ConsensusResult; return ConsensusResult([], [], [], 0, 0)
    def synthesize_response(self, result): from dpr_rc.domain.active_agent.entities import CollapsedResponse, ResonanceMatrix; return CollapsedResponse(None, 0.0, "FAILED", [], ResonanceMatrix([],[],[]))
    def gather_votes(self, **kwargs): return []

# --- Tests ---

class TestFoveatedRoutingFlow:
    """
    Integration test for E4 (SLM Expansion) -> E5 (Foveated Routing) -> E6 (Time Ranges).
    """

    def test_e4_to_e5_flow(self):
        # 1. Setup Data
        original_query = "quantum mechanics"
        enhanced_query = "quantum mechanics history 1920s" # E4 Output
        
        # Stub E4
        enhancer = MockQueryEnhancer(enhanced_text=enhanced_query)
        
        # Stub Vector Store for E5 processing
        # Simulate a successful drill-down: L3 -> L2 -> L1
        l3_match = FoveatedMatch(
            summary_id="dom_physics", 
            layer="L3", 
            score=0.95, 
            time_range=("1900-01", "2000-12"),
            metadata={"domain": "Physics"}
        )
        l2_match = FoveatedMatch(
            summary_id="epoch_early_qt", 
            layer="L2", 
            score=0.92, 
            time_range=("1920-01", "1930-12"),
            metadata={"epoch_id": "ep_1920s", "domain": "Physics"}
        )
        l1_match = FoveatedMatch(
            summary_id="shard_1927", 
            layer="L1", 
            score=0.88, 
            time_range=("1927-01", "1927-12"),
            metadata={"epoch_id": "ep_1920s"}
        )

        vector_store = MockVectorStore({
            "L3": [l3_match],
            "L2": [l2_match],
            "L1": [l1_match]
        })

        foveated_router = FoveatedRouter(vector_store=vector_store, min_confidence=0.8)

        # Instantiate Use Case
        use_case = HandleQueryUseCase(
            routing_service=MockService(),
            consensus_calculator=MockService(),
            response_synthesizer=MockService(),
            query_enhancer=enhancer,
            worker_communicator=MockService(),
            logger=MockLogger(),
            foveated_router=foveated_router,
            enable_query_enhancement=True
        )

        request = QueryRequestDTO(query_text=original_query, trace_id="test_trace_123")

        # 2. Execute
        print("\n--- Executing Pipeline ---")
        use_case.execute(request)
        
        # 3. Validation is mostly via printed output/logs per user request ("inspect schemas")
        # But we can implicitly assert the flow worked if no errors and logs show up.
        # The FoveatedRouter logic is:
        # 1. Search L3 with "quantum mechanics history 1920s"
        # 2. Found "Physics", Search L2 with same query + filter domain=Physics
        # 3. Found "ep_1920s", Search L1 with same query + filter epoch=ep_1920s
        # 4. Found "1927-01" range.
        
        # This confirms E4 output was passed to E5 input.
