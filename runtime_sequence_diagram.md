# Current State Runtime Sequence Diagram

## Diagram

```mermaid
sequenceDiagram
    autonumber
    participant N1 as Client
    participant N2 as Active Controller (Facade)
    participant N3 as HandleQueryUseCase (Orchestrator)
    participant N4 as QueryEnhancer (SLM)
    participant N5 as RoutingService (L1)
    participant N6 as WorkerCommunicator (Infra)
    participant N7 as Passive Workers (Shards)
    participant N8 as ConsensusCalculator (L3)
    participant N9 as ResponseSynthesizer

    N1->>N2: E1: POST /query
    activate N2
    N2->>N3: E2: execute(QueryRequestDTO)
    activate N3
    
    N3->>N4: E3: enhance(query, context)
    activate N4
    N4-->>N3: E4: Validation/Expansions
    deactivate N4

    N3->>N5: E5: get_target_shards(timestamp)
    activate N5
    N5-->>N3: E6: List[shard_id]
    deactivate N5

    N3->>N6: E7: gather_votes(query, shards)
    activate N6
    N6->>N7: E8: POST /process_rfi (Broadcast)
    activate N7
    N7-->>N6: E9: List[ConsensusVote]
    deactivate N7
    N6-->>N3: E10: Aggregated Votes
    deactivate N6

    N3->>N8: E11: calculate_consensus(votes)
    activate N8
    N8-->>N3: E12: ConsensusResult (Tiers)
    deactivate N8

    N3->>N9: E13: synthesize_response(result)
    activate N9
    N9-->>N3: E14: Collapsed Response
    deactivate N9

    N3-->>N2: E15: QueryResponseDTO
    deactivate N3
    N2-->>N1: E16: RetrievalResult
    deactivate N2
```

## Legend

### Nodes (Components)
- **N1 Client**: The external user or system initiating the query.
- **N2 Active Controller (Facade)**: The FastAPI entry point (`active_agent.py`) that handles HTTP protocol details and delegates to the application layer.
- **N3 HandleQueryUseCase (Orchestrator)**: The core application logic (`handle_query_use_case.py`) that coordinates the DPR-RC pipeline steps.
- **N4 QueryEnhancer (SLM)**: The interface to the Small Language Model service used to refine and expand the raw query.
- **N5 RoutingService (L1)**: The domain service (`routing_service.py`) responsible for selecting relevant time shards based on the query timestamp.
- **N6 WorkerCommunicator (Infra)**: The infrastructure client (`worker_communicator.py`) that manages network communication with the distributed worker nodes.
- **N7 Passive Workers (Shards)**: The distributed worker instances holding the actual document shards and performing local retrieval.
- **N8 ConsensusCalculator (L3)**: The domain service (`consensus_calculator.py`) that executes the Resonant Consensus Protocol (RCP v4) equations.
- **N9 ResponseSynthesizer**: The domain service (`response_synthesizer.py`) that converts complex consensus states into a final user-facing answer.

### Edges (Interactions)
- **E1 POST /query**: The initial HTTP request containing the raw query text and timestamp context.
- **E2 execute(QueryRequestDTO)**: Passing the normalized request object into the business logic layer.
- **E3 enhance(query, context)**: Requesting the SLM to improve the query (e.g., adding synonyms, clarifying intent).
- **E4 Validation/Expansions**: Returning the "Enhanced Query" and any additional search terms.
- **E5 get_target_shards(timestamp)**: Asking the routing service which shards cover the relevant time period.
- **E6 List[shard_id]**: Returning the list of target shard identifiers (e.g., `shard_001_2015_2020`) based on simple date comparison.
- **E7 gather_votes(query, shards)**: commanding the infrastructure layer to contact the identified shards.
- **E8 POST /process_rfi (Broadcast)**: Sending the Request For Information (RFI) payload to the Passive Workers (via HTTP or Redis).
- **E9 List[ConsensusVote]**: The workers return their local findings as "Votes" (binary approval + snippets).
- **E10 Aggregated Votes**: The communicator collects all worker responses into a single list for processing.
- **E11 calculate_consensus(votes)**: Pushing the raw votes into the RCP v4 engine to determine truth/consensus.
- **E12 ConsensusResult (Tiers)**: Returning the artifacts classified into `Consensus`, `Polar`, or `Negative_Consensus` tiers.
- **E13 synthesize_response(result)**: Asking the synthesizer to format the consensus result into a readable final answer.
- **E14 Collapsed Response**: The final "Superposition State" and simple text answer.
- **E15 QueryResponseDTO**: Returning the complete domain object to the controller.
- **E16 RetrievalResult**: The final JSON HTTP response sent back to the client.
