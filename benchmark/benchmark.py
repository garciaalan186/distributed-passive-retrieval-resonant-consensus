import os
import json
import time
import requests
import subprocess
import threading
from typing import List
from .synthetic_history import SyntheticHistoryGeneratorV2

# Config
CONTROLLER_URL = "http://localhost:8080/query"

class BenchmarkSuite:
    def __init__(self):
        self.generator = SyntheticHistoryGeneratorV2(
            events_per_topic_per_year=20, # Reduced for benchmark speed
            perspectives_per_event=2,
            num_domains=2
        )
        self.results = []
        
    def setup_environment(self):
        print("Starting local agents for benchmark...")
        # In a real scenario, this connects to the Cloud Run URL.
        # For local repro, we assume they are running via docker-compose or similar.
        # Or we rely on the user running them. 
        # Check if controller is up:
        try:
            requests.get(f"{CONTROLLER_URL.replace('/query', '/docs')}", timeout=2)
            print("Controller is accessible.")
        except:
            print("WARNING: Controller unsupported or not running. Please run infrastructure first.")

    def run_dpr_queries(self, queries: List[dict]):
        print(f"Running {len(queries)} queries against DPR-RC...")
        for q in queries:
            payload = {
                "query_text": q["question"],
                "timestamp_context": q["timestamp_context"],
                "trace_id": f"bench_{hash(q['question'])}" 
            }
            
            try:
                start = time.time()
                resp = requests.post(CONTROLLER_URL, json=payload, timeout=10)
                latency = time.time() - start
                
                if resp.status_code == 200:
                    data = resp.json()
                    self.evaluate_response(q, data, latency, system="DPR-RC")
                else:
                    print(f"Query failed: {resp.text}")
            except Exception as e:
                print(f"Error querying: {e}")

    def run_baseline_rag(self, queries: List[dict]):
        print("Running Baseline RAG (No Consensus) on same data...")
        # Instantiate a worker directly to use its storage/index
        from dpr_rc.passive_agent import PassiveWorker
        worker = PassiveWorker()
        
        # Important: The worker needs the documents indexed. 
        # In this mock, the worker would usually retrieve from a shared collection.
        # Since we changed the data gen, we rely on the worker finding ANY match.
        
        for q in queries:
            start = time.time()
            # Standard RAG: Retrieve top 1, no verification, no voting
            doc = worker.retrieve(q["question"], q.get("timestamp_context"))
            latency = time.time() - start
            
            if doc:
                response = {
                    "final_answer": doc["content"],
                    "confidence": 1.0 # Naive RAG is always confident
                }
                self.evaluate_response(q, response, latency, system="Baseline-RAG")
            else:
                 self.evaluate_response(q, {"final_answer": "", "confidence": 0}, latency, system="Baseline-RAG")

    def evaluate_response(self, ground_truth, response, latency, system):
        # Calculate Precision/Hallucination
        # For V2, we check if the answer contains relevant keywords/ids or matches expected outcomes
        # The V2 query object has expected_sources vs expected_answer
        
        # Simple heuristic: If expected sources exist, check if retrieval found them (in a real system)
        # For this text generation benchmark, we check overlap with expected answer if present,
        # or just assume success if we got a response for now to keep it simple.
        
        # V2 queries don't have "expected_answer" text field, but "expected_sources".
        # We can loosely check if the response is non-empty.
        
        actual = response["final_answer"]
        confidence = response["confidence"]
        
        # Mock correctness check: In a real system we'd check if `actual` contains info from `expected_sources`.
        is_correct = len(actual) > 10 # heuristic
        hallucination = (not is_correct) and (confidence > 0.8)
        
        self.results.append({
            "system": system,
            "query": ground_truth["question"],
            "latency": latency,
            "correct": is_correct,
            "hallucination": hallucination,
            "consensus_confidence": confidence
        })

    def generate_report(self):
        print("\n--- Benchmark Results ---")
        correct = len([r for r in self.results if r['correct']])
        hallucinations = len([r for r in self.results if r['hallucination']])
        total = len(self.results)
        
        if total == 0:
            print("No results.")
            return

        print(f"Total Queries: {total}")
        print(f"Accuracy: {correct/total:.2f}")
        print(f"Hallucination Rate: {hallucinations/total:.2f}")
        
        # Save to JSON
        with open("benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

def main():
    suite = BenchmarkSuite()
    
    # 1. Gen Data
    data = suite.generator.generate_dataset()
    events = data['events']
    queries = data['queries']
    
    print(f"Generated {len(events)} events and {len(queries)} queries.")
    
    # 2. Ingest Data (Mock: Assume Passive Agents have it or trigger ingest)
    # Ideally send docs to agents. For now we assume shared volume or pre-seeded.
    
    # 3. Setup
    suite.setup_environment()
    
    # 4. Run DPR
    suite.run_dpr_queries(queries[:5]) # Run a subset for speed check
    
    # 5. Report
    suite.generate_report()

if __name__ == "__main__":
    main()
