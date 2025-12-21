#!/bin/bash
set -e

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest fakeredis httpx pytest-asyncio redis fastapi uvicorn

# Run tests
echo "Running Integration Tests..."
export PYTHONPATH=$PYTHONPATH:.
python3 -m pytest tests/integration/test_consensus_synthesis.py tests/integration/test_foveated_routing_flow.py -v -s
