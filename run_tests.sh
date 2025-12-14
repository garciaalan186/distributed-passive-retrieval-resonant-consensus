#!/bin/bash
set -e

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest fakeredis httpx pytest-asyncio redis fastapi uvicorn

# Run tests
echo "Running Unit Tests..."
python3 -m pytest tests/test_active_unit.py tests/test_passive_unit.py -v -s

echo "Running Integration Tests..."
# specialized flag or separate run for integration because it might require local redis
python3 -m pytest tests/test_integration_local.py -v
