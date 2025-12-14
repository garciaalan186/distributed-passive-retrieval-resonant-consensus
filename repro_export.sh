#!/bin/bash
# repro_export.sh

EXPORT_DIR="reproducibility_package"
mkdir -p $EXPORT_DIR

echo "Exporting Reproducibility Package..."

# 1. Export Codebase
echo "Copying Code..."
rsync -av --exclude '__pycache__' --exclude 'reproducibility_package' --exclude '.git' \
    dpr_rc benchmark Dockerfile infrastructure.sh \
    $EXPORT_DIR/

# 2. Export Artifacts
echo "Copying Results..."
cp -f benchmark_results.json $EXPORT_DIR/ 2>/dev/null || echo "No benchmark results found."
cp -f dpr_rc_benchmark_phonotactic.json $EXPORT_DIR/ 2>/dev/null || echo "No history data found."

# 3. Export Logs (Using gcloud if available, else local file)
# If local json logs exist:
# Note: Python `python-json-logger` writes to stdout/stderr usually, 
# but if we configured a file handler we would copy it.
# We will assume users pipe output to a file: `python benchmark.py > run.log`
# We provide a placeholder instruction.

echo "To complete the package, run the benchmark and save output:"
echo "  python -m benchmark.benchmark > $EXPORT_DIR/execution.log 2>&1"

# 4. Generate Checksum
find $EXPORT_DIR -type f -exec shasum -a 256 {} \; > $EXPORT_DIR/MANIFEST.sha256

echo "Package created at $EXPORT_DIR"
