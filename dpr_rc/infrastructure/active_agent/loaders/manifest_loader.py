"""
Infrastructure: Manifest Loader

Loads shard manifest and causal index from GCS.
"""

import json
import tempfile
import os
from typing import Optional, Dict
from dpr_rc.logging_utils import StructuredLogger, ComponentType


class ManifestLoader:
    """Loads routing manifests from GCS."""

    def __init__(self, bucket_name: Optional[str], scale: str = "medium"):
        self.bucket_name = bucket_name
        self.scale = scale
        self.logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

    def load_manifests(self) -> tuple[Optional[Dict], Optional[Dict]]:
        """
        Load shard manifest and causal index from GCS.

        Returns:
            Tuple of (manifest, causal_index) or (None, None) on failure
        """
        if not self.bucket_name:
            self.logger.logger.warning("HISTORY_BUCKET not set, skipping manifest loading")
            return None, None

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            # Load shard manifest
            manifest = None
            manifest_blob = bucket.blob(f"indices/{self.scale}/shard_manifest.json")
            if manifest_blob.exists():
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                    manifest_blob.download_to_filename(f.name)
                    f.seek(0)
                    manifest = json.load(open(f.name))
                    self.logger.logger.info(
                        f"Loaded shard manifest: {len(manifest.get('shards', []))} shards"
                    )

            # Load causal index
            causal_index = None
            causal_blob = bucket.blob(f"indices/{self.scale}/causal_index.json")
            if causal_blob.exists():
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                    causal_blob.download_to_filename(f.name)
                    f.seek(0)
                    causal_index = json.load(open(f.name))
                    self.logger.logger.info("Loaded causal index")

            return manifest, causal_index

        except Exception as e:
            self.logger.logger.error(f"Failed to load manifests: {e}")
            return None, None
