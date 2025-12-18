import logging
import json
import os
import hashlib
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger
from .models import LogEntry, ComponentType, EventType

# Configuration for full payload logging
ENABLE_FULL_PAYLOAD_LOGGING = os.getenv("ENABLE_FULL_PAYLOAD_LOGGING", "true").lower() == "true"
MAX_PAYLOAD_SIZE_BYTES = int(os.getenv("MAX_PAYLOAD_SIZE_BYTES", "100000"))

# Initialize Google Cloud Logging Client (optional)
# Only attempt GCP logging if USE_GCP_LOGGING is set
if os.getenv("USE_GCP_LOGGING", "").lower() in ("true", "1", "yes"):
    try:
        from google.cloud import logging as google_logging
        client = google_logging.Client()
        client.setup_logging()
    except Exception:
        # Fallback to standard logging if not on GCP or no creds
        pass

class DPRJSONFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(DPRJSONFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # use valid timestamp
            import time
            log_record['timestamp'] = time.time()
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

def get_logger(name: str):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = DPRJSONFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Prevent duplicate logs if propagtion is on
    logger.propagate = False
    return logger

class StructuredLogger:
    def __init__(self, component: ComponentType):
        self.logger = get_logger(component.value)
        self.component = component

    def hash_payload(self, payload: Any) -> str:
        """Create a hash of the payload for audit."""
        dumped = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(dumped.encode()).hexdigest()[:16]

    def log_event(self,
                  trace_id: str,
                  event_type: EventType,
                  payload: Any,
                  metrics: Dict[str, Any] = None):

        payload_hash = self.hash_payload(payload)

        entry = LogEntry(
            trace_id=trace_id,
            component=self.component,
            event_type=event_type,
            payload_hash=payload_hash,
            metrics=metrics or {},
            message=str(payload)[:200] # Log a snippet for debug
        )

        self.logger.info(json.dumps(entry.model_dump(), default=str))

    def log_message(self,
                    trace_id: str,
                    direction: str,
                    message_type: str,
                    payload: Dict[str, Any],
                    metadata: Optional[Dict] = None):
        """
        Log full message content with trace correlation.

        Args:
            trace_id: Trace ID for correlation
            direction: "request" | "response" | "internal"
            message_type: Descriptive message type (e.g., "client_query", "worker_rfi", "slm_verify")
            payload: Full message payload (request or response)
            metadata: Additional metadata (e.g., endpoint, worker_url, timing)
        """
        if not ENABLE_FULL_PAYLOAD_LOGGING:
            # Fall back to hash-only logging
            self.logger.info(json.dumps({
                "trace_id": trace_id,
                "component": self.component.value,
                "direction": direction,
                "message_type": message_type,
                "payload_hash": self.hash_payload(payload),
                "metadata": metadata or {}
            }, default=str))
            return

        # Serialize payload
        payload_str = json.dumps(payload, default=str)
        payload_size = len(payload_str.encode('utf-8'))

        # Truncate if exceeds max size
        truncated = False
        if payload_size > MAX_PAYLOAD_SIZE_BYTES:
            # Truncate to max size
            payload_str = payload_str[:MAX_PAYLOAD_SIZE_BYTES]
            truncated = True

        # Construct log entry
        log_entry = {
            "trace_id": trace_id,
            "component": self.component.value,
            "direction": direction,
            "message_type": message_type,
            "content_size_bytes": payload_size,
            "truncated": truncated,
            "metadata": metadata or {}
        }

        # Add payload based on direction
        if direction == "request":
            log_entry["request_payload"] = payload
        elif direction == "response":
            log_entry["response_payload"] = payload
        else:  # internal
            log_entry["internal_payload"] = payload

        self.logger.info(json.dumps(log_entry, default=str))

