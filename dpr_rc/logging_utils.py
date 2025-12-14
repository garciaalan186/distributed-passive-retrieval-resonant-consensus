import logging
import json
import os
from typing import Any, Dict, Optional
import xxhash
from pythonjsonlogger import jsonlogger
from .models import LogEntry, ComponentType, EventType

# Skip GCP logging entirely - use standard logging only
# GCP logging can be enabled in production via environment variable
_USE_GCP_LOGGING = os.getenv("USE_GCP_LOGGING", "false").lower() == "true"

if _USE_GCP_LOGGING:
    try:
        from google.cloud import logging as google_logging
        client = google_logging.Client()
        client.setup_logging()
    except Exception:
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
        """Create a cryptographic hash of the payload for audit."""
        dumped = json.dumps(payload, sort_keys=True, default=str)
        return xxhash.xxh64(dumped).hexdigest()

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
