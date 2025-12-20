"""
Infrastructure: Logger Adapter

Adapter for StructuredLogger to implement ILogger interface.
"""

from typing import Dict, Any, Optional
from dpr_rc.logging_utils import StructuredLogger, ComponentType, EventType


class LoggerAdapter:
    """
    Adapter that wraps StructuredLogger to implement ILogger protocol.
    """

    def __init__(self, logger: StructuredLogger):
        """
        Initialize adapter.

        Args:
            logger: StructuredLogger instance to wrap
        """
        self.logger = logger

    def log_message(
        self,
        trace_id: str,
        direction: str,
        message_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a message with structured metadata.

        Args:
            trace_id: Correlation ID
            direction: 'request', 'response', or 'internal'
            message_type: Type of message
            payload: Message content
            metadata: Additional metadata
        """
        self.logger.log_message(
            trace_id=trace_id,
            direction=direction,
            message_type=message_type,
            payload=payload,
            metadata=metadata or {},
        )

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        data: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an event with metrics.

        Args:
            trace_id: Correlation ID
            event_type: Type of event (converted to EventType)
            data: Event data
            metrics: Metrics to track
        """
        # Convert string to EventType enum
        try:
            event_enum = EventType[event_type]
        except KeyError:
            event_enum = EventType.VOTE_CAST  # Default fallback

        self.logger.log_event(
            trace_id=trace_id,
            event_type=event_enum,
            payload=data,
            metrics=metrics or {},
        )
