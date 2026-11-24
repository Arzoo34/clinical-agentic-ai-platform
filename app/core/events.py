"""Event bus and event models for pipeline communication."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    """Pipeline event types."""
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    FILE_INGESTED = "file_ingested"
    SCHEMA_HARMONIZED = "schema_harmonized"
    QUALITY_ASSESSED = "quality_assessed"
    OPERATIONS_EVALUATED = "operations_evaluated"
    ANOMALY_DETECTED = "anomaly_detected"
    RISK_CALCULATED = "risk_calculated"
    INSIGHT_GENERATED = "insight_generated"
    ALERT_CREATED = "alert_created"


@dataclass
class Event:
    """An event emitted by an agent or orchestrator."""
    event_type: EventType
    session_id: str
    source_agent: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventBus:
    """Publish-subscribe event bus with history retention."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._history: List[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe a callback handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers and record in history."""
        self._history.append(event)
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass

    def get_events(self, event_type: Optional[EventType] = None) -> List[Event]:
        """Retrieve events, optionally filtered by event type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]

    def clear(self) -> None:
        """Clear event history."""
        self._history.clear()
