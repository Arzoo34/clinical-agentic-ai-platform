"""Base agent class from which all pipeline agents inherit."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.core.events import Event, EventBus, EventType
from app.core.logging import StructuredLogger
from app.core.models import AgentResult, PipelineContext


class BaseAgent(ABC):
    """Abstract base agent providing common logging, event publishing, and execution interface."""

    def __init__(
        self,
        name: str,
        event_bus: Optional[EventBus] = None,
        logger: Optional[StructuredLogger] = None,
        **kwargs,
    ):
        self.name = name
        self.event_bus = event_bus or EventBus()
        self.logger = logger or StructuredLogger(name)

    @abstractmethod
    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute agent task and update context."""
        pass

    def log_start(self, context: PipelineContext) -> None:
        """Log agent execution start."""
        self.logger.info(
            f"Agent started: {self.name}",
            context={"session_id": context.session_id, "agent": self.name},
        )

    def log_completion(self, context: PipelineContext, result: AgentResult) -> None:
        """Log agent execution completion."""
        self.logger.info(
            f"Agent completed: {self.name}",
            context={
                "session_id": context.session_id,
                "agent": self.name,
                "success": result.success,
                "data": result.data,
                "errors": result.errors,
                "warnings": result.warnings,
            },
        )

    def log_error(self, context: PipelineContext, error_msg: str) -> None:
        """Log agent execution error."""
        self.logger.error(
            f"Agent failed: {self.name} - {error_msg}",
            context={"session_id": context.session_id, "agent": self.name, "error": error_msg},
        )

    def publish_event(
        self,
        event_type: EventType,
        session_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish an event to the shared event bus."""
        if self.event_bus:
            event = Event(
                event_type=event_type,
                session_id=session_id,
                source_agent=self.name,
                data=data or {},
            )
            self.event_bus.publish(event)
