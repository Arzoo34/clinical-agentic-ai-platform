"""Pipeline orchestrator that coordinates all agents."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.agents.alert_agent import AlertAgent
from app.agents.anomaly_agent import AnomalyAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.insight_agent import InsightAgent
from app.agents.operational_agent import OperationalAgent
from app.agents.quality_agent import QualityAgent
from app.agents.risk_agent import RiskAgent
from app.agents.schema_agent import SchemaAgent
from app.core.config import ConfigLoader
from app.core.events import EventBus, EventType
from app.core.logging import StructuredLogger
from app.core.models import PipelineContext
from app.core.storage import Storage
from app.llm.insight import LocalInsightGenerator


class PipelineOrchestrator:
    """Orchestrates the multi-agent clinical trial intelligence pipeline."""

    def __init__(
        self,
        config_path: Path = Path("config/config.yaml"),
        data_base_path: Path = Path("data"),
        log_path: Optional[Path] = None,
    ):
        """Initialize orchestrator.

        Args:
            config_path: Path to YAML configuration file.
            data_base_path: Base path for data storage.
            log_path: Path for structured logs.
        """
        self.config = ConfigLoader(config_path)
        self.storage = Storage(data_base_path)
        self.event_bus = EventBus()
        self.logger = StructuredLogger(
            "orchestrator",
            log_file=log_path / "orchestrator.jsonl" if log_path else None,
        )

        # Initialize all agents
        self._init_agents()

    def _init_agents(self):
        """Initialize all agents."""
        self.ingestion_agent = IngestionAgent(
            storage=self.storage,
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.schema_agent = SchemaAgent(
            storage=self.storage,
            config=self.config,
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.quality_agent = QualityAgent(
            config=self.config,
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.operational_agent = OperationalAgent(
            config=self.config,
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.anomaly_agent = AnomalyAgent(
            config=self.config,
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.risk_agent = RiskAgent(
            config=self.config,
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.insight_agent = InsightAgent(
            insight_generator=LocalInsightGenerator(),
            event_bus=self.event_bus,
            logger=self.logger,
        )
        self.alert_agent = AlertAgent(
            config=self.config,
            storage=self.storage,
            event_bus=self.event_bus,
            logger=self.logger,
        )

    def run(
        self,
        zip_path: Optional[Path] = None,
        existing_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline.

        Args:
            zip_path: Path to ZIP file to extract and ingest.
            existing_dir: Path to existing directory with Excel files.

        Returns:
            Pipeline execution results.
        """
        session_id = str(uuid.uuid4())
        context = PipelineContext(session_id=session_id)

        self.logger.info(
            f"Starting pipeline execution",
            context={"session_id": session_id},
        )

        self.event_bus.publish(
            self._create_event(
                EventType.PIPELINE_STARTED,
                session_id,
                data={"session_id": session_id},
            )
        )

        try:
            # Step 1: Prepare data
            if zip_path:
                IngestionAgent.extract_zip(zip_path, self.storage.raw_path)
            elif existing_dir:
                self._copy_existing_dir(existing_dir)
            else:
                raise ValueError("Must provide either zip_path or existing_dir")

            # Step 2: Run pipeline agents in sequence
            self.ingestion_agent.execute(context)
            self.schema_agent.execute(context)
            self.quality_agent.execute(context)
            self.operational_agent.execute(context)
            self.anomaly_agent.execute(context)
            self.risk_agent.execute(context)
            self.insight_agent.execute(context)
            self.alert_agent.execute(context)

            # Mark completion
            from datetime import datetime
            context.completed_at = datetime.utcnow().isoformat()

            self.event_bus.publish(
                self._create_event(
                    EventType.PIPELINE_COMPLETED,
                    session_id,
                    data={
                        "session_id": session_id,
                        "files_ingested": len(context.ingestion_metadata),
                        "quality_metrics": len(context.quality_metrics),
                        "anomalies_detected": len(context.anomalies),
                        "alerts_created": len(context.alerts),
                    },
                )
            )

            # Compile results
            results = self._compile_results(context)

            self.logger.info(
                "Pipeline completed successfully",
                context={"session_id": session_id, "results": results},
            )

            return results

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(
                error_msg,
                context={"session_id": session_id, "error": str(e)},
            )
            context.errors.append(error_msg)

            self.event_bus.publish(
                self._create_event(
                    EventType.PIPELINE_FAILED,
                    session_id,
                    data={"error": str(e)},
                )
            )

            return {"success": False, "error": str(e), "session_id": session_id}

    def _copy_existing_dir(self, source_dir: Path):
        """Copy existing directory to raw storage."""
        import shutil

        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Find all Excel files in source directory
        excel_files = list(source_dir.glob("**/*.xlsx")) + list(
            source_dir.glob("**/*.xls")
        )

        for excel_file in excel_files:
            # Preserve relative path structure
            rel_path = excel_file.relative_to(source_dir.parent)
            dest_path = self.storage.raw_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(excel_file, dest_path)

    def _create_event(self, event_type: EventType, session_id: str, data: Dict[str, Any]):
        """Create an event."""
        from app.core.events import Event
        return Event(
            event_type=event_type,
            session_id=session_id,
            source_agent="orchestrator",
            data=data,
        )

    def _compile_results(self, context: PipelineContext) -> Dict[str, Any]:
        """Compile pipeline results for output."""
        return {
            "success": True,
            "session_id": context.session_id,
            "summary": {
                "files_ingested": len(context.ingestion_metadata),
                "datasets_harmonized": len(context.harmonized_data),
                "quality_metrics": len(context.quality_metrics),
                "operational_kpis": len(context.operational_kpis),
                "anomalies_detected": len(context.anomalies),
                "risk_scores": len(context.risk_scores),
                "alerts_created": len(context.alerts),
                "insights_generated": len(context.insights),
            },
            "quality": {
                "metrics": [
                    {
                        "dataset": name,
                        "quality_score": metric.overall_quality_score,
                        "completeness": metric.completeness_score,
                        "validity": metric.validity_score,
                        "consistency": metric.consistency_score,
                        "issues": metric.issues[:5],  # Top 5 issues
                    }
                    for name, metric in context.quality_metrics.items()
                ],
            },
            "risk": {
                "scores": [
                    {
                        "dataset": score.dataset_name,
                        "overall_score": score.overall_score,
                        "risk_level": score.risk_level.value,
                        "quality_component": score.quality_component,
                        "operational_component": score.operational_component,
                        "anomaly_component": score.anomaly_component,
                        "study_id": score.study_id,
                        "site_id": score.site_id,
                    }
                    for score in context.risk_scores
                ],
            },
            "anomalies": [
                {
                    "id": a.anomaly_id,
                    "type": a.anomaly_type,
                    "severity": a.severity.value,
                    "metric": a.metric_name,
                    "observed": a.observed_value,
                    "expected": a.expected_value,
                    "confidence": a.confidence,
                    "study_id": a.study_id,
                    "site_id": a.site_id,
                }
                for a in context.anomalies
            ],
            "alerts": [
                {
                    "id": alert.alert_id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "study_id": alert.study_id,
                    "site_id": alert.site_id,
                    "recommended_action": alert.recommended_action,
                }
                for alert in context.alerts
            ],
            "insights": context.insights[:10],  # Top 10 insights
            "errors": context.errors,
        }
