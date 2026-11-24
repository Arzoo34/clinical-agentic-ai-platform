"""Operational intelligence agent for tracking trial operations and freshness."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.config import ConfigLoader
from app.core.events import EventType
from app.core.models import AgentResult, OperationalKPI, PipelineContext


class OperationalAgent(BaseAgent):
    """Monitors operational health, data freshness, file volume, and site throughput."""

    def __init__(self, config: Optional[ConfigLoader] = None, **kwargs):
        super().__init__("operational_agent", **kwargs)
        self.config = config or ConfigLoader()
        self.freshness_threshold = int(self.config.get("operations.freshness_days_threshold", 14))

    def execute(self, context: PipelineContext) -> AgentResult:
        """Evaluate operational metrics and KPIs."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            kpis: List[OperationalKPI] = []

            # 1. Total Volume KPI
            total_records = sum(m.row_count for m in context.ingestion_metadata)
            total_files = len(context.ingestion_metadata)
            kpis.append(
                OperationalKPI(
                    kpi_id=str(uuid.uuid4()),
                    name="Total Ingested Records",
                    metric_name="total_records",
                    value=float(total_records),
                    status="normal" if total_records > 0 else "warning",
                    description=f"Total of {total_records} rows ingested across {total_files} files.",
                )
            )

            # 2. File Freshness KPIs
            now = datetime.now()
            stale_files = 0
            for meta in context.ingestion_metadata:
                if meta.modification_time:
                    try:
                        mod_dt = datetime.fromisoformat(meta.modification_time)
                        days_old = (now - mod_dt).days
                        if days_old > self.freshness_threshold * 2:
                            stale_files += 1
                    except Exception:
                        pass

            freshness_status = "critical" if stale_files > 10 else ("warning" if stale_files > 0 else "normal")
            kpis.append(
                OperationalKPI(
                    kpi_id=str(uuid.uuid4()),
                    name="Data Submission Freshness",
                    metric_name="stale_files_count",
                    value=float(stale_files),
                    target_value=0.0,
                    status=freshness_status,
                    description=f"{stale_files} data files have not been updated within {self.freshness_threshold * 2} days.",
                )
            )

            # 3. Harmonization Throughput KPI
            harmonized_count = len(context.harmonized_data)
            harmonization_ratio = harmonized_count / max(1, total_files)
            harm_status = "normal" if harmonization_ratio >= 0.8 else "warning"
            kpis.append(
                OperationalKPI(
                    kpi_id=str(uuid.uuid4()),
                    name="Schema Harmonization Rate",
                    metric_name="harmonization_rate",
                    value=round(harmonization_ratio * 100, 1),
                    target_value=100.0,
                    status=harm_status,
                    description=f"Successfully harmonized {harmonized_count} of {total_files} files ({harmonization_ratio*100:.1f}%).",
                )
            )

            # 4. Site-Level Operations if available
            studies_seen = set(m.study_id for m in context.ingestion_metadata if m.study_id)
            for study in studies_seen:
                study_files = [m for m in context.ingestion_metadata if m.study_id == study]
                study_rows = sum(m.row_count for m in study_files)
                kpis.append(
                    OperationalKPI(
                        kpi_id=str(uuid.uuid4()),
                        name=f"Study Activity: {study}",
                        metric_name="study_row_volume",
                        value=float(study_rows),
                        study_id=study,
                        status="normal" if study_rows > 0 else "warning",
                        description=f"{len(study_files)} files ingested with {study_rows} records.",
                    )
                )

            context.operational_kpis.extend(kpis)
            result.data = {"kpis_generated": len(kpis)}

            self.publish_event(
                EventType.OPERATIONS_EVALUATED,
                context.session_id,
                data=result.data,
            )

        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result
