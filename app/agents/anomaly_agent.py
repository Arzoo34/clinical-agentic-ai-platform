"""Anomaly detection agent for clinical trial data."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from app.agents.base import BaseAgent
from app.core.config import ConfigLoader
from app.core.events import EventType
from app.core.models import AgentResult, Anomaly, PipelineContext, Severity


class AnomalyAgent(BaseAgent):
    """Detects statistical anomalies such as unexpected row counts, distribution drift, and spikes."""

    def __init__(self, config: Optional[ConfigLoader] = None, **kwargs):
        super().__init__("anomaly_agent", **kwargs)
        self.config = config or ConfigLoader()
        self.zscore_threshold = float(self.config.get("anomaly.zscore_threshold", 2.5))

    def execute(self, context: PipelineContext) -> AgentResult:
        """Run anomaly detection across metadata and quality metrics."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            anomalies: List[Anomaly] = []

            # 1. Row count anomalies across files
            if context.ingestion_metadata:
                row_counts = [m.row_count for m in context.ingestion_metadata]
                row_anomalies = self._detect_row_count_anomalies(row_counts, context)
                anomalies.extend(row_anomalies)

            # 2. Quality-based anomalies (high missingness, excessive duplicates)
            for name, qm in context.quality_metrics.items():
                if qm.completeness_score < 60.0:
                    anomalies.append(
                        Anomaly(
                            anomaly_id=str(uuid.uuid4()),
                            anomaly_type="high_missingness",
                            severity=Severity.HIGH if qm.completeness_score < 40 else Severity.MEDIUM,
                            metric_name=f"{name}_completeness",
                            observed_value=float(qm.completeness_score),
                            expected_value=100.0,
                            confidence=0.95,
                            dataset_name=name,
                            reason=f"Completeness is abnormally low ({qm.completeness_score:.1f}%) in {name}.",
                        )
                    )
                if qm.duplicate_count > 50:
                    anomalies.append(
                        Anomaly(
                            anomaly_id=str(uuid.uuid4()),
                            anomaly_type="duplicate_detection",
                            severity=Severity.HIGH,
                            metric_name=f"{name}_duplicate_rows",
                            observed_value=float(qm.duplicate_count),
                            expected_value=0.0,
                            confidence=0.99,
                            dataset_name=name,
                            reason=f"Significant duplicate records detected ({qm.duplicate_count} rows) in {name}.",
                        )
                    )

            context.anomalies.extend(anomalies)
            result.data = {"anomalies_detected": len(anomalies)}

            for anomaly in anomalies:
                self.publish_event(
                    EventType.ANOMALY_DETECTED,
                    context.session_id,
                    data={
                        "anomaly_id": anomaly.anomaly_id,
                        "type": anomaly.anomaly_type,
                        "severity": anomaly.severity.value,
                        "metric": anomaly.metric_name,
                    },
                )

        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result

    def _detect_row_count_anomalies(
        self,
        row_counts: List[int],
        context: PipelineContext,
    ) -> List[Anomaly]:
        """Detect row count outliers via modified z-score or standard z-score."""
        anomalies: List[Anomaly] = []
        if len(row_counts) < 3:
            return anomalies

        arr = np.array(row_counts, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std == 0.0:
            return anomalies

        z_scores = np.abs((arr - mean) / std)

        for i, z in enumerate(z_scores):
            # In small samples (e.g. N=3), max theoretical z-score cannot exceed ~1.414
            is_anomaly = (
                z >= self.zscore_threshold
                or (len(arr) < 10 and z >= 1.3 and (arr[i] > mean * 1.5 or arr[i] < mean * 0.5))
            )
            if is_anomaly:
                meta = context.ingestion_metadata[i] if i < len(context.ingestion_metadata) else None
                severity = Severity.CRITICAL if (z >= 2.0 or arr[i] > mean * 2.0) else Severity.HIGH

                anomalies.append(
                    Anomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="row_count_deviation",
                        severity=severity,
                        metric_name="row_count",
                        observed_value=float(arr[i]),
                        expected_value=round(mean, 1),
                        confidence=min(0.99, round(0.5 + (float(z) / 5.0), 2)),
                        study_id=meta.study_id if meta else None,
                        site_id=meta.site_id if meta else None,
                        dataset_name=meta.filename if meta else None,
                        reason=(
                            f"File {meta.filename if meta else i} row count ({int(arr[i])}) "
                            f"deviates significantly from average ({mean:.1f}, z-score {z:.2f})."
                        ),
                    )
                )

        return anomalies
