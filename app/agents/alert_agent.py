"""Alert and task automation agent."""
from __future__ import annotations

import uuid
from typing import List

from app.agents.base import BaseAgent
from app.core.config import ConfigLoader
from app.core.events import EventType
from app.core.models import AgentResult, Alert, PipelineContext, Severity
from app.core.storage import Storage


class AlertAgent(BaseAgent):
    """Creates alerts and automated tasks based on risks and anomalies."""

    def __init__(self, config: ConfigLoader, storage: Storage, **kwargs):
        """Initialize alert agent."""
        super().__init__("alert_agent", **kwargs)
        self.config = config
        self.storage = storage
        self.risk_threshold = config.get("alerts.risk_score_threshold", 60.0)
        self.quality_threshold = config.get("alerts.quality_score_threshold", 70.0)

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute alert generation."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            alerts = self._generate_alerts(context)
            context.alerts.extend(alerts)

            # Persist alerts to JSONL
            if alerts:
                alert_dicts = [
                    {
                        "alert_id": a.alert_id,
                        "severity": a.severity.value,
                        "title": a.title,
                        "description": a.description,
                        "study_id": a.study_id,
                        "site_id": a.site_id,
                        "status": a.status,
                        "created_at": a.created_at,
                    }
                    for a in alerts
                ]
                self.storage.append_jsonl(alert_dicts, "alerts", subdir="alerts")

            # Publish alert events
            for alert in alerts:
                self.publish_event(
                    EventType.ALERT_CREATED,
                    context.session_id,
                    data={
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                    },
                )

            result.data = {"alerts_created": len(alerts)}
        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result

    def _generate_alerts(self, context: PipelineContext) -> List[Alert]:
        """Generate alerts from risks and anomalies."""
        alerts: List[Alert] = []

        # Alerts from high-risk scores
        for risk_score in context.risk_scores:
            if risk_score.overall_score >= self.risk_threshold:
                severity = (
                    Severity.CRITICAL
                    if risk_score.overall_score >= 75
                    else Severity.HIGH
                )
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    severity=severity,
                    title=f"High Risk: {risk_score.dataset_name}",
                    description=(
                        f"Risk score: {risk_score.overall_score:.1f}. "
                        f"Quality: {risk_score.quality_component:.1f}, "
                        f"Operational: {risk_score.operational_component:.1f}, "
                        f"Anomaly: {risk_score.anomaly_component:.1f}"
                    ),
                    study_id=risk_score.study_id,
                    site_id=risk_score.site_id,
                    evidence=risk_score.evidence,
                    recommended_action=risk_score.recommended_action,
                )
                alerts.append(alert)

        # Alerts from low-quality datasets
        for dataset_name, metrics in context.quality_metrics.items():
            if metrics.overall_quality_score < self.quality_threshold:
                severity = (
                    Severity.HIGH
                    if metrics.overall_quality_score < 50
                    else Severity.MEDIUM
                )
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    severity=severity,
                    title=f"Data Quality Issue: {dataset_name}",
                    description=(
                        f"Quality score: {metrics.overall_quality_score:.1f}%. "
                        f"Issues: {', '.join(metrics.issues[:3])}"
                    ),
                    evidence=metrics.issues,
                    recommended_action=(
                        "Review and remediate data quality issues. "
                        "Contact source site if external data."
                    ),
                )
                alerts.append(alert)

        # Alerts from critical anomalies
        for anomaly in context.anomalies:
            if anomaly.severity == Severity.CRITICAL:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    severity=Severity.CRITICAL,
                    title=f"Critical Anomaly: {anomaly.anomaly_type}",
                    description=(
                        f"Metric: {anomaly.metric_name}. "
                        f"Observed: {anomaly.observed_value}, "
                        f"Expected: {anomaly.expected_value}. "
                        f"Reason: {anomaly.reason}"
                    ),
                    study_id=anomaly.study_id,
                    site_id=anomaly.site_id,
                    evidence=[anomaly.reason],
                    recommended_action="Investigate immediately and take corrective action.",
                    related_anomalies=[anomaly.anomaly_id],
                )
                alerts.append(alert)

        return alerts
