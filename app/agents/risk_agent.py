"""Risk scoring and prioritization agent."""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.config import ConfigLoader
from app.core.events import EventType
from app.core.models import (
    AgentResult,
    PipelineContext,
    RiskLevel,
    RiskScore,
)


class RiskAgent(BaseAgent):
    """Combines quality, operational, and anomaly signals into risk scores."""

    def __init__(self, config: ConfigLoader, **kwargs):
        """Initialize risk agent."""
        super().__init__("risk_agent", **kwargs)
        self.config = config
        self.risk_weights = config.get(
            "risk.weights", {"quality": 0.5, "operations": 0.25, "anomaly": 0.25}
        )

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute risk scoring."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            risk_scores = self._calculate_risk_scores(context)
            context.risk_scores.extend(risk_scores)

            # Publish risk events
            for score in risk_scores:
                self.publish_event(
                    EventType.RISK_CALCULATED,
                    context.session_id,
                    data={
                        "study_id": score.study_id,
                        "site_id": score.site_id,
                        "risk_score": score.overall_score,
                        "risk_level": score.risk_level.value,
                    },
                )

            result.data = {"risk_scores_calculated": len(risk_scores)}
        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result

    def _calculate_risk_scores(self, context: PipelineContext) -> List[RiskScore]:
        """Calculate composite risk scores."""
        risk_scores: List[RiskScore] = []

        # Calculate global risk score
        if context.quality_metrics or context.anomalies:
            global_score = self._calculate_composite_score(context)
            if global_score:
                risk_scores.append(global_score)

        # Study-level risk scores
        studies = set()
        for metadata in context.ingestion_metadata:
            if metadata.study_id:
                studies.add(metadata.study_id)

        for study in studies:
            study_score = self._calculate_study_risk(context, study)
            if study_score:
                risk_scores.append(study_score)

        return risk_scores

    def _calculate_composite_score(self, context: PipelineContext) -> Optional[RiskScore]:
        """Calculate overall risk score."""
        # Quality component: average of all quality scores
        quality_score = 0.0
        if context.quality_metrics:
            quality_scores = [
                m.overall_quality_score for m in context.quality_metrics.values()
            ]
            quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 50.0

        # Operational component: based on KPI status
        operational_score = self._calculate_operational_component(context)

        # Anomaly component: based on anomaly count and severity
        anomaly_score = self._calculate_anomaly_component(context)

        # Composite risk score (inverse of quality, direct of anomaly/operational)
        overall_score = (
            (100 - quality_score) * self.risk_weights.get("quality", 0.5)
            + operational_score * self.risk_weights.get("operations", 0.25)
            + anomaly_score * self.risk_weights.get("anomaly", 0.25)
        )

        # Clamp to 0-100
        overall_score = max(0.0, min(100.0, overall_score))

        risk_level = self._score_to_level(overall_score)

        evidence = []
        if quality_score < 80:
            evidence.append(f"Low data quality score: {quality_score:.1f}")
        if len(context.anomalies) > 0:
            evidence.append(f"Detected {len(context.anomalies)} anomalies")
        if operational_score > 50:
            evidence.append("Operational issues detected")

        return RiskScore(
            dataset_name="overall",
            overall_score=overall_score,
            risk_level=risk_level,
            quality_component=100 - quality_score,
            operational_component=operational_score,
            anomaly_component=anomaly_score,
            evidence=evidence,
            recommended_action=self._recommend_action(risk_level, evidence),
        )

    def _calculate_study_risk(self, context: PipelineContext, study_id: str) -> Optional[RiskScore]:
        """Calculate risk score for a specific study."""
        # Filter metrics for this study
        study_metadata = [
            m for m in context.ingestion_metadata if m.study_id == study_id
        ]
        if not study_metadata:
            return None

        # Calculate components
        quality_scores = []
        for metadata in study_metadata:
            dataset_key = metadata.filename.replace(".xlsx", "").replace(".xls", "")
            if dataset_key in context.quality_metrics:
                quality_scores.append(
                    context.quality_metrics[dataset_key].overall_quality_score
                )

        quality_component = (
            (100 - sum(quality_scores) / len(quality_scores))
            if quality_scores
            else 50.0
        )

        # Count anomalies for this study
        study_anomalies = [
            a for a in context.anomalies if a.study_id == study_id
        ]
        anomaly_component = min(100.0, len(study_anomalies) * 10)

        operational_component = self._calculate_operational_component(context)

        overall_score = (
            quality_component * self.risk_weights.get("quality", 0.5)
            + operational_component * self.risk_weights.get("operations", 0.25)
            + anomaly_component * self.risk_weights.get("anomaly", 0.25)
        )

        overall_score = max(0.0, min(100.0, overall_score))
        risk_level = self._score_to_level(overall_score)

        return RiskScore(
            study_id=study_id,
            dataset_name=f"study_{study_id}",
            overall_score=overall_score,
            risk_level=risk_level,
            quality_component=quality_component,
            operational_component=operational_component,
            anomaly_component=anomaly_component,
            affected_records=sum(m.row_count for m in study_metadata),
            recommended_action=self._recommend_action(risk_level),
        )

    def _calculate_operational_component(self, context: PipelineContext) -> float:
        """Calculate risk from operational KPIs."""
        if not context.operational_kpis:
            return 0.0

        risk_score = 0.0
        critical_count = sum(
            1 for kpi in context.operational_kpis if kpi.status == "critical"
        )
        warning_count = sum(
            1 for kpi in context.operational_kpis if kpi.status == "warning"
        )

        risk_score += critical_count * 25
        risk_score += warning_count * 10

        return min(100.0, risk_score)

    def _calculate_anomaly_component(self, context: PipelineContext) -> float:
        """Calculate risk from anomalies."""
        if not context.anomalies:
            return 0.0

        risk_score = 0.0
        for anomaly in context.anomalies:
            if anomaly.severity.value == "critical":
                risk_score += 30
            elif anomaly.severity.value == "high":
                risk_score += 15
            elif anomaly.severity.value == "medium":
                risk_score += 5
            else:
                risk_score += 2

        return min(100.0, risk_score)

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score < 25:
            return RiskLevel.LOW
        elif score < 50:
            return RiskLevel.MODERATE
        elif score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _recommend_action(self, risk_level: RiskLevel, evidence: List[str] = None) -> str:
        """Generate recommended action based on risk level."""
        if risk_level == RiskLevel.CRITICAL:
            return "Immediate escalation required. Investigate all flagged issues urgently."
        elif risk_level == RiskLevel.HIGH:
            return "Schedule data review and investigation of high-risk areas within 48 hours."
        elif risk_level == RiskLevel.MODERATE:
            return "Monitor closely and plan corrective actions for identified issues."
        else:
            return "Continue normal operations with routine monitoring."
