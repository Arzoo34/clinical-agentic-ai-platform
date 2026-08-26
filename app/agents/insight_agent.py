"""Insight and explanation generation agent."""
from __future__ import annotations

from typing import List, Optional

from app.agents.base import BaseAgent
from app.core.events import EventType
from app.core.models import AgentResult, PipelineContext
from app.llm.insight import InsightGenerator, LocalInsightGenerator


class InsightAgent(BaseAgent):
    """Generates human-readable insights and explanations."""

    def __init__(
        self,
        insight_generator: Optional[InsightGenerator] = None,
        **kwargs,
    ):
        """Initialize insight agent."""
        super().__init__("insight_agent", **kwargs)
        self.insight_generator = insight_generator or LocalInsightGenerator()

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute insight generation."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            insights = self._generate_insights(context)
            context.insights.extend(insights)

            # Publish insight events
            for insight in insights:
                self.publish_event(
                    EventType.INSIGHT_GENERATED,
                    context.session_id,
                    data={"insight": insight},
                )

            result.data = {"insights_generated": len(insights)}
        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result

    def _generate_insights(self, context: PipelineContext) -> List[str]:
        """Generate insights from anomalies and risk scores."""
        insights: List[str] = []

        # Generate insights from anomalies
        for anomaly in context.anomalies:
            insight = self.insight_generator.generate_insight(
                anomaly_type=anomaly.anomaly_type,
                metric_name=anomaly.metric_name,
                observed_value=anomaly.observed_value,
                expected_value=anomaly.expected_value,
                context=f"Study: {anomaly.study_id}, Severity: {anomaly.severity.value}",
            )
            insights.append(insight)

        # Generate insights from high-risk scores
        for risk_score in context.risk_scores:
            if risk_score.overall_score > 60:
                insight = self._generate_risk_insight(risk_score)
                insights.append(insight)

        return insights

    def _generate_risk_insight(self, risk_score) -> str:
        """Generate insight from risk score."""
        parts = []
        parts.append(
            f"Risk assessment for {risk_score.dataset_name}: "
            f"Overall risk score is {risk_score.overall_score:.1f} ({risk_score.risk_level.value})."
        )

        if risk_score.evidence:
            parts.append("Key findings: " + " | ".join(risk_score.evidence))

        if risk_score.recommended_action:
            parts.append(f"Recommended action: {risk_score.recommended_action}")

        return " ".join(parts)
