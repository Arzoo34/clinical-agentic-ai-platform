"""LLM abstraction layer for insight generation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class InsightGenerator(ABC):
    """Abstract base for insight generation."""

    @abstractmethod
    def generate_insight(
        self,
        anomaly_type: str,
        metric_name: str,
        observed_value: float,
        expected_value: Optional[float] = None,
        context: Optional[str] = None,
    ) -> str:
        """Generate human-readable insight."""
        pass


class LocalInsightGenerator(InsightGenerator):
    """Deterministic, template-based insight generation (no external API)."""

    def generate_insight(
        self,
        anomaly_type: str,
        metric_name: str,
        observed_value: float,
        expected_value: Optional[float] = None,
        context: Optional[str] = None,
    ) -> str:
        """Generate insight using templates."""
        if anomaly_type == "row_count_deviation":
            if expected_value:
                variance = ((observed_value - expected_value) / expected_value) * 100
                if variance > 0:
                    return (
                        f"Data submission contains {variance:.1f}% more records than expected. "
                        "This may indicate duplicated data entry or a data submission error. "
                        "Recommend verifying the submission and checking for duplicate records."
                    )
                else:
                    return (
                        f"Data submission contains {abs(variance):.1f}% fewer records than expected. "
                        "This may indicate incomplete data submission or missing records. "
                        "Recommend verifying submission completeness."
                    )

        elif anomaly_type == "distribution_skew":
            return (
                f"The distribution of {metric_name} shows unusual skewness, "
                "which may indicate data entry errors, outliers, or genuine patterns in the trial data. "
                "Recommend reviewing extreme values for data quality."
            )

        elif anomaly_type == "high_missingness":
            return (
                f"High percentage of missing values detected in {metric_name}. "
                "This may affect analysis validity. "
                "Recommend contacting the site to obtain missing data or documenting reasons for missingness."
            )

        elif anomaly_type == "duplicate_detection":
            return (
                f"Duplicate records were detected in {metric_name}. "
                "Recommend deduplication before analysis or verification with the data source."
            )

        else:
            # Generic insight
            return (
                f"Anomaly detected in {metric_name}: observed value is {observed_value}. "
                f"Recommend investigating the underlying cause and taking corrective action."
            )


class OpenAIInsightGenerator(InsightGenerator):
    """OpenAI-based insight generation (requires API key)."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize with OpenAI API key."""
        self.api_key = api_key
        self.model = model
        try:
            import openai
            self.openai = openai
            openai.api_key = api_key
        except ImportError:
            raise ImportError("openai package required for OpenAI insight generation")

    def generate_insight(
        self,
        anomaly_type: str,
        metric_name: str,
        observed_value: float,
        expected_value: Optional[float] = None,
        context: Optional[str] = None,
    ) -> str:
        """Generate insight using OpenAI API."""
        prompt = self._build_prompt(
            anomaly_type, metric_name, observed_value, expected_value, context
        )

        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical trial data expert. "
                            "Provide concise, actionable insights about data quality issues. "
                            "Be specific but avoid medical claims. Focus on data integrity."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to local generator on API error
            self.logger.warning(f"OpenAI API error: {e}. Using local generator.")
            local_gen = LocalInsightGenerator()
            return local_gen.generate_insight(
                anomaly_type, metric_name, observed_value, expected_value, context
            )

    def _build_prompt(self, anomaly_type, metric_name, observed_value, expected_value, context):
        """Build prompt for the LLM."""
        prompt = f"Anomaly: {anomaly_type}\n"
        prompt += f"Metric: {metric_name}\n"
        prompt += f"Observed value: {observed_value}\n"
        if expected_value is not None:
            prompt += f"Expected value: {expected_value}\n"
        if context:
            prompt += f"Context: {context}\n"
        prompt += "\nProvide a brief, actionable insight:"
        return prompt
