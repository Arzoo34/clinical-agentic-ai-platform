"""
Generative and agentic AI layers that produce human-readable insights and alerts.
"""
from __future__ import annotations

from typing import List

from pyspark.sql import DataFrame, functions as F, types as T


def _agent_rules() -> T.ArrayType:
    """Define rule-based alerts as a Spark UDF returning a list of reasons."""
    def evaluate(open_issues, sae_events, missing_pages, missing_labs, coding_backlog, inactivated_forms, risk_probability):
        alerts: List[str] = []
        if sae_events and sae_events > 0:
            alerts.append("Serious adverse events present")
        if open_issues and open_issues >= 5:
            alerts.append("High unresolved data queries")
        if missing_pages and missing_pages >= 3:
            alerts.append("Frequent missing pages")
        if missing_labs and missing_labs >= 2:
            alerts.append("Lab data gaps")
        if coding_backlog and coding_backlog >= 2:
            alerts.append("Backlog in medical coding")
        if inactivated_forms and inactivated_forms >= 1:
            alerts.append("Forms/folders inactivated")
        if risk_probability and risk_probability > 0.6:
            alerts.append("Model: probability > 0.6")
        return alerts

    return F.udf(evaluate, T.ArrayType(T.StringType()))


def _insight_text() -> T.StringType:
    """Return UDF that crafts a concise English explanation for a site."""
    def craft(site_id, country, risk_probability, open_issues, sae_events, missing_pages, missing_labs, coding_backlog):
        parts = [
            f"Site {site_id or 'N/A'} ({country or 'unknown'}) shows {risk_probability:.2f} predicted probability of operational risk."
        ]
        parts.append(
            f"Signals: {int(open_issues or 0)} open issues, {int(sae_events or 0)} SAE discrepancies, {int(missing_pages or 0)} missing pages, {int(missing_labs or 0)} lab gaps, {int(coding_backlog or 0)} coding backlog items."
        )
        if risk_probability >= 0.6 or (sae_events or 0) > 0:
            parts.append("Recommend urgent data cleaning, SAE reconciliation, and site follow-up within 48 hours.")
        else:
            parts.append("Monitor routinely; focus on clearing open queries and missing documents.")
        return " ".join(parts)

    return F.udf(craft, T.StringType())


def attach_ai_layers(df: DataFrame) -> DataFrame:
    """Add alert reasons and generated insights to the predictions DataFrame."""
    agent_udf = _agent_rules()
    insight_udf = _insight_text()

    df = df.withColumn(
        "alerts",
        agent_udf(
            F.col("open_issues"),
            F.col("sae_events"),
            F.col("missing_pages"),
            F.col("missing_labs"),
            F.col("coding_backlog"),
            F.col("inactivated_forms"),
            F.col("risk_probability"),
        ),
    )
    df = df.withColumn("alert_count", F.size("alerts"))
    df = df.withColumn(
        "ai_insight",
        insight_udf(
            F.col("site_id"),
            F.col("country"),
            F.col("risk_probability"),
            F.col("open_issues"),
            F.col("sae_events"),
            F.col("missing_pages"),
            F.col("missing_labs"),
            F.col("coding_backlog"),
        ),
    )
    df = df.withColumn("alert_flag", F.when(F.col("alert_count") > 0, F.lit(1)).otherwise(F.lit(0)))
    return df

