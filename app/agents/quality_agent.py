"""Data quality monitoring agent for clinical datasets."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent
from app.core.config import ConfigLoader
from app.core.events import EventType
from app.core.models import AgentResult, DataQualityMetrics, PipelineContext


class QualityAgent(BaseAgent):
    """Monitors data completeness, duplicate records, outliers, and schema compliance."""

    def __init__(self, config: Optional[ConfigLoader] = None, **kwargs):
        super().__init__("quality_agent", **kwargs)
        self.config = config or ConfigLoader()
        self.missing_rate_threshold = float(self.config.get("quality.missing_rate_threshold", 0.15))
        self.duplicate_threshold = float(self.config.get("quality.duplicate_threshold", 0.05))
        self.outlier_zscore_threshold = float(self.config.get("quality.outlier_zscore_threshold", 3.0))

    def execute(self, context: PipelineContext) -> AgentResult:
        """Assess quality across all harmonized datasets."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            if not context.harmonized_data:
                result.warnings.append("No harmonized datasets available to assess")
                self.log_completion(context, result)
                return result

            for name, df in context.harmonized_data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue

                completeness = self._calculate_completeness(df)
                dup_count = self._detect_duplicates(df)
                outlier_count = self._detect_outliers(df)

                row_count = len(df)
                dup_rate = dup_count / row_count if row_count > 0 else 0.0
                validity = max(0.0, 100.0 - (dup_rate * 100.0))
                consistency = max(0.0, 100.0 - (outlier_count / max(1, row_count) * 100.0))

                # Composite quality score
                overall_score = (completeness * 0.5) + (validity * 0.3) + (consistency * 0.2)
                overall_score = round(max(0.0, min(100.0, overall_score)), 2)

                issues: List[str] = []
                missing_rate = (100.0 - completeness) / 100.0
                if missing_rate > self.missing_rate_threshold:
                    issues.append(f"High missingness: {missing_rate * 100:.1f}% values missing")
                if dup_count > 0:
                    issues.append(f"Detected {dup_count} duplicate rows ({dup_rate * 100:.1f}%)")
                if outlier_count > 0:
                    issues.append(f"Detected {outlier_count} statistical outlier values")

                metrics = DataQualityMetrics(
                    dataset_name=name,
                    completeness_score=round(completeness, 2),
                    validity_score=round(validity, 2),
                    consistency_score=round(consistency, 2),
                    overall_quality_score=overall_score,
                    row_count=row_count,
                    duplicate_count=dup_count,
                    outlier_count=outlier_count,
                    issues=issues,
                )
                context.quality_metrics[name] = metrics

            result.data = {"assessed_datasets": len(context.quality_metrics)}
            self.publish_event(
                EventType.QUALITY_ASSESSED,
                context.session_id,
                data=result.data,
            )

        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result

    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate percentage of non-null cells across the DataFrame."""
        total_cells = df.size
        if total_cells == 0:
            return 100.0
        non_null_cells = int(df.notna().sum().sum())
        return float((non_null_cells / total_cells) * 100.0)

    def _detect_duplicates(self, df: pd.DataFrame) -> int:
        """Count identical duplicate rows."""
        if df.empty:
            return 0
        return int(df.duplicated().sum())

    def _detect_outliers(self, df: pd.DataFrame) -> int:
        """Count outlier points in numeric columns using z-score."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0 or len(df) < 3:
            return 0

        outlier_count = 0
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            std = float(series.std())
            if std == 0.0 or np.isnan(std):
                continue
            mean = float(series.mean())
            z_scores = np.abs((series - mean) / std)
            outlier_count += int((z_scores > self.outlier_zscore_threshold).sum())

        return outlier_count
