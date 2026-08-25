"""Schema harmonization agent."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from app.agents.base import BaseAgent
from app.core.config import ConfigLoader
from app.core.events import EventType
from app.core.models import AgentResult, PipelineContext
from app.core.storage import Storage


class SchemaAgent(BaseAgent):
    """Harmonizes heterogeneous schemas into canonical form."""

    def __init__(self, storage: Storage, config: ConfigLoader, **kwargs):
        """Initialize schema agent."""
        super().__init__("schema_agent", **kwargs)
        self.storage = storage
        self.config = config
        self.alias_map = config.get("schema.alias_map", {})
        self.required_fields = config.get("schema.required_fields", [])

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute schema harmonization."""
        self.log_start(context)
        result = AgentResult(agent_name=self.name)

        try:
            # Get raw file metadata
            if not context.ingestion_metadata:
                result.warnings.append("No files ingested")
                self.log_completion(context, result)
                return result

            # Process each file
            harmonized_count = 0
            for metadata in context.ingestion_metadata:
                try:
                    df = pd.read_excel(metadata.file_path)
                    if df.empty:
                        result.warnings.append(
                            f"Empty file: {metadata.filename}"
                        )
                        continue

                    # Harmonize schema
                    harmonized_df = self._harmonize_columns(df)

                    # Store harmonized data
                    dataset_name = metadata.filename.replace(".xlsx", "").replace(".xls", "")
                    self.storage.save_dataframe_parquet(
                        harmonized_df,
                        dataset_name,
                        study=metadata.study_id,
                    )
                    context.harmonized_data[dataset_name] = harmonized_df
                    harmonized_count += 1

                except Exception as e:
                    result.errors.append(
                        f"Failed to harmonize {metadata.filename}: {str(e)}"
                    )

            result.data = {"harmonized_count": harmonized_count}
            self.publish_event(
                EventType.SCHEMA_HARMONIZED,
                context.session_id,
                data=result.data,
            )

        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            self.log_error(context, str(e))

        self.log_completion(context, result)
        return result

    def _harmonize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names using alias mapping."""
        # Create mapping from current names to canonical names
        column_mapping = {}
        for col in df.columns:
            canonical_name = self._normalize_column_name(col)
            column_mapping[col] = canonical_name

        # Rename columns
        df = df.rename(columns=column_mapping)

        return df

    def _normalize_column_name(self, col: str) -> str:
        """Normalize a single column name."""
        # Try exact match in alias map
        col_lower = col.lower().strip()
        if col_lower in self.alias_map:
            return self.alias_map[col_lower]

        # Try partial matching and common patterns
        for alias, canonical in self.alias_map.items():
            if alias.replace(" ", "") == col_lower.replace(" ", ""):
                return canonical

        # If no match, use the column name as-is (lowercased, spaces to underscores)
        return col_lower.replace(" ", "_").replace("-", "_")

    def get_missing_required_fields(self, df: pd.DataFrame) -> Set[str]:
        """Check which required fields are missing."""
        available_cols = set(df.columns.str.lower())
        missing = set()
        for field in self.required_fields:
            if field.lower() not in available_cols:
                missing.add(field)
        return missing
