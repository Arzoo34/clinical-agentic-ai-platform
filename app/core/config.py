"""Configuration loader for the platform."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "raw_root": "data/raw",
        "staging_root": "data/staging",
        "curated_root": "data/curated",
        "logs_root": "data/logs",
        "demo_data_root": "data/demo",
    },
    "schema": {
        "alias_map": {
            "subject id": "subject_id",
            "subjectid": "subject_id",
            "subj_id": "subject_id",
            "site": "site_id",
            "site id": "site_id",
            "sitename": "site_id",
            "visit": "visit_name",
            "visit name": "visit_name",
            "visit_date": "visit_date",
            "visit date": "visit_date",
            "datetime": "visit_date",
            "date": "visit_date",
            "adverse event": "adverse_event",
            "sae": "serious_adverse_event",
            "status": "record_status",
            "metric": "metric_name",
            "value": "metric_value",
        },
        "required_fields": ["subject_id", "site_id", "visit_date"],
    },
    "quality": {
        "missing_rate_threshold": 0.15,
        "duplicate_threshold": 0.05,
        "outlier_zscore_threshold": 3.0,
    },
    "operations": {
        "freshness_days_threshold": 14,
    },
    "anomaly": {
        "zscore_threshold": 2.5,
    },
    "risk": {
        "weights": {
            "quality": 0.5,
            "operations": 0.25,
            "anomaly": 0.25,
        }
    },
    "alerts": {
        "risk_score_threshold": 60.0,
        "quality_score_threshold": 70.0,
    },
}


class ConfigLoader:
    """Loads and provides access to platform configuration."""

    def __init__(self, config_path: Optional[Path | str] = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to YAML config file. If None, tries 'config/config.yaml'.
        """
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self._config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self.load()

    def load(self) -> Dict[str, Any]:
        """Load configuration from disk if available."""
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self._deep_update(self._config, loaded)
            except Exception:
                pass
        return self._config

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively update dictionary."""
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot-notation.

        Example: config.get("schema.alias_map", {})
        """
        parts = key.split(".")
        current: Any = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration dictionary."""
        return self._config
