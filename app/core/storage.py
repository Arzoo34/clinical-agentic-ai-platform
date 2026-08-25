"""Storage abstraction layer for datasets and artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class Storage:
    """File-based storage for datasets, metadata, and artifacts."""

    def __init__(self, base_path: Path = Path("data")):
        """Initialize storage."""
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.staging_path = self.base_path / "staging"
        self.curated_path = self.base_path / "curated"
        self.logs_path = self.base_path / "logs"

        # Create directories
        for path in [self.raw_path, self.staging_path, self.curated_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)

    def save_raw_file(self, data: bytes, filename: str) -> Path:
        """Save raw file."""
        path = self.raw_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def save_dataframe_parquet(self, df: pd.DataFrame, name: str, study: Optional[str] = None) -> Path:
        """Save dataframe as parquet."""
        subdir = self.curated_path / (study or "general")
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def save_dataframe_csv(self, df: pd.DataFrame, name: str, study: Optional[str] = None) -> Path:
        """Save dataframe as CSV."""
        subdir = self.curated_path / (study or "general")
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def load_dataframe_parquet(self, name: str, study: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load dataframe from parquet."""
        subdir = self.curated_path / (study or "general")
        path = subdir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def load_dataframe_csv(self, name: str, study: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load dataframe from CSV."""
        subdir = self.curated_path / (study or "general")
        path = subdir / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
        return None

    def save_json(self, data: Dict[str, Any] | List[Any], name: str, subdir: str = "metadata") -> Path:
        """Save JSON file."""
        path = self.logs_path / subdir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def append_jsonl(self, data: Dict[str, Any] | List[Dict[str, Any]], name: str, subdir: str = "metadata") -> Path:
        """Append to JSONL file (append-only log format)."""
        path = self.logs_path / subdir / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        items = data if isinstance(data, list) else [data]
        with open(path, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, default=str) + "\n")
        return path

    def load_jsonl(self, name: str, subdir: str = "metadata") -> List[Dict[str, Any]]:
        """Load JSONL file."""
        path = self.logs_path / subdir / f"{name}.jsonl"
        if not path.exists():
            return []
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def load_json(self, name: str, subdir: str = "metadata") -> Optional[Dict[str, Any] | List[Any]]:
        """Load JSON file."""
        path = self.logs_path / subdir / f"{name}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_raw_files(self, pattern: str = "*") -> List[Path]:
        """List raw files."""
        return list(self.raw_path.glob(pattern))

    def list_curated_datasets(self) -> List[str]:
        """List all curated dataset names."""
        datasets = []
        for parquet_file in self.curated_path.glob("**/*.parquet"):
            datasets.append(parquet_file.stem)
        return sorted(set(datasets))
