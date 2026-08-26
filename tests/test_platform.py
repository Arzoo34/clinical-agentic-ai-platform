"""Comprehensive tests for the clinical trial intelligence platform."""
import hashlib
import json
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.agents.ingestion_agent import IngestionAgent
from app.agents.schema_agent import SchemaAgent
from app.agents.quality_agent import QualityAgent
from app.agents.operational_agent import OperationalAgent
from app.agents.anomaly_agent import AnomalyAgent
from app.agents.risk_agent import RiskAgent
from app.agents.alert_agent import AlertAgent
from app.core.config import ConfigLoader
from app.core.events import EventBus, EventType
from app.core.logging import StructuredLogger
from app.core.models import (
    FileMetadata,
    DataQualityMetrics,
    Anomaly,
    RiskScore,
    Alert,
    PipelineContext,
    Severity,
    RiskLevel,
)
from app.core.storage import Storage
from app.llm.insight import LocalInsightGenerator
from app.orchestrator import PipelineOrchestrator


class TestModels:
    """Test data models and dataclasses."""

    def test_file_metadata(self):
        """Test FileMetadata creation."""
        metadata = FileMetadata(
            filename="test.xlsx",
            file_path="/path/to/test.xlsx",
            study_id="Study 1",
            row_count=100,
            column_count=10,
        )
        assert metadata.filename == "test.xlsx"
        assert metadata.row_count == 100
        assert metadata.study_id == "Study 1"

    def test_data_quality_metrics(self):
        """Test DataQualityMetrics creation."""
        metrics = DataQualityMetrics(
            dataset_name="test_data",
            completeness_score=95.0,
            overall_quality_score=88.5,
        )
        assert metrics.dataset_name == "test_data"
        assert metrics.completeness_score == 95.0
        assert 0 <= metrics.overall_quality_score <= 100

    def test_risk_score(self):
        """Test RiskScore creation and level assignment."""
        score = RiskScore(
            dataset_name="test",
            overall_score=75.0,
            risk_level=RiskLevel.HIGH,
        )
        assert score.overall_score == 75.0
        assert score.risk_level == RiskLevel.HIGH

    def test_alert(self):
        """Test Alert creation."""
        alert = Alert(
            alert_id="test-123",
            severity=Severity.HIGH,
            title="Test Alert",
            description="Test description",
        )
        assert alert.alert_id == "test-123"
        assert alert.severity == Severity.HIGH
        assert alert.status == "open"


class TestStorage:
    """Test storage abstraction."""

    def test_storage_initialization(self):
        """Test storage initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(Path(tmpdir))
            assert storage.raw_path.exists()
            assert storage.curated_path.exists()
            assert storage.logs_path.exists()

    def test_save_and_load_dataframe(self):
        """Test saving and loading dataframes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(Path(tmpdir))
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

            # Save as parquet
            storage.save_dataframe_parquet(df, "test_data")
            loaded = storage.load_dataframe_parquet("test_data")
            assert loaded is not None
            assert len(loaded) == 3
            pd.testing.assert_frame_equal(df, loaded)

    def test_save_and_load_json(self):
        """Test saving and loading JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(Path(tmpdir))
            data = {"key": "value", "number": 42}

            storage.save_json(data, "test_data")
            loaded = storage.load_json("test_data")
            assert loaded == data

    def test_append_jsonl(self):
        """Test JSONL append functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(Path(tmpdir))
            data1 = {"id": 1, "value": "first"}
            data2 = {"id": 2, "value": "second"}

            storage.append_jsonl(data1, "test_log")
            storage.append_jsonl(data2, "test_log")

            loaded = storage.load_jsonl("test_log")
            assert len(loaded) == 2
            assert loaded[0]["id"] == 1
            assert loaded[1]["id"] == 2


class TestEventBus:
    """Test event bus functionality."""

    def test_publish_and_subscribe(self):
        """Test event publishing and subscription."""
        event_bus = EventBus()
        events_received = []

        def handler(event):
            events_received.append(event)

        event_bus.subscribe(EventType.FILE_INGESTED, handler)

        from app.core.events import Event
        event = Event(
            event_type=EventType.FILE_INGESTED,
            session_id="test-session",
            data={"count": 5},
        )
        event_bus.publish(event)

        assert len(events_received) == 1
        assert events_received[0].data["count"] == 5

    def test_event_filtering(self):
        """Test event filtering."""
        event_bus = EventBus()

        from app.core.events import Event
        event1 = Event(
            event_type=EventType.FILE_INGESTED,
            session_id="session1",
        )
        event2 = Event(
            event_type=EventType.SCHEMA_HARMONIZED,
            session_id="session1",
        )
        event_bus.publish(event1)
        event_bus.publish(event2)

        ingestion_events = event_bus.get_events(EventType.FILE_INGESTED)
        assert len(ingestion_events) == 1
        assert ingestion_events[0].event_type == EventType.FILE_INGESTED


class TestQualityAgent:
    """Test data quality assessment."""

    def test_calculate_completeness(self):
        """Test completeness calculation."""
        config = ConfigLoader()
        event_bus = EventBus()
        agent = QualityAgent(config=config, event_bus=event_bus)

        # Create test dataframe with missing values
        df = pd.DataFrame({
            "col1": [1, 2, None, 4],
            "col2": ["a", "b", "c", "d"],
        })

        completeness = agent._calculate_completeness(df)
        assert 0 <= completeness <= 100
        assert completeness < 100  # Should be less than 100 due to missing value

    def test_duplicate_detection(self):
        """Test duplicate row detection."""
        config = ConfigLoader()
        event_bus = EventBus()
        agent = QualityAgent(config=config, event_bus=event_bus)

        df = pd.DataFrame({
            "col1": [1, 2, 1, 4],
            "col2": ["a", "b", "a", "d"],
        })

        duplicates = agent._detect_duplicates(df)
        assert duplicates == 1  # One duplicate row

    def test_outlier_detection(self):
        """Test outlier detection."""
        config = ConfigLoader()
        event_bus = EventBus()
        agent = QualityAgent(config=config, event_bus=event_bus)

        # Create data with outliers
        data = [1, 2, 3, 4, 5] * 10 + [100]  # One outlier
        df = pd.DataFrame({"values": data})

        outliers = agent._detect_outliers(df)
        assert outliers > 0  # Should detect the outlier


class TestAnomalyAgent:
    """Test anomaly detection."""

    def test_row_count_anomaly_detection(self):
        """Test row count anomaly detection."""
        config = ConfigLoader()
        event_bus = EventBus()
        agent = AnomalyAgent(config=config, event_bus=event_bus)

        # Create context with varying row counts
        context = PipelineContext(session_id="test")
        context.ingestion_metadata = [
            FileMetadata(filename="file1.xlsx", file_path="path1", row_count=100),
            FileMetadata(filename="file2.xlsx", file_path="path2", row_count=110),
            FileMetadata(filename="file3.xlsx", file_path="path3", row_count=500),  # Outlier
        ]

        anomalies = agent._detect_row_count_anomalies([100, 110, 500], context)
        assert len(anomalies) > 0  # Should detect the outlier


class TestRiskAgent:
    """Test risk scoring."""

    def test_score_to_level(self):
        """Test score to risk level conversion."""
        config = ConfigLoader()
        event_bus = EventBus()
        agent = RiskAgent(config=config, event_bus=event_bus)

        assert agent._score_to_level(10) == RiskLevel.LOW
        assert agent._score_to_level(40) == RiskLevel.MODERATE
        assert agent._score_to_level(60) == RiskLevel.HIGH
        assert agent._score_to_level(90) == RiskLevel.CRITICAL

    def test_composite_risk_calculation(self):
        """Test composite risk score calculation."""
        config = ConfigLoader()
        event_bus = EventBus()
        agent = RiskAgent(config=config, event_bus=event_bus)

        context = PipelineContext(session_id="test")
        context.quality_metrics["test"] = DataQualityMetrics(
            dataset_name="test",
            overall_quality_score=75.0,
        )

        risk_score = agent._calculate_composite_score(context)
        assert risk_score is not None
        assert 0 <= risk_score.overall_score <= 100


class TestInsightGenerator:
    """Test insight generation."""

    def test_local_insight_row_count(self):
        """Test insight generation for row count anomaly."""
        generator = LocalInsightGenerator()
        insight = generator.generate_insight(
            anomaly_type="row_count_deviation",
            metric_name="row_count",
            observed_value=150.0,
            expected_value=100.0,
        )
        assert isinstance(insight, str)
        assert len(insight) > 0
        assert "50.0%" in insight or "50" in insight

    def test_local_insight_missingness(self):
        """Test insight generation for high missingness."""
        generator = LocalInsightGenerator()
        insight = generator.generate_insight(
            anomaly_type="high_missingness",
            metric_name="test_column",
            observed_value=0.0,
        )
        assert isinstance(insight, str)
        assert "missing" in insight.lower()


class TestSchemaAgent:
    """Test schema harmonization."""

    def test_column_normalization(self):
        """Test column name normalization."""
        config = ConfigLoader()
        storage = Storage(Path(tempfile.gettempdir()))
        event_bus = EventBus()
        agent = SchemaAgent(
            storage=storage, config=config, event_bus=event_bus
        )

        # Test various column name variations
        assert agent._normalize_column_name("Subject ID") == "subject_id"
        assert agent._normalize_column_name("site id") == "site_id"
        assert agent._normalize_column_name("Visit Date") == "visit_date"
        assert agent._normalize_column_name("unknown_column") == "unknown_column"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_pipeline_with_synthetic_data(self):
        """Test full pipeline with synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data"
            data_dir.mkdir()

            # Create synthetic Excel file
            df = pd.DataFrame({
                "Subject ID": ["S001", "S002", "S003"],
                "Site ID": ["Site1", "Site1", "Site2"],
                "Visit Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "Value": [10.5, 11.2, 9.8],
            })

            excel_file = data_dir / "Study1" / "data.xlsx"
            excel_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(excel_file, index=False)

            # Initialize and run orchestrator
            config_path = Path(tmpdir) / "config.yaml"
            orchestrator = PipelineOrchestrator(
                config_path=config_path,
                data_base_path=tmpdir / "output",
            )

            # Copy synthetic data to raw
            import shutil
            shutil.copytree(data_dir, orchestrator.storage.raw_path, dirs_exist_ok=True)

            results = orchestrator.run(existing_dir=data_dir)

            # Verify results
            assert results["success"]
            assert results["summary"]["files_ingested"] > 0
            assert results["summary"]["datasets_harmonized"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
