"""Enhanced Streamlit dashboard for clinical trial intelligence."""
import io
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.models import RiskLevel
from app.orchestrator import PipelineOrchestrator

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()
    st.session_state.pipeline_results = None
    st.session_state.last_run = None

orchestrator = st.session_state.orchestrator

# Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .risk-critical { color: #ff1744; font-weight: bold; }
    .risk-high { color: #ff9800; font-weight: bold; }
    .risk-moderate { color: #ffc107; font-weight: bold; }
    .risk-low { color: #4caf50; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR - CONFIGURATION AND CONTROLS
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Demo Dataset", "Upload ZIP"],
        key="data_source",
    )
    
    uploaded_zip = None
    demo_dir = None
    
    if data_source == "Upload ZIP":
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing Excel files",
            type="zip",
            key="zip_upload",
        )
    else:
        demo_path = st.text_input(
            "Demo data path:",
            value="Data for problem Statement 1/QC Anonymized Study Files",
            key="demo_path",
        )
        demo_dir = Path(demo_path)
    
    st.divider()
    
    st.subheader("Quality Thresholds")
    quality_threshold = st.slider(
        "Quality Score Threshold (%)",
        0,
        100,
        70,
        key="quality_threshold",
    )
    
    risk_threshold = st.slider(
        "Risk Score Threshold",
        0.0,
        100.0,
        60.0,
        key="risk_threshold",
    )
    
    st.divider()
    
    # Run button
    run_pipeline = st.button(
        "🚀 Run Pipeline",
        use_container_width=True,
        key="run_button",
    )
    
    if st.session_state.last_run:
        st.caption(f"Last run: {st.session_state.last_run}")

# ============================================================================
# MAIN PAGE
# ============================================================================
st.title("🏥 Agentic AI Platform for Clinical Trial Data Intelligence")

st.markdown(
    """
    **Multi-Agent Pipeline for Data Quality, Risk Analysis & Operational Intelligence**
    
    This platform ingests clinical trial data, harmonizes schemas, monitors quality,
    detects anomalies, calculates risk, and generates actionable insights.
    
    ⚠️ *Disclaimer: This is a prototype for decision-support only and should not be used for medical decisions.*
    """
)

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================
if run_pipeline:
    try:
        with st.spinner("🔄 Running multi-agent pipeline..."):
            if data_source == "Upload ZIP" and uploaded_zip:
                zip_bytes = uploaded_zip.read()
                temp_zip = Path("temp_upload.zip")
                temp_zip.write_bytes(zip_bytes)
                results = orchestrator.run(zip_path=temp_zip)
                temp_zip.unlink()
            elif data_source == "Demo Dataset" and demo_dir.exists():
                results = orchestrator.run(existing_dir=demo_dir)
            else:
                st.error("❌ Please select valid data source and upload file or verify demo path.")
                st.stop()
        
        st.session_state.pipeline_results = results
        st.session_state.last_run = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("✅ Pipeline execution completed!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)

# ============================================================================
# RESULTS DISPLAY
# ============================================================================
if st.session_state.pipeline_results and st.session_state.pipeline_results.get("success"):
    results = st.session_state.pipeline_results
    summary = results.get("summary", {})
    
    # Summary metrics
    st.header("📊 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Files Ingested", summary.get("files_ingested", 0))
    with col2:
        st.metric("Datasets Harmonized", summary.get("datasets_harmonized", 0))
    with col3:
        st.metric("Anomalies Detected", summary.get("anomalies_detected", 0))
    with col4:
        st.metric("Alerts Created", summary.get("alerts_created", 0))
    with col5:
        st.metric("Insights Generated", summary.get("insights_generated", 0))
    
    # ========================================================================
    # TAB 1: QUALITY METRICS
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🔍 Quality", "⚠️ Anomalies", "🎯 Risk", "🚨 Alerts", "💡 Insights", "📈 Operations"]
    )
    
    with tab1:
        st.subheader("Data Quality Assessment")
        
        quality_metrics = results.get("quality", {}).get("metrics", [])
        if quality_metrics:
            quality_df = pd.DataFrame(quality_metrics)
            
            # Quality score chart
            fig = px.bar(
                quality_df,
                x="dataset",
                y="quality_score",
                title="Quality Score by Dataset",
                color="quality_score",
                color_continuous_scale=["red", "yellow", "green"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Metrics")
            display_df = quality_df.copy()
            display_df["completeness"] = display_df["completeness"].round(2)
            display_df["validity"] = display_df["validity"].round(2)
            display_df["consistency"] = display_df["consistency"].round(2)
            display_df["quality_score"] = display_df["quality_score"].round(2)
            st.dataframe(display_df, use_container_width=True)
            
            # Issue heatmap
            st.subheader("Top Issues by Dataset")
            issue_text = []
            for metric in quality_metrics:
                if metric["issues"]:
                    issue_text.append(
                        f"**{metric['dataset']}**: " + " | ".join(metric["issues"])
                    )
            if issue_text:
                for text in issue_text:
                    st.markdown(text)
            else:
                st.info("No quality issues detected.")
        else:
            st.info("No quality metrics available yet. Run the pipeline first.")
    
    with tab2:
        st.subheader("Detected Anomalies")
        
        anomalies = results.get("anomalies", [])
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            
            # Anomaly severity distribution
            severity_counts = anomalies_df["severity"].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Anomalies by Severity",
                color_discrete_map={
                    "critical": "#ff1744",
                    "high": "#ff9800",
                    "medium": "#ffc107",
                    "low": "#4caf50",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly table
            st.subheader("Anomaly Details")
            display_df = anomalies_df[["type", "severity", "metric", "observed", "expected", "confidence", "study_id"]].copy()
            display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = anomalies_df.to_csv(index=False)
            st.download_button(
                "📥 Download Anomalies as CSV",
                csv,
                "anomalies.csv",
                "text/csv",
            )
        else:
            st.info("No anomalies detected. Data looks good!")
    
    with tab3:
        st.subheader("Risk Scoring & Prioritization")
        
        risk_scores = results.get("risk", {}).get("scores", [])
        if risk_scores:
            risk_df = pd.DataFrame(risk_scores)
            
            # Risk distribution
            risk_counts = risk_df["risk_level"].value_counts()
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Distribution",
                labels={"x": "Risk Level", "y": "Count"},
                color=risk_counts.index,
                color_discrete_map={
                    "low": "#4caf50",
                    "moderate": "#ffc107",
                    "high": "#ff9800",
                    "critical": "#ff1744",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk component breakdown
            st.subheader("Risk Components")
            risk_high = risk_df[risk_df["overall_score"] >= 60]
            if not risk_high.empty:
                fig = go.Figure()
                for _, row in risk_high.iterrows():
                    fig.add_trace(
                        go.Bar(
                            name=row["dataset"],
                            x=["Quality", "Operational", "Anomaly"],
                            y=[
                                row["quality_component"],
                                row["operational_component"],
                                row["anomaly_component"],
                            ],
                        )
                    )
                fig.update_layout(title="Risk Components (High-Risk Datasets)", barmode="group")
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk table
            st.subheader("All Risk Scores")
            display_df = risk_df[["dataset", "overall_score", "risk_level", "study_id"]].copy()
            display_df["overall_score"] = display_df["overall_score"].round(1)
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = risk_df.to_csv(index=False)
            st.download_button(
                "📥 Download Risk Scores as CSV",
                csv,
                "risk_scores.csv",
                "text/csv",
            )
        else:
            st.info("No risk scores available yet.")
    
    with tab4:
        st.subheader("Alerts & Recommended Actions")
        
        alerts = results.get("alerts", [])
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            
            # Alert severity distribution
            severity_counts = alerts_df["severity"].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Alerts by Severity",
                color_discrete_map={
                    "critical": "#ff1744",
                    "high": "#ff9800",
                    "medium": "#ffc107",
                    "low": "#4caf50",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Alert details
            st.subheader("Alert Details")
            for _, alert in alerts_df.iterrows():
                severity_color = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(alert["severity"], "⚪")
                
                with st.expander(
                    f"{severity_color} {alert['title']} ({alert['severity'].upper()})"
                ):
                    st.write(f"**Description**: {alert['description']}")
                    if alert["study_id"]:
                        st.write(f"**Study**: {alert['study_id']}")
                    if alert["site_id"]:
                        st.write(f"**Site**: {alert['site_id']}")
                    st.write(f"**Recommended Action**: {alert['recommended_action']}")
            
            # Download option
            csv = alerts_df.to_csv(index=False)
            st.download_button(
                "📥 Download Alerts as CSV",
                csv,
                "alerts.csv",
                "text/csv",
            )
        else:
            st.success("✅ No alerts generated. System is healthy!")
    
    with tab5:
        st.subheader("AI-Generated Insights")
        
        insights = results.get("insights", [])
        if insights:
            for i, insight in enumerate(insights, 1):
                with st.container():
                    st.info(f"**Insight {i}**: {insight}")
        else:
            st.info("No insights generated yet.")
    
    with tab6:
        st.subheader("Operational KPIs")
        st.info("Operational metrics summary and trends.")
        # Additional operational dashboard content can be added here

else:
    if st.session_state.pipeline_results and not st.session_state.pipeline_results.get("success"):
        st.error(f"❌ Pipeline Error: {st.session_state.pipeline_results.get('error', 'Unknown error')}")
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Pipeline' to start the analysis.")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(
    """
    ---
    **Clinical Trial Intelligence Platform** | Powered by Multi-Agent Architecture
    
    📧 Support & Documentation: See README.md for setup and usage instructions.
    """
)
