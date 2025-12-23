import io
from pathlib import Path

import pandas as pd
import streamlit as st

from app.orchestrator import AgentOrchestrator


st.set_page_config(page_title="Agentic Clinical Data Intelligence", layout="wide")
st.title("Agentic AI Platform for Clinical Trial Data Intelligence")

st.sidebar.header("Run Settings")
use_demo = st.sidebar.checkbox("Use demo folder (QC Anonymized Study Files)", value=True)
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of Excel files", type=["zip"])

status = st.empty()
orchestrator = AgentOrchestrator()

def run_pipeline(zip_bytes: bytes | None, use_demo_folder: bool):
    if zip_bytes:
        tmp_path = Path("data/uploads")
        tmp_path.mkdir(parents=True, exist_ok=True)
        zip_file = tmp_path / "upload.zip"
        zip_file.write_bytes(zip_bytes)
        return orchestrator.run(zip_path=zip_file)
    if use_demo_folder:
        demo_root = Path(orchestrator.config["paths"]["demo_data_root"])
        return orchestrator.run(existing_dir=demo_root)
    st.warning("Please upload a ZIP or enable demo folder.")
    return None


if st.sidebar.button("Run Pipeline"):
    with st.spinner("Running multi-agent pipeline..."):
        result = run_pipeline(uploaded_zip.read() if uploaded_zip else None, use_demo)
    if result is None:
        st.stop()

    st.success("Pipeline completed")

    st.subheader("Risk Scores")
    risk_df = pd.DataFrame(result["risk"].risk_scores)
    st.dataframe(risk_df)

    st.subheader("Quality Metrics")
    qual_df = pd.DataFrame(
        [{"dataset": k, "quality_score": v} for k, v in result["quality"].scores.items()]
    )
    st.dataframe(qual_df)

    st.subheader("Operational KPIs")
    st.dataframe(pd.DataFrame(result["operations"].kpis))

    st.subheader("Anomalies")
    st.dataframe(pd.DataFrame(result["anomalies"].anomalies))

    st.subheader("Insights")
    for n in result["insights"].narratives:
        st.write(f"- {n}")

    st.subheader("Alerts / Tasks")
    st.json({"alerts": result["alerts"].alerts, "tasks": result["alerts"].tasks})
else:
    st.info("Configure inputs in the sidebar and click 'Run Pipeline'.")

