"""Streamlit web interface for the ISDS traffic congestion pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from demo.pipeline_runner import results_to_json, results_to_records, run_pipeline, summarize_by_lane
from src.utils import configure_logging

try:  # Optional guard so the UI can inform the user when PyTorch is missing.
    import torch  # type: ignore
except Exception:  # pragma: no cover - import guard for environments without PyTorch
    torch = None  # type: ignore


if "logging_configured" not in st.session_state:
    configure_logging()
    st.session_state["logging_configured"] = True


st.set_page_config(page_title="ISDS Traffic Congestion Demo", layout="wide")

st.title("ISDS Traffic Congestion Pipeline Demo")
st.write(
    "Upload a traffic video to execute the congestion estimation pipeline described in the ISDS paper. "
    "The app loads the same YAML configuration used by the command-line runner and produces the per-window "
    "lane metrics and JSON payload."
)

with st.sidebar:
    st.header("Pipeline Settings")
    config_path_str = st.text_input("Configuration file", value="configs/pipeline.yaml")
    config_path = Path(config_path_str).expanduser()

    device_options = ["cpu"]
    if torch is not None and torch.cuda.is_available():
        device_options.insert(0, "cuda")
    device_label = st.selectbox("Execution device", options=device_options, index=0)

    if not config_path.exists():
        st.warning(f"Config file not found: {config_path}")
    if torch is None:
        st.error("PyTorch is not installed or failed to import; the pipeline cannot run until it is available.")

st.divider()

uploaded_video = st.file_uploader(
    "Traffic video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Provide a short clip focused on lane-level traffic. Longer videos will take more time to process.",
)

st.caption(
    "Tip: Ensure the configuration points to valid segmentation and detection checkpoints. "
    "The defaults expect weights under `outputs/`."
)

run_clicked = st.button("Run Full Pipeline", type="primary")

if run_clicked:
    if uploaded_video is None:
        st.warning("Please upload a traffic video before running the pipeline.")
    elif not config_path.exists():
        st.error(f"Unable to locate the configuration file at {config_path}.")
    elif torch is None:
        st.error("PyTorch is required to run the pipeline. Install the project requirements and restart the app.")
    else:
        tmp_path: Optional[Path] = None
        try:
            suffix = Path(uploaded_video.name or "upload.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_video.getbuffer())
                tmp_file.flush()
                tmp_path = Path(tmp_file.name)

            overrides = {
                "segmentation": {"device": device_label},
                "tracking": {"device": device_label},
            }

            with st.spinner("Running congestion pipeline. This may take several minutes for long videos."):
                results, effective_config = run_pipeline(
                    video_path=tmp_path,
                    config_path=config_path,
                    overrides=overrides,
                )
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            st.error(f"Pipeline execution failed: {exc}")
        else:
            if not results:
                st.info(
                    "Pipeline completed without producing any window-level metrics. "
                    "Verify that the input video contains clear lane markings and sufficient traffic."
                )
            else:
                records = results_to_records(results)
                st.success(
                    f"Pipeline complete. Generated {len(records)} lane-window metrics across {len(results)} time windows."
                )

                st.subheader("Per-window Lane Metrics")
                st.dataframe(records, use_container_width=True)

                lane_summary = summarize_by_lane(records)
                if lane_summary:
                    st.subheader("Lane Summary")
                    st.table(lane_summary)

                json_payload = results_to_json(results)
                st.download_button(
                    label="Download JSON Results",
                    data=json.dumps(json_payload, indent=2),
                    file_name="pipeline_results.json",
                    mime="application/json",
                    use_container_width=False,
                )

                with st.expander("Effective configuration"):
                    st.json(effective_config)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
