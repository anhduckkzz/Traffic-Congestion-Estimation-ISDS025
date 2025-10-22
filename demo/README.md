# Demo Web Application

This directory hosts a Streamlit interface that drives the end-to-end traffic congestion pipeline described in the paper. It wraps the existing Python modules so that an uploaded traffic video can be processed without using the command-line script.

## Getting Started

1. Install the project dependencies (adds Streamlit):  
   `pip install -r requirements.txt`
2. Launch the app from the project root:  
   `streamlit run demo/app.py`
3. Open the URL printed by Streamlit (usually http://localhost:8501).
4. Provide the path to your pipeline configuration if it differs from `configs/pipeline.yaml`.
5. Upload a traffic video and press **Run Full Pipeline**.

The app produces a per-window lane metrics table and offers the same JSON export structure as the CLI runner.

## Requirements and Tips

- Ensure the segmentation and detection checkpoints referenced in the configuration exist (defaults expect weights under `outputs/`).
- A CUDA-capable GPU is recommended; switch to CPU in the sidebar when GPU is unavailable.
- Processing time scales with the length and resolution of the uploaded clip.

