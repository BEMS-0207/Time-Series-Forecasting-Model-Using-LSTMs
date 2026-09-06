from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_FILES = [
    "1_distribution_analysis.png",
    "2_training_loss.png",
    "3_actual_vs_predicted.png",
    "4_scatter_plot.png",
    "5_residual_plot.png",
]
DATASET_PATH = PROJECT_DIR / "Outage_Data.xlsx"

st.set_page_config(
    page_title="Power Outage Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_metrics():
    metrics_path = PROJECT_DIR / "model_metrics.txt"
    if not metrics_path.exists():
        return {}

    metrics = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("=") or line.startswith("MODEL"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            metrics[key.strip()] = value.strip()
    return metrics


@st.cache_data
def load_predictions():
    predictions_path = PROJECT_DIR / "predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame()
    return pd.read_csv(predictions_path)


@st.cache_data
def load_dataset_preview():
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    return pd.read_excel(DATASET_PATH, header=5, skiprows=[6], engine="openpyxl")

st.title("DEPARTMENT OF COMPUTER SCIENCE, FACULTY OF COMPUTING, UNIVERSITY OF CALABAR")
st.title("OHAEGBULAM NMESOMA I . 22/095244209")


st.title("Power Outage Duration Forecasting Dashboard")
st.caption("LSTM model output, evaluation metrics, and prediction plots")

dataset = load_dataset_preview()
if dataset.empty:
    st.warning("The bundled Outage_Data.xlsx dataset is not available yet.")
else:
    st.subheader("Dataset inspection")
    info_cols = st.columns(4)
    info_cols[0].metric("Original records", f"{len(dataset):,}")
    info_cols[1].metric("Variables", f"{len(dataset.columns):,}")
    info_cols[2].metric("Missing target values", f"{dataset['OUTAGE.DURATION'].isna().sum():,}")
    info_cols[3].metric("Target", "Outage duration (minutes)")
    st.dataframe(dataset.head(10), use_container_width=True)

if st.button("Train model and refresh results"):
    with st.spinner("Running the forecasting pipeline. This may take a minute..."):
        try:
            subprocess.run([sys.executable, str(PROJECT_DIR / "app.py")], check=True)
        except subprocess.CalledProcessError as exc:
            st.error(f"Training failed with exit code {exc.returncode}. Check the terminal output for details.")
            st.stop()
    st.success("Training run completed. Refreshing results...")
    st.rerun()

metrics = load_metrics()
if metrics:
    st.subheader("Evaluation metrics")
    metric_items = [
        ("MAE", metrics.get("MAE", "-")),
        ("MSE", metrics.get("MSE", "-")),
        ("RMSE", metrics.get("RMSE", "-")),
        ("R²", metrics.get("R²", "-")),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metric_items):
        col.metric(label, value)

    with st.expander("Preprocessing and model configuration"):
        configuration_keys = [
            "Original Records", "Original Variables", "Missing Target Records",
            "Total Records", "Features Used", "Sequence Length", "Training Samples",
            "Testing Samples", "Train/Test Split", "Epochs Trained", "Batch Size",
            "LSTM Layers", "LSTM Units", "Dense Layers", "Activation Functions",
            "Dropout", "Optimizer", "Loss Function", "Features",
        ]
        for key in configuration_keys:
            if key in metrics:
                st.write(f"**{key}:** {metrics[key]}")
else:
    st.info("No saved metrics were found yet. Click 'Train model and refresh results' to generate them.")

st.subheader("Generated charts")
chart_cols = st.columns(2)
for index, filename in enumerate(OUTPUT_FILES):
    path = PROJECT_DIR / filename
    if not path.exists():
        st.info("This chart is not available yet. Train the model to generate it.")
        continue
    with chart_cols[index % 2]:
        st.image(str(path), use_container_width=True)

st.subheader("Predictions")
predictions = load_predictions()
if predictions.empty:
    st.info("No prediction file is available yet.")
else:
    st.dataframe(predictions.head(20), use_container_width=True)

st.subheader("Project files")
for filename in ["model_metrics.txt", "predictions.csv"]:
    path = PROJECT_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            st.code(file.read(), language="text")
