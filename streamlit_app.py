from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Outage_Data.xlsx"
PROGRESS_PREFIX = "TRAINING_PROGRESS "

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


def load_predictions():
    predictions_path = PROJECT_DIR / "predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame()
    return pd.read_csv(predictions_path)


def load_training_history():
    history_path = PROJECT_DIR / "training_history.csv"
    if not history_path.exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


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
    st.subheader("Live training progress")
    progress_chart = st.empty()
    progress_status = st.empty()
    training_log = st.empty()
    epochs = []
    loss_values = []
    validation_loss_values = []
    log_lines = []

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-u", str(PROJECT_DIR / "app.py")],
        cwd=PROJECT_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        line = line.rstrip()
        if line.startswith(PROGRESS_PREFIX):
            try:
                update = json.loads(line[len(PROGRESS_PREFIX):])
            except json.JSONDecodeError:
                continue
            if update.get("event") == "epoch":
                epochs.append(update["epoch"])
                loss_values.append(update.get("loss"))
                validation_loss_values.append(update.get("val_loss"))
                progress_data = pd.DataFrame(
                    {
                        "Epoch": epochs,
                        "Training loss": loss_values,
                        "Validation loss": validation_loss_values,
                    }
                ).set_index("Epoch")
                progress_chart.line_chart(
                    progress_data,
                )
                progress_status.info(f"Completed epoch {update['epoch']} of up to 100")
        elif line:
            log_lines.append(line)
            training_log.code("\n".join(log_lines[-12:]), language="text")

    return_code = process.wait()
    if return_code:
        st.error(f"Training failed with exit code {return_code}. See the training log above.")
        st.stop()

    progress_status.success(f"Training completed after {len(epochs)} epoch(s).")
    load_metrics.clear()
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

history = load_training_history()
predictions = load_predictions()

st.subheader("Generated charts")
st.caption("Five combined charts generated from the complete evaluation result.")
chart_columns = st.columns(2)

with chart_columns[0]:
    st.write("1. Outage duration distribution")
    if dataset.empty:
        st.info("Dataset data is unavailable.")
    else:
        durations = pd.to_numeric(dataset["OUTAGE.DURATION"], errors="coerce").dropna()
        durations = durations[durations > 0]
        distribution = pd.cut(durations, bins=20).value_counts().sort_index().rename("Records")
        distribution.index = distribution.index.astype(str)
        st.bar_chart(distribution, height=300)

with chart_columns[1]:
    st.write("2. Training and validation loss")
    if history.empty:
        st.info("Train the model to generate this chart.")
    else:
        st.line_chart(history.set_index("Epoch")[["Training Loss", "Validation Loss"]], height=300)

with chart_columns[0]:
    st.write("3. Actual versus predicted")
    if predictions.empty:
        st.info("Train the model to generate this chart.")
    else:
        prediction_chart = predictions[["Actual", "Predicted"]].copy()
        prediction_chart.index.name = "Test sample"
        st.line_chart(prediction_chart, height=300)

with chart_columns[1]:
    st.write("4. Prediction comparison")
    if predictions.empty:
        st.info("Train the model to generate this chart.")
    else:
        comparison = predictions[["Actual", "Predicted"]].rename(
            columns={"Actual": "Actual duration", "Predicted": "Predicted duration"}
        )
        st.scatter_chart(comparison, x="Actual duration", y="Predicted duration", height=300)

with chart_columns[0]:
    st.write("5. Residual analysis")
    if predictions.empty:
        st.info("Train the model to generate this chart.")
    else:
        residual_chart = predictions[["Predicted", "Residual"]].rename(
            columns={"Predicted": "Predicted duration", "Residual": "Residual"}
        )
        st.scatter_chart(residual_chart, x="Predicted duration", y="Residual", height=300)

st.subheader("Predictions")
if predictions.empty:
    st.info("No prediction file is available yet.")
else:

    st.dataframe(predictions.head(20), use_container_width=True)

st.subheader("Project files")
for filename in ["model_metrics.txt", "training_history.csv", "predictions.csv"]:
    path = PROJECT_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            st.code(file.read(), language="text")
