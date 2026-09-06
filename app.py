"""SOFTWARE DEVELOPMENT BRIEF

1. Project Title

Development of Time-Series Forecasting Model Using LSTMs for Prediction of Power Outages

Specific prediction target

The system will predict power outage duration, measured in minutes.


2. Purpose of the Software

The purpose of the software is to develop a machine-learning-based forecasting system that uses historical power-outage records to learn temporal patterns and predict the expected duration of a power outage.

The system will use a Long Short-Term Memory (LSTM) neural network because LSTM is designed to learn patterns from sequential/time-dependent data.


3. Dataset

The software should use an existing public dataset, not a newly collected dataset.

The dataset we have selected is the Major Power Outage Events in the Continental U.S. dataset. It covers major outages from January 2000 to July 2016 and contains 1,534 recorded outage events. The data includes outage timing, location, cause, duration, customers affected and other regional/economic variables. The original data was compiled from public sources including the U.S. Department of Energy's OE-417 outage reports and other government datasets. 

The important target variable is:

OUTAGE.DURATION — duration of the outage in minutes. 

Important instruction to the developer

Do not invent the dataset size, variables, missing values, or train-test split. Load the actual dataset, inspect it, and report what is actually found.


4. Main Functions of the Software

The software should be able to:

1. Load the historical power-outage dataset.


2. Display/inspect the dataset.


3. Clean and preprocess the data.


4. Handle missing or unsuitable values appropriately.


5. Select the relevant variables.


6. Convert relevant date/time information into usable numerical/time-series features.


7. Arrange the observations chronologically.


8. Prepare the data into sequences suitable for an LSTM.


9. Normalize/scale numerical data where appropriate.


10. Split the data into training and testing sets.


11. Build the LSTM model.


12. Train the LSTM model.


13. Test the trained model.


14. Generate predicted outage durations.


15. Compare actual and predicted outage durations.


16. Calculate MAE, MSE, RMSE and R².


17. Display graphs of the model's results.


18. Save/export the relevant results if possible.


5. Proposed Software Workflow

The software should follow this general process:

Dataset → Data Cleaning → Feature Selection → Time-Series Preparation → Data Scaling → Train/Test Split → LSTM Model → Model Training → Prediction → Evaluation → Results

This should eventually be represented as a proper flow diagram in the project, not simply written with arrows.


6. LSTM Model

The machine-learning model must qbe:

Long Short-Term Memory (LSTM)

The developer should implement an LSTM model using Python and TensorFlow/Keras or an equivalent appropriate framework.

The basic structure should be:

Input → LSTM Layer → Dense Layer → Predicted Outage Duration

The exact number of LSTM units, epochs, batch size and other hyperparameters should be selected during implementation and documented. They should not simply be invented for the report.

The developer should record the final model architecture because we will need to describe it in Chapter 3.5 Model Selection and Development and show it in the software/results.


7. Data Splitting

Because this is a time-series forecasting project, the data should be divided chronologically rather than randomly shuffled.

For example, the developer may use a training portion followed by a testing portion, such as 80% training and 20% testing, if appropriate after inspecting the dataset.

Important: The final percentage and exact number of records must be documented after implementation and used consistently in Chapter 3.


8. Model Evaluation

The software MUST calculate these four metrics because they were specifically requested by my supervisor:

MAE

Mean Absolute Error

MSE

Mean Squared Error

RMSE

Root Mean Squared Error

R²

Coefficient of Determination

The software should display the numerical value of all four metrics.

These values will later be inserted into Chapter Four, Results and Discussion.


9. Required Graphs/Visualizations

The software should generate clear graphs that can be captured as screenshots and included in Chapter Four.

At minimum, I need:

1. Actual vs Predicted Outage Duration

A line graph showing:

Actual outage duration

Predicted outage duration


This is particularly important because it visually demonstrates how well the LSTM model predicts outage duration.

2. Training and Validation Loss

A graph showing model loss during training.

This can help demonstrate whether the model learned appropriately and whether there are signs of overfitting.

3. Additional useful visualization

If appropriate, include a graph showing the distribution or pattern of outage duration over time.

The developer should not create unnecessary graphs just to make the software look complicated.


10. Software Interface

Because this is an undergraduate project, keep the interface simple.

The system does NOT need to be an elaborate commercial application.

A simple interface should allow the user to:

Load/select the dataset

Start preprocessing

Train the model

Run prediction

View predicted results

View MAE, MSE, RMSE and R²

View the graphs


A simple Streamlit interface would be suitable if the developer wants to make it a proper user-facing application.

Alternatively, if the supervisor only requires a working ML software demonstration, a well-organized Python/Jupyter implementation can serve as the underlying system, with a simple interface added if necessary.

My preference: Keep it simple and functional rather than spending time building an elaborate website.


11. Screenshots Required for Chapter Four

The developer should make sure the software produces screens/results that we can capture for the project report.

We will need screenshots of:

1. Dataset loaded into the system


2. Data preprocessing


3. LSTM model/training process


4. Training results


5. Actual vs predicted graph


6. Evaluation metrics


7. Software prediction/result interface



These screenshots will become evidence of the actual implementation in Chapter Four.



12. Information the Developer Must Give Me

When the software is completed, the developer should provide me with the following:

Dataset

Dataset name

Original source

Number of records before preprocessing

Number of variables

Relevant variables used

Number of records after preprocessing

Missing values handled

Final train/test sizes

Exact train/test split used


Model

LSTM architecture

Number of LSTM layers

Number of units

Dense layer configuration

Activation functions

Optimizer

Loss function

Number of epochs

Batch size

Any other important hyperparameters


Results

MAE

MSE

RMSE

R²

Training/validation loss

Actual vs predicted results

Graphs


This information is very important because we will use the actual values when correcting Chapter Three and writing Chapter Four.


13. Important Restrictions

Please tell the developer:

Do not fabricate results.

Do not simply give me impressive-looking accuracy numbers.

Do not randomly choose values and write them into the report.

The model must actually be trained using the selected dataset, and the metrics must come from the actual test results.

Also, the implementation should be kept simple enough for an undergraduate student to explain to a supervisor.

If my supervisor asks:

> "How did you get this result?"



I need to be able to explain the process.


14. Final Expected System

At the end, the software should demonstrate:

> Historical Power-Outage Data → Preprocessing → LSTM Training → Outage Duration Prediction → Evaluation → Graphical Results



The primary output of the system is:

Predicted Power Outage Duration

with model performance reported using:

MAE, MSE, RMSE and R².


One important correction before you send this

There is one thing I wouldn't tell the developer to hard-code yet: the exact train-test percentage, exact LSTM units, epochs, batch size, or exact features.

Those should come after they inspect the actual dataset and run the model. Your supervisor specifically wants the dataset description and methodology to reflect what was actually done.

Also, the dataset's original publication is from 2018, which is fine for describing the dataset/source; your supervisor's 2019/2020–2026 restriction applies to your research literature/citations, not necessarily the original dataset itself.

LSTM Power Outage Duration Prediction System
Compatible with Python 3.12 and TensorFlow 2.16+
"""

from pathlib import Path
import os
import json
import tempfile
import urllib.request
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR
DATASET_NAMES = ("outage data.xlsx", "Outage_Data.xlsx")
DATASET_URL = os.getenv(
    "OUTAGE_DATA_URL",
    "https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks/Outage_Data.xlsx",
)

print("=" * 80)
print("LSTM POWER OUTAGE DURATION PREDICTION SYSTEM")
print("=" * 80)


def load_dataset():
    local_sources = [PROJECT_DIR / name for name in DATASET_NAMES]
    for source in local_sources:
        if not source.is_file() or source.stat().st_size == 0:
            continue
        try:
            with zipfile.ZipFile(source):
                pass
            df = pd.read_excel(source, header=5, skiprows=[6], engine="openpyxl")
            print(f"✓ Dataset loaded from: {source} | {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as exc:
            print(f"  - Could not load {source}: {type(exc).__name__}: {exc}")

    print(f"  - Local workbook not found or invalid; downloading from {DATASET_URL}")
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, dir=PROJECT_DIR) as temporary:
            temporary_path = Path(temporary.name)
        try:
            urllib.request.urlretrieve(DATASET_URL, temporary_path)
            if temporary_path.stat().st_size == 0:
                raise ValueError("the downloaded file is empty")
            with zipfile.ZipFile(temporary_path):
                pass
            dataset_path = PROJECT_DIR / "Outage_Data.xlsx"
            temporary_path.replace(dataset_path)
            df = pd.read_excel(dataset_path, header=5, skiprows=[6], engine="openpyxl")
            print(f"✓ Dataset downloaded to: {dataset_path} | {len(df)} rows, {len(df.columns)} columns")
            return df
        finally:
            temporary_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"  - Could not download {DATASET_URL}: {type(exc).__name__}: {exc}")

    raise FileNotFoundError(
        "Could not load the outage dataset. Place 'outage data.xlsx' or 'Outage_Data.xlsx' "
        "in the project folder, or set OUTAGE_DATA_URL to a valid .xlsx URL."
    )


def find_target_column(df):
    candidates = [
        "OUTAGE.DURATION",
        "OUTAGE DURATION",
        "DURATION",
        "Outage Duration",
        "outage_duration",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        if "duration" in str(col).lower():
            return col

    raise ValueError(f"No target duration column found. Available columns: {list(df.columns)}")


def prepare_features(data, target_col):
    data = data.copy()

    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    initial_rows = len(data)
    data = data.dropna(subset=[target_col])
    print(f"  - Removed {initial_rows - len(data)} rows with missing target values")

    start_date = pd.to_datetime(data["OUTAGE.START.DATE"], errors="coerce")
    start_time = pd.to_timedelta(data["OUTAGE.START.TIME"].astype(str), errors="coerce")
    data["OUTAGE.START.DATETIME"] = start_date + start_time
    data = data.sort_values("OUTAGE.START.DATETIME", na_position="last").reset_index(drop=True)
    data["START_YEAR"] = data["OUTAGE.START.DATETIME"].dt.year
    data["START_MONTH"] = data["OUTAGE.START.DATETIME"].dt.month
    data["START_DAYOFYEAR"] = data["OUTAGE.START.DATETIME"].dt.dayofyear
    data["START_DAYOFWEEK"] = data["OUTAGE.START.DATETIME"].dt.dayofweek
    data["START_MONTH_SIN"] = np.sin(2 * np.pi * data["START_MONTH"].fillna(0) / 12)
    data["START_MONTH_COS"] = np.cos(2 * np.pi * data["START_MONTH"].fillna(0) / 12)

    data = data[data[target_col] > 0]
    print(f"  - Removed rows with zero or negative outage durations")

    excluded = {
        target_col,
        "YEAR",
        "MONTH",
        "DAY",
        "OUTAGE.START.DATE",
        "OUTAGE.START.TIME",
        "OUTAGE.RESTORATION.DATE",
        "OUTAGE.RESTORATION.TIME",
        "OUTAGE.START.DATETIME",
    }
    numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in excluded]
    features = [col for col in numeric_cols if data[col].nunique(dropna=True) > 1]

    print(f"  - Selected {len(features)} numeric features")
    if not features:
        raise ValueError("No usable numeric feature columns were found after cleaning the data.")

    for col in features:
        if data[col].isnull().sum() > 0:
            median_value = float(data[col].median())
            data[col] = data[col].fillna(median_value)
            print(f"  - Filled missing values in '{col}' with median={median_value:.4f}")

    X = data[features].values.astype(np.float32)
    y = data[target_col].astype(float).values.astype(np.float32)
    return X, y, features, data


def create_sequences(X, y, seq_length):
    if len(X) <= seq_length:
        raise ValueError(f"Not enough data to create sequences. Need at least {seq_length + 1}, got {len(X)}.")

    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length : i])
        y_seq.append(y[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


print("\n[1/8] Loading dataset...")
try:
    df = load_dataset()
except Exception as exc:
    print(f"\n❌ Fatal error: {exc}\n")
    raise SystemExit(1)

print(f"\nDataset Info:")
print(f"  - Columns: {list(df.columns)}")
print(f"  - Missing values: {int(df.isnull().sum().sum())}")

print("\n[2/8] Preprocessing data...")
target_col = find_target_column(df)
print(f"  - Target column identified: '{target_col}'")
X, y, features, prepared_data = prepare_features(df, target_col)
print(f"✓ Final dataset: {len(y)} records")

print("\n[3/8] Exploratory Data Analysis...")
print("\nTarget Variable Statistics:")
print(f"  - Mean: {np.mean(y):.2f} minutes")
print(f"  - Median: {np.median(y):.2f} minutes")
print(f"  - Std Dev: {np.std(y):.2f} minutes")
print(f"  - Min: {np.min(y):.2f} minutes")
print(f"  - Max: {np.max(y):.2f} minutes")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=50, edgecolor="black", alpha=0.7)
axes[0].axvline(np.mean(y), color="red", linestyle="--", label=f"Mean: {np.mean(y):.0f} min")
axes[0].axvline(np.median(y), color="green", linestyle="--", label=f"Median: {np.median(y):.0f} min")
axes[0].set_xlabel("Outage Duration (minutes)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Outage Durations")
axes[0].legend()

axes[1].boxplot(y)
axes[1].set_ylabel("Outage Duration (minutes)")
axes[1].set_title("Box Plot of Outage Durations")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "1_distribution_analysis.png", dpi=300, bbox_inches="tight")
print("✓ Saved distribution plot as '1_distribution_analysis.png'")

print("\n[4/8] Scaling data...")
split_row = int(len(X) * 0.8)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(X[:split_row])
scaler_y.fit(y[:split_row].reshape(-1, 1))
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
print("✓ Data scaled to [0, 1] range")

print("\n[5/8] Creating sequences...")
seq_length = min(30, max(5, int(len(y) * 0.05)))
print(f"  - Sequence length: {seq_length}")
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
print(f"✓ Created {len(X_seq)} sequences")

print("\n[6/8] Splitting data chronologically...")
split_idx = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]

print(f"  - Training: {len(X_train)} samples ({len(X_train) / len(X_seq) * 100:.1f}%)")
print(f"  - Testing:  {len(X_test)} samples ({len(X_test) / len(X_seq) * 100:.1f}%)")

if len(X_train) < 8 or len(X_test) < 1:
    raise ValueError("Not enough samples to train or evaluate the LSTM model.")

print("\n[7/8] Building and training LSTM model...")
model = Sequential(
    [
        LSTM(64, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ]
)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("\nModel Architecture:")
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    shuffle=False,
    verbose=0,
)
print(f"✓ Model trained for {len(history.history['loss'])} epochs")

print("\n[8/8] Evaluating model...")
y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"\n  MAE  (Mean Absolute Error):  {mae:.2f} minutes")
print(f"  MSE  (Mean Squared Error):   {mse:.2f} minutes²")
print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} minutes")
print(f"  R²   (Coefficient of Determination): {r2:.4f}")
print("=" * 60)

print("\nGenerating visualizations...")

fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history["loss"], label="Training Loss", linewidth=2)
if "val_loss" in history.history:
    ax.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.set_title("Training and Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "2_training_loss.png", dpi=300, bbox_inches="tight")
print("✓ Saved training loss plot as '2_training_loss.png'")

fig2, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(len(y_actual)), y_actual, "b-", label="Actual Duration", linewidth=2, alpha=0.8)
ax.plot(range(len(y_pred)), y_pred, "r-", label="Predicted Duration", linewidth=2, alpha=0.8)
ax.set_xlabel("Test Sample Index")
ax.set_ylabel("Outage Duration (minutes)")
ax.set_title("Actual vs Predicted Outage Duration (Test Set)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "3_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
print("✓ Saved actual vs predicted plot as '3_actual_vs_predicted.png'")

fig3, ax = plt.subplots(figsize=(8, 8))
min_val = min(np.min(y_actual), np.min(y_pred))
max_val = max(np.max(y_actual), np.max(y_pred))
ax.scatter(y_actual, y_pred, alpha=0.5, s=30)
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Outage Duration (minutes)")
ax.set_ylabel("Predicted Outage Duration (minutes)")
ax.set_title(f"Predicted vs Actual (R² = {r2:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(OUTPUT_DIR / "4_scatter_plot.png", dpi=300, bbox_inches="tight")
print("✓ Saved scatter plot as '4_scatter_plot.png'")

fig4, ax = plt.subplots(figsize=(12, 5))
residuals = y_actual - y_pred
ax.scatter(y_pred, residuals, alpha=0.5, s=30)
ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
ax.set_xlabel("Predicted Duration (minutes)")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Residual Plot")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig(OUTPUT_DIR / "5_residual_plot.png", dpi=300, bbox_inches="tight")
print("✓ Saved residual plot as '5_residual_plot.png'")

print("\nSaving results...")
with open(OUTPUT_DIR / "model_metrics.txt", "w", encoding="utf-8") as f:
    f.write("LSTM POWER OUTAGE DURATION PREDICTION - MODEL METRICS\n")
    f.write("=" * 60 + "\n")
    f.write(f"MAE:  {mae:.4f} minutes\n")
    f.write(f"MSE:  {mse:.4f} minutes²\n")
    f.write(f"RMSE: {rmse:.4f} minutes\n")
    f.write(f"R²:   {r2:.4f}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("MODEL CONFIGURATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"Total Records: {len(y)}\n")
    f.write(f"Original Records: {len(df)}\n")
    f.write(f"Original Variables: {len(df.columns)}\n")
    f.write(f"Missing Target Records: {int(pd.to_numeric(df[target_col], errors='coerce').isna().sum())}\n")
    f.write(f"Features Used: {len(features)}\n")
    f.write(f"Sequence Length: {seq_length}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n")
    f.write("Train/Test Split: 80% / 20% chronological\n")
    f.write(f"Epochs Trained: {len(history.history['loss'])}\n")
    f.write("Batch Size: 32\n")
    f.write("LSTM Layers: 2\n")
    f.write("LSTM Units: 64, 32\n")
    f.write("Dense Layers: 16 ReLU, 1 linear\n")
    f.write("Activation Functions: tanh (LSTM), relu, linear\n")
    f.write("Dropout: 0.2\n")
    f.write("Optimizer: Adam\n")
    f.write("Loss Function: MSE\n")
    f.write(f"Features: {features}\n")
print("✓ Saved metrics to 'model_metrics.txt'")

predictions_df = pd.DataFrame({
    "Actual": y_actual,
    "Predicted": y_pred,
    "Residual": residuals,
})
predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
print("✓ Saved predictions to 'predictions.csv'")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. 1_distribution_analysis.png - Outage duration distribution")
print("  2. 2_training_loss.png - Training and validation loss")
print("  3. 3_actual_vs_predicted.png - Actual vs predicted line plot")
print("  4. 4_scatter_plot.png - Scatter plot with R²")
print("  5. 5_residual_plot.png - Residual analysis")
print("  6. model_metrics.txt - All evaluation metrics")
print("  7. predictions.csv - Actual and predicted values")
print("\n" + "=" * 80) 