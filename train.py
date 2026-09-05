import os, json, joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

DATA = "data/power_outages.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["load_mw","voltage_v","temperature_c","humidity_pct","rainfall_mm","previous_outage"]
TARGET = "outage"
WINDOW = 24

df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp").dropna().reset_index(drop=True)

split = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()

scaler = StandardScaler()
X_train_raw = scaler.fit_transform(train_df[FEATURES])
X_test_raw = scaler.transform(test_df[FEATURES])

def make_sequences(X, y, window):
    xs, ys = [], []
    for i in range(window, len(X)):
        xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_train, y_train = make_sequences(X_train_raw, train_df[TARGET].values, WINDOW)
X_test, y_test = make_sequences(X_test_raw, test_df[TARGET].values, WINDOW)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, len(FEATURES))),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=40,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

prob = model.predict(X_test, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else None,
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "window_hours": WINDOW,
    "features": FEATURES
}

model.save(os.path.join(MODEL_DIR, "outage_lstm.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
