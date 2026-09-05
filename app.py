import os, json, joblib, numpy as np, pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)
MODEL_PATH = "models/outage_lstm.keras"
SCALER_PATH = "models/scaler.joblib"
METRICS_PATH = "models/metrics.json"
DATA_PATH = "data/power_outages.csv"

FEATURES = ["load_mw","voltage_v","temperature_c","humidity_pct","rainfall_mm","previous_outage"]
WINDOW = 24

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)

def predict(values):
    x = np.array(values, dtype=float).reshape(WINDOW, len(FEATURES))
    x = scaler.transform(x)
    p = float(model.predict(x[np.newaxis, ...], verbose=0)[0][0])
    return p

@app.route("/")
def index():
    df = pd.read_csv(DATA_PATH)
    latest = df.tail(WINDOW).copy()
    return render_template("index.html", metrics=metrics, latest=latest.to_dict("records"))

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True)
    rows = payload.get("rows", [])
    if len(rows) != WINDOW:
        return jsonify({"error": f"Exactly {WINDOW} hourly records are required."}), 400
    try:
        values = [[float(r[k]) for k in FEATURES] for r in rows]
        p = predict(values)
        return jsonify({
            "probability": round(p, 4),
            "prediction": int(p >= 0.5),
            "message": "High outage risk" if p >= 0.5 else "Low outage risk"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
