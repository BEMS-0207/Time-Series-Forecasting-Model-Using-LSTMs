import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000
ts = pd.date_range("2024-01-01", periods=n, freq="h")
hour = ts.hour.values
day = ts.dayofweek.values

temperature = 28 + 5*np.sin(2*np.pi*(hour-13)/24) + np.random.normal(0, 1.2, n)
humidity = 70 - 12*np.sin(2*np.pi*(hour-13)/24) + np.random.normal(0, 4, n)
rainfall = np.maximum(0, np.random.gamma(1.2, 1.2, n) * (np.random.rand(n) < 0.18))
load = 390 + 90*np.sin(2*np.pi*(hour-7)/24) + 35*(day < 5) + np.random.normal(0, 18, n)

# Synthetic voltage stress related to high demand and weather.
voltage = 230 - 0.020*(load-390) - 0.35*rainfall + np.random.normal(0, 1.5, n)

risk = (
    -5.0
    + 0.018*(load-390)
    + 0.11*np.maximum(0, 225-voltage)
    + 0.055*np.maximum(0, temperature-31)
    + 0.045*rainfall
    + 0.012*np.maximum(0, humidity-80)
    + 0.35*(day >= 5)
)
p = 1/(1+np.exp(-risk))
outage = (np.random.rand(n) < p).astype(int)

# Previous outage is based only on past information.
previous_outage = np.r_[0, outage[:-1]]

df = pd.DataFrame({
    "timestamp": ts,
    "load_mw": load.round(2),
    "voltage_v": voltage.round(2),
    "temperature_c": temperature.round(2),
    "humidity_pct": humidity.round(2),
    "rainfall_mm": rainfall.round(2),
    "previous_outage": previous_outage,
    "outage": outage
})
os.makedirs("data", exist_ok=True)
df.to_csv("data/power_outages.csv", index=False)
print(f"Saved {len(df)} rows to data/power_outages.csv")
