"""Small read-only API for the Vercel-hosted results dashboard."""

from __future__ import annotations

import csv
import json
from ast import literal_eval
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_DIR / "model_metrics.txt"
PREDICTIONS_PATH = PROJECT_DIR / "predictions.csv"


def read_metrics() -> dict[str, str | list[str]]:
    metrics: dict[str, str | list[str]] = {}
    if not METRICS_PATH.exists():
        return metrics

    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("=") or line.startswith("MODEL"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if key.strip() == "Features":
            try:
                parsed = literal_eval(value)
                value = parsed if isinstance(parsed, list) else value
            except (ValueError, SyntaxError):
                pass
        metrics[key.strip()] = value
    return metrics


def read_predictions() -> list[dict[str, float]]:
    if not PREDICTIONS_PATH.exists():
        return []

    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
        return [
            {
                key: float(value)
                for key, value in row.items()
                if key and value not in (None, "")
            }
            for row in csv.DictReader(file)
        ]


def response_payload() -> dict[str, object]:
    metrics = read_metrics()
    return {
        "status": "ready" if metrics else "missing-artifacts",
        "metrics": {
            key: metrics[key]
            for key in ("MAE", "MSE", "RMSE", "R²")
            if key in metrics
        },
        "configuration": {
            key: value
            for key, value in metrics.items()
            if key not in {"MAE", "MSE", "RMSE", "R²"}
        },
        "predictions": read_predictions(),
        "charts": [
            "/1_distribution_analysis.png",
            "/2_training_loss.png",
            "/3_actual_vs_predicted.png",
            "/4_scatter_plot.png",
            "/5_residual_plot.png",
        ],
    }


class handler(BaseHTTPRequestHandler):
    """Vercel Python runtime entry point."""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path not in {"/api", "/api/", "/api/index.py", "/api/results", "/api/health"}:
            self.send_error(404, "Not found")
            return

        payload = {"status": "ok"} if path == "/api/health" else response_payload()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, max-age=60")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return
