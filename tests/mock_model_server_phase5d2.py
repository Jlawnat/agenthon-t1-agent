from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

HOST = "0.0.0.0"
PORT = 8000


EXEMPLAR_SOLVER = r'''
from pathlib import Path
import math
import os

import numpy as np
import pandas as pd

root = Path(os.environ.get("INPUT_DIR", "/input"))
output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
input_path = next(
    path for path in (
        root / "data" / "options.parquet",
        root / "environment" / "data" / "options.parquet",
    ) if path.is_file()
)
options = pd.read_parquet(input_path)
spot = options["S"].to_numpy(dtype=float)
strike = options["K"].to_numpy(dtype=float)
maturity = options["T"].to_numpy(dtype=float)
rate = options["r"].to_numpy(dtype=float)
volatility = options["sigma"].to_numpy(dtype=float)
option_type = options["option_type"].astype(str).to_numpy()
sqrt_t = np.sqrt(maturity)
d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * sqrt_t)
d2 = d1 - volatility * sqrt_t
discount = np.exp(-rate * maturity)
density = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
is_call = option_type == "call"
normal_cdf = lambda values: 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(values / math.sqrt(2.0)))
cdf_d1 = normal_cdf(d1)
cdf_d2 = normal_cdf(d2)
cdf_minus_d1 = normal_cdf(-d1)
cdf_minus_d2 = normal_cdf(-d2)
call_price = spot * cdf_d1 - strike * discount * cdf_d2
put_price = strike * discount * cdf_minus_d2 - spot * cdf_minus_d1
call_theta = -spot * density * volatility / (2.0 * sqrt_t) - rate * strike * discount * cdf_d2
put_theta = -spot * density * volatility / (2.0 * sqrt_t) + rate * strike * discount * cdf_minus_d2
output = pd.DataFrame({
    "option_id": options["option_id"],
    "price": np.where(is_call, call_price, put_price),
    "delta": np.where(is_call, cdf_d1, cdf_d1 - 1.0),
    "gamma": density / (spot * volatility * sqrt_t),
    "vega": spot * density * sqrt_t,
    "theta": np.where(is_call, call_theta, put_theta) / 365.0,
})
output_dir.mkdir(parents=True, exist_ok=True)
output.to_parquet(output_dir / "results.parquet", index=False)
'''.strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "AgenthonMockModel/1.0"

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        content_length = int(
            self.headers.get("Content-Length", "0")
        )
        raw = self.rfile.read(content_length)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        model = payload.get("model", "mock-model")
        messages = payload.get("messages", [])
        prompt = ""
        if messages and isinstance(messages, list):
            first = messages[0]
            if isinstance(first, dict):
                prompt = str(first.get("content", ""))

        content = (
            "MOCK_MODEL_OK"
            if prompt == "ping"
            else EXEMPLAR_SOLVER
        )

        response = {
            "id": "mock-chatcmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

        body = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"mock model listening on {HOST}:{PORT}",
        flush=True,
    )
    server.serve_forever()
