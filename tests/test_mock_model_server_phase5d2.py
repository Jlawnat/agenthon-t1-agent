from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

HOST = "0.0.0.0"
PORT = 8000

EXPECTED_PATH = "/v1/chat/completions"
EXPECTED_AUTH = "Bearer test-token"

SOLVER_CODE = r"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _input_file(input_root: Path) -> Path:
    candidates = (
        input_root / "data" / "options.parquet",
        input_root / "environment" / "data" / "options.parquet",
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "options.parquet not found under INPUT_DIR"
    )


def main() -> None:
    input_root = Path(
        os.environ.get("INPUT_DIR", "/input")
    )
    output_dir = Path(
        os.environ.get("OUTPUT_DIR", "/output")
    )

    df = pd.read_parquet(
        _input_file(input_root)
    )

    s = df["S"].to_numpy(dtype=float)
    k = df["K"].to_numpy(dtype=float)
    t = df["T"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)

    sqrt_t = np.sqrt(t)
    d1 = (
        np.log(s / k)
        + (r + 0.5 * sigma * sigma) * t
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    disc = np.exp(-r * t)
    pdf_d1 = norm.pdf(d1)

    call = (
        df["option_type"]
        .astype(str)
        .str.lower()
        .to_numpy()
        == "call"
    )

    price_call = (
        s * norm.cdf(d1)
        - k * disc * norm.cdf(d2)
    )
    price_put = (
        k * disc * norm.cdf(-d2)
        - s * norm.cdf(-d1)
    )

    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1.0

    gamma = (
        pdf_d1
        / (s * sigma * sqrt_t)
    )

    vega = (
        s * pdf_d1 * sqrt_t
    )

    theta_call_annual = (
        -s * pdf_d1 * sigma
        / (2.0 * sqrt_t)
        - r * k * disc * norm.cdf(d2)
    )
    theta_put_annual = (
        -s * pdf_d1 * sigma
        / (2.0 * sqrt_t)
        + r * k * disc * norm.cdf(-d2)
    )

    result = pd.DataFrame(
        {
            "option_id": (
                df["option_id"]
                .astype(str)
                .to_numpy()
            ),
            "price": np.where(
                call,
                price_call,
                price_put,
            ).astype(float),
            "delta": np.where(
                call,
                delta_call,
                delta_put,
            ).astype(float),
            "gamma": gamma.astype(float),
            "vega": vega.astype(float),
            "theta": (
                np.where(
                    call,
                    theta_call_annual,
                    theta_put_annual,
                )
                / 365.0
            ).astype(float),
        }
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_dir / "results.parquet",
        index=False,
    )


if __name__ == "__main__":
    main()
""".strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "AgenthonMockModel/2.0"

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != EXPECTED_PATH:
            self.send_response(404)
            self.end_headers()
            return

        if (
            self.headers.get("Authorization")
            != EXPECTED_AUTH
        ):
            body = b'{"error":"unauthorized"}'
            self.send_response(401)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )
        raw = self.rfile.read(
            content_length
        )

        try:
            payload = json.loads(
                raw.decode("utf-8")
            )
        except Exception:
            payload = {}

        model = payload.get(
            "model",
            "mock-model",
        )

        response = {
            "id": "mock-chatcmpl-runtime-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": SOLVER_CODE,
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

        body = json.dumps(
            response
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )
    print(
        f"mock model listening on "
        f"{HOST}:{PORT}",
        flush=True,
    )
    server.serve_forever()
