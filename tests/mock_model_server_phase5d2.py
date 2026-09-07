from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

HOST = "0.0.0.0"
PORT = 8000


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

        response = {
            "id": "mock-chatcmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "MOCK_MODEL_OK",
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
