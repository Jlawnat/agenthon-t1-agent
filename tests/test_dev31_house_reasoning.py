from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.model_client import ModelClient


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.payload


class HouseReasoningPayloadTests(unittest.TestCase):
    def test_house_thinking_is_disabled_and_output_is_capped(self) -> None:
        body = json.dumps({
            "choices": [{"message": {"content": "print('ok')"}}]
        }).encode("utf-8")
        seen = {}

        def fake_urlopen(request, timeout):
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(body)

        client = ModelClient(
            endpoint="https://example.test",
            model="house",
            token="token",
        )

        with patch(
            "agent.model_client.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client.complete("hello")

        self.assertEqual(
            seen["payload"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(seen["payload"]["max_tokens"], 4000)
        self.assertEqual(result.text, "print('ok')")

    def test_large_requested_output_is_clamped_to_house_limit(self) -> None:
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}}]
        }).encode("utf-8")
        seen = {}

        def fake_urlopen(request, timeout):
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(body)

        client = ModelClient(
            endpoint="https://example.test",
            model="house",
            token="token",
            max_output_tokens=8000,
        )

        with patch(
            "agent.model_client.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            client.complete("hello", max_output_tokens=8000)

        self.assertEqual(seen["payload"]["max_tokens"], 4000)


if __name__ == "__main__":
    unittest.main()
