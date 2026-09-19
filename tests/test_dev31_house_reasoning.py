from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.model_client import ModelClient


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size: int) -> bytes:
        return self.payload


class HouseReasoningPayloadTests(unittest.TestCase):
    def test_low_effort_thinking_is_enabled(self) -> None:
        response_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "print('ok')"
                        }
                    }
                ]
            }
        ).encode("utf-8")

        client = ModelClient(
            endpoint="https://example.test",
            model="house",
            token="test-token",
        )

        seen = {}

        def fake_urlopen(request, timeout):
            seen["payload"] = json.loads(
                request.data.decode("utf-8")
            )
            return _FakeResponse(response_body)

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client.complete(
                "write code"
            )

        kwargs = seen["payload"][
            "chat_template_kwargs"
        ]

        self.assertIs(
            kwargs["enable_thinking"],
            True,
        )
        self.assertIs(
            kwargs["low_effort"],
            True,
        )
        self.assertIs(
            kwargs["force_nonempty_content"],
            True,
        )
        self.assertEqual(
            seen["payload"]["max_tokens"],
            4000,
        )
        self.assertEqual(
            result.text,
            "print('ok')",
        )


if __name__ == "__main__":
    unittest.main()
