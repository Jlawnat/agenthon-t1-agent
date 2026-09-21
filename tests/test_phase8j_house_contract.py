from __future__ import annotations

import json
from unittest.mock import patch

from agent.model_client import ModelClient


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self, size: int = -1) -> bytes:
        return self.body


def test_every_house_request_disables_thinking_and_never_exceeds_4k() -> None:
    body = json.dumps({
        "choices": [{"message": {"content": "{}"}}]
    }).encode("utf-8")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(body)

    client = ModelClient(
        endpoint="https://house.test",
        model="house",
        token="token",
        max_output_tokens=99999,
    )

    with patch(
        "agent.model_client.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        client.complete("prompt", max_output_tokens=99999)

    assert seen["payload"]["max_tokens"] == 4000
    assert seen["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_house_url_uses_required_v1_chat_completions_route() -> None:
    assert ModelClient._chat_completions_url("https://house.test") == (
        "https://house.test/v1/chat/completions"
    )
    assert ModelClient._chat_completions_url("https://house.test/v1") == (
        "https://house.test/v1/chat/completions"
    )
