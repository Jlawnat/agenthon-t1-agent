from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from agent.model_client import ModelClient


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
    ) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def read(
        self,
        size: int = -1,
    ) -> bytes:
        if size < 0:
            return self.body

        return self.body[:size]


class ModelClientSecurityTests(
    unittest.TestCase
):
    def test_rejects_non_http_endpoint(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            ModelClient(
                endpoint="file:///tmp/model",
                model="m",
            )

    def test_rejects_embedded_credentials(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            ModelClient(
                endpoint=(
                    "https://user:secret@"
                    "example.test/v1"
                ),
                model="m",
            )

    def test_rejects_endpoint_query_and_fragment(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            ModelClient(
                endpoint=(
                    "https://example.test/v1"
                    "?token=secret"
                ),
                model="m",
            )

        with self.assertRaises(
            ValueError
        ):
            ModelClient(
                endpoint=(
                    "https://example.test/v1"
                    "#fragment"
                ),
                model="m",
            )

    def test_model_name_has_no_silent_default(
        self,
    ) -> None:
        client = ModelClient(
            endpoint="https://example.test/v1",
            model="",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "MODEL_NAME",
        ):
            client.complete(
                "hello"
            )

    def test_timeout_override_is_capped_by_client(
        self,
    ) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok"
                        }
                    }
                ]
            }
        ).encode("utf-8")

        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
            timeout_seconds=10.0,
        )

        seen = {}

        def fake_urlopen(
            request,
            timeout,
        ):
            seen["timeout"] = timeout
            return _FakeResponse(body)

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            response = client.complete(
                "hello",
                timeout_seconds=30.0,
            )

        self.assertEqual(
            response.text,
            "ok",
        )
        self.assertEqual(
            seen["timeout"],
            10.0,
        )

    def test_smaller_deadline_timeout_wins(
        self,
    ) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok"
                        }
                    }
                ]
            }
        ).encode("utf-8")

        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
            timeout_seconds=120.0,
        )

        seen = {}

        def fake_urlopen(
            request,
            timeout,
        ):
            seen["timeout"] = timeout
            return _FakeResponse(body)

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            client.complete(
                "hello",
                timeout_seconds=7.5,
            )

        self.assertEqual(
            seen["timeout"],
            7.5,
        )

    def test_request_contains_authorization_header(
        self,
    ) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok"
                        }
                    }
                ]
            }
        ).encode("utf-8")

        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
        )

        seen = {}

        def fake_urlopen(
            request,
            timeout,
        ):
            seen["headers"] = {
                key.lower(): value
                for key, value
                in request.header_items()
            }
            return _FakeResponse(body)

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            client.complete(
                "hello"
            )

        self.assertEqual(
            seen["headers"]["authorization"],
            "Bearer test-token",
        )

    def test_missing_model_token_fails_closed(
        self,
    ) -> None:
        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "MODEL_TOKEN is not configured",
        ):
            client.complete(
                "hello"
            )
    def test_request_caps_output_tokens(
        self,
    ) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok"
                        }
                    }
                ]
            }
        ).encode("utf-8")

        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
        )

        seen = {}

        def fake_urlopen(
            request,
            timeout,
        ):
            seen["payload"] = json.loads(
                request.data.decode(
                    "utf-8"
                )
            )

            return _FakeResponse(body)

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            client.complete(
                "hello"
            )

        self.assertEqual(
            seen["payload"]["max_tokens"],
            16000,
        )

    def test_response_size_is_bounded(
        self,
    ) -> None:
        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
            max_response_bytes=20,
        )

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(
                b"x" * 21
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "maximum response size",
            ):
                client.complete(
                    "hello"
                )

    def test_invalid_json_fails_closed(
        self,
    ) -> None:
        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
        )

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(
                b"not-json"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "invalid JSON",
            ):
                client.complete(
                    "hello"
                )

    def test_http_error_body_is_bounded(
        self,
    ) -> None:
        client = ModelClient(
            endpoint="https://example.test/v1",
            model="m",
            token="test-token",
        )

        error = urllib.error.HTTPError(
            url="https://example.test/v1",
            code=500,
            msg="boom",
            hdrs=None,
            fp=io.BytesIO(
                b"A" * 10000
            ),
        )

        with patch(
            "urllib.request.urlopen",
            side_effect=error,
        ):
            with self.assertRaises(
                RuntimeError
            ) as captured:
                client.complete(
                    "hello"
                )

        message = str(
            captured.exception
        )

        self.assertIn(
            "HTTP 500",
            message,
        )
        self.assertLess(
            len(message),
            5000,
        )

    def test_chat_completion_path_is_stable(
        self,
    ) -> None:
        self.assertEqual(
            ModelClient._chat_completions_url(
                "https://example.test/v1"
            ),
            (
                "https://example.test/v1/"
                "chat/completions"
            ),
        )

        self.assertEqual(
            ModelClient._chat_completions_url(
                (
                    "https://example.test/"
                    "v1/chat/completions"
                )
            ),
            (
                "https://example.test/"
                "v1/chat/completions"
            ),
        )


if __name__ == "__main__":
    unittest.main()
