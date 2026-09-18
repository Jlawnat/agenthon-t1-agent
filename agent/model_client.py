from __future__ import annotations

import json
import math
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 4000
MAX_ERROR_BODY_BYTES = 4096
MAX_ENDPOINT_CHARS = 2048
MAX_MODEL_NAME_CHARS = 256


@dataclass
class ModelResponse:
    text: str
    raw: dict[str, Any]


class ModelClient:
    """
    Minimal OpenAI-compatible client for the organizer-provided
    model endpoint.

    The client intentionally:
    - uses only stdlib HTTP support,
    - sends no API keys or arbitrary credentials,
    - sends no tool definitions,
    - accepts only HTTP(S) endpoints,
    - bounds request/response sizes and timeouts.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        token: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.endpoint = (
            endpoint
            if endpoint is not None
            else os.getenv("MODEL_ENDPOINT")
        )

        self.model = (
            model
            if model is not None
            else os.getenv("MODEL_NAME")
        )

        self.token = (
            token
            if token is not None
            else os.getenv("MODEL_TOKEN")
        )

        self.timeout_seconds = self._positive_number(
            timeout_seconds,
            name="timeout_seconds",
        )

        self.max_request_bytes = self._positive_integer(
            max_request_bytes,
            name="max_request_bytes",
        )

        self.max_response_bytes = self._positive_integer(
            max_response_bytes,
            name="max_response_bytes",
        )

        self.max_output_tokens = self._positive_integer(
            max_output_tokens,
            name="max_output_tokens",
        )

        if self.endpoint is not None:
            self.endpoint = self._validate_endpoint(
                self.endpoint
            )

        if self.model is not None:
            self.model = self._validate_model_name(
                self.model
            )

    def is_available(self) -> bool:
        # Preserve the previous availability meaning: an endpoint
        # exists. Production CLI validation separately requires
        # MODEL_NAME as well.
        return bool(self.endpoint)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> ModelResponse:
        if not self.endpoint:
            raise RuntimeError(
                "MODEL_ENDPOINT is not configured."
            )

        if not self.model:
            raise RuntimeError(
                "MODEL_NAME is not configured."
            )

        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a string."
            )

        if not self.token:
            raise RuntimeError(
                "MODEL_TOKEN is not configured."
            )

        temperature_value = self._temperature(
            temperature
        )

        effective_timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else min(
                self.timeout_seconds,
                self._positive_number(
                    timeout_seconds,
                    name="timeout_seconds",
                ),
            )
        )

        url = self._chat_completions_url(
            self.endpoint
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature_value,
            "max_tokens": self.max_output_tokens,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        if len(body) > self.max_request_bytes:
            raise RuntimeError(
                "Model request exceeds the configured "
                "maximum request size."
            )

        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=effective_timeout,
            ) as response:
                response_body = response.read(
                    self.max_response_bytes + 1
                )

        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read(
                    MAX_ERROR_BODY_BYTES + 1
                )
            except Exception:
                error_body = b""

            error_text = error_body[
                :MAX_ERROR_BODY_BYTES
            ].decode(
                "utf-8",
                errors="replace",
            )

            # Keep the body bounded and do not include endpoint
            # URLs, proxy details, or environment values.
            raise RuntimeError(
                f"Model endpoint HTTP {exc.code}: "
                f"{error_text}"
            ) from exc

        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise RuntimeError(
                "Model endpoint connection failed."
            ) from exc

        if (
            len(response_body)
            > self.max_response_bytes
        ):
            raise RuntimeError(
                "Model response exceeds the configured "
                "maximum response size."
            )

        try:
            decoded = response_body.decode(
                "utf-8"
            )
            raw = json.loads(decoded)

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "Model endpoint returned invalid JSON."
            ) from exc

        if not isinstance(raw, dict):
            raise RuntimeError(
                "Unexpected model response format."
            )

        text = self._extract_text(raw)

        return ModelResponse(
            text=text,
            raw=raw,
        )

    @staticmethod
    def _positive_number(
        value: float,
        *,
        name: str,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise ValueError(
                f"{name} must be a positive number."
            )

        number = float(value)

        if (
            not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(
                f"{name} must be a positive finite number."
            )

        return number

    @staticmethod
    def _positive_integer(
        value: int,
        *,
        name: str,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

        return value

    @staticmethod
    def _temperature(
        value: float,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise ValueError(
                "temperature must be numeric."
            )

        temperature = float(value)

        if (
            not math.isfinite(temperature)
            or temperature < 0.0
            or temperature > 2.0
        ):
            raise ValueError(
                "temperature must be within [0, 2]."
            )

        return temperature

    @staticmethod
    def _validate_endpoint(
        endpoint: str,
    ) -> str:
        if not isinstance(endpoint, str):
            raise ValueError(
                "MODEL_ENDPOINT must be a string."
            )

        endpoint = endpoint.strip()

        if not endpoint:
            return ""

        if len(endpoint) > MAX_ENDPOINT_CHARS:
            raise ValueError(
                "MODEL_ENDPOINT is too long."
            )

        if any(
            character in endpoint
            for character in ("\r", "\n", "\x00")
        ):
            raise ValueError(
                "MODEL_ENDPOINT contains invalid characters."
            )

        parsed = urllib.parse.urlsplit(
            endpoint
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "MODEL_ENDPOINT must use http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "MODEL_ENDPOINT must include a host."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "MODEL_ENDPOINT must not embed credentials."
            )

        if parsed.query or parsed.fragment:
            raise ValueError(
                "MODEL_ENDPOINT must not include a query "
                "string or fragment."
            )

        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(
                "MODEL_ENDPOINT contains an invalid port."
            ) from exc

        return endpoint.rstrip("/")

    @staticmethod
    def _validate_model_name(
        model: str,
    ) -> str:
        if not isinstance(model, str):
            raise ValueError(
                "MODEL_NAME must be a string."
            )

        model = model.strip()

        if not model:
            return ""

        if len(model) > MAX_MODEL_NAME_CHARS:
            raise ValueError(
                "MODEL_NAME is too long."
            )

        if any(
            character in model
            for character in ("\r", "\n", "\x00")
        ):
            raise ValueError(
                "MODEL_NAME contains invalid characters."
            )

        return model

    @staticmethod
    def _chat_completions_url(
        endpoint: str,
    ) -> str:
        endpoint = endpoint.rstrip("/")

        if endpoint.endswith(
            "/chat/completions"
        ):
            return endpoint

        if endpoint.endswith("/v1"):
            return (
                endpoint
                + "/chat/completions"
            )

        return (
            endpoint
            + "/v1/chat/completions"
        )

    @staticmethod
    def _extract_text(
        raw: dict[str, Any],
    ) -> str:
        try:
            text = raw[
                "choices"
            ][0]["message"]["content"]

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Unexpected model response format."
            ) from exc

        if not isinstance(text, str):
            raise RuntimeError(
                "Unexpected model response format."
            )

        return text