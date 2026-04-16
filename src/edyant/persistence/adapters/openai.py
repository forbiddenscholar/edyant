from __future__ import annotations
import os
import time
from typing import Any

from .base import AdapterError, ModelAdapter, register_adapter
from edyant.persistence.types import ModelOutput


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        super().__init__(model)
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAIAdapter requires OPENAI_API_KEY")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_sleep = retry_sleep

    def generate(self, prompt: str, **kwargs: Any) -> ModelOutput:
        try:
            from openai import OpenAI, APIError
        except ImportError as exc:
            raise AdapterError("pip install openai") from exc

        client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                text = response.choices[0].message.content or ""
                return ModelOutput(text=text, raw=response.model_dump())
            except APIError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_sleep)

        raise AdapterError(
            f"OpenAI request failed after {self._max_retries} attempts"
        ) from last_error


# Self-register at import time
register_adapter("openai", OpenAIAdapter)