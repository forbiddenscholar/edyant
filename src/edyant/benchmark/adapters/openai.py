"""OpenAI adapter for cloud model execution."""

from __future__ import annotations

import os
import time
from typing import Any

from .base import AdapterError, ModelAdapter, register_adapter
from ..types import ModelOutput


class OpenAIAdapter(ModelAdapter):
    """Adapter for the OpenAI Chat Completions API."""

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
            raise ValueError(
                "OpenAIAdapter requires an API key via the 'api_key' argument or the OPENAI_API_KEY environment variable"
            )
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_sleep = retry_sleep

    def generate(self, prompt: str, **kwargs: Any) -> ModelOutput:
        """Send a prompt to OpenAI and return the response."""
        try:
            from openai import OpenAI, APIError, APITimeoutError, RateLimitError
        except ImportError as exc:
            raise AdapterError(
                "openai package is required: pip install openai"
            ) from exc

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
                raw = response.model_dump()
                return ModelOutput(text=text, raw=raw)
            except (APIError, APITimeoutError, RateLimitError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_sleep)

        raise AdapterError(
            f"OpenAI request failed after {self._max_retries} attempts"
        ) from last_error


class OpenAIJudgeAdapter(OpenAIAdapter):
    """Adapter for the judge model using OPENAI_JUDGE_MODEL and OPENAI_JUDGE_API_KEY."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        model = model or os.getenv("OPENAI_JUDGE_MODEL")
        if not model:
            raise ValueError(
                "OpenAIJudgeAdapter requires a model via the 'model' argument or the OPENAI_JUDGE_MODEL environment variable"
            )

        api_key = api_key or os.getenv("OPENAI_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAIJudgeAdapter requires an API key via the 'api_key' argument, OPENAI_JUDGE_API_KEY, or OPENAI_API_KEY environment variable"
            )

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )


register_adapter("openai", OpenAIAdapter)
register_adapter("openai_judge", OpenAIJudgeAdapter)
