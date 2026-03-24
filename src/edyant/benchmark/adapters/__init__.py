"""Model adapter registry and built-in adapters."""

from .base import (
    AdapterError,
    ModelAdapter,
    available_adapters,
    create_adapter,
    get_adapter,
    register_adapter,
)
from .ollama import OllamaAdapter, OllamaJudgeAdapter
from .openai import OpenAIAdapter, OpenAIJudgeAdapter

__all__ = [
    "AdapterError",
    "ModelAdapter",
    "OllamaAdapter",
    "OllamaJudgeAdapter",
    "OpenAIAdapter",
    "OpenAIJudgeAdapter",
    "available_adapters",
    "create_adapter",
    "get_adapter",
    "register_adapter",
]
