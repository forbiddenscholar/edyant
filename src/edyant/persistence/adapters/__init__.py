"""Adapters for the persistence layer."""

from .base import (
    AdapterError,
    ModelAdapter,
    available_adapters,
    create_adapter,
    get_adapter,
    lazy_register,
    register_adapter,
)
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter

__all__ = [
    "AdapterError",
    "ModelAdapter",
    "available_adapters",
    "create_adapter",
    "get_adapter",
    "lazy_register",
    "register_adapter",
    "OllamaAdapter",
    "OpenAIAdapter",
]
