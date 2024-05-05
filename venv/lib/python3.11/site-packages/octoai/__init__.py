"""Initializes the octoai module."""
from importlib.metadata import version

from . import chat, clients, errors, types, utils

__version__ = version("octoai-sdk")
__all__ = [
    "chat",
    "clients",
    "client",
    "errors",
    "server",
    "service",
    "types",
    "utils",
]
