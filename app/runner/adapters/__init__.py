"""Implementations of the common start / ask / stop protocol."""

from .base import AgentAdapter, CallableAdapter
from .subprocess import SubprocessJSONLAdapter

__all__ = ["AgentAdapter", "CallableAdapter", "SubprocessJSONLAdapter"]
