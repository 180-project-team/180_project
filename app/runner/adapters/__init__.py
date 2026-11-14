"""Implementations of the common start / ask / stop protocol."""

from .base import AgentAdapter, CallableAdapter
from .subprocess import SubprocessJSONLAdapter
from .madqa import MADQAAdapter

__all__ = ["AgentAdapter", "CallableAdapter", "SubprocessJSONLAdapter", "MADQAAdapter"]
