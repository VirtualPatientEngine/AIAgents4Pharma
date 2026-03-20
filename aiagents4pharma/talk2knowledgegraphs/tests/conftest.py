"""
Shared fixtures and helpers for Talk2KnowledgeGraphs unit tests.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.unit_mock


def tool_call_args(tool, args: dict[str, Any], tool_call_id: str = "tool_call_id") -> dict:
    """Create a ToolCall-like dict for BaseTool.invoke."""
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


@pytest.fixture(name="tool_call")
def tool_call_fixture():
    """Provide a ToolCall factory for tool invocation inputs."""
    return tool_call_args
