"""
Shared test fixtures and helpers for Talk2BioModels unit tests.
"""

from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit_mock


class FakeApp:
    """
    Minimal stub that mimics the interface of the langgraph app used in tests.
    """

    def __init__(self, states: list[dict[str, Any]] | None = None):
        self.states = states or [{}]
        self.state_index = 0
        self.current_values: dict[str, Any] = dict(self.states[0]) if self.states else {}

    def update_state(self, _config: dict[str, Any], updates: dict[str, Any]) -> None:
        """Update the current state values."""
        self.current_values.update(updates)

    def invoke(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Return the next canned state as a response."""
        if self.state_index < len(self.states):
            self.current_values = dict(self.states[self.state_index])
            self.state_index += 1
        return {"messages": self.current_values.get("messages", [])}

    def get_state(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        """Return a namespace with current state values."""
        return SimpleNamespace(values=self.current_values)


@pytest.fixture
def fake_app_factory():
    """
    Create a FakeApp with a predefined sequence of states.
    """

    def _factory(states: list[dict[str, Any]] | None = None) -> FakeApp:
        """Instantiate FakeApp with optional canned states."""
        return FakeApp(states)

    return _factory
