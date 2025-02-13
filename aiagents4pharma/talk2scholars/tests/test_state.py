"""
Tests for state management functionality.
"""

from ..state.state_talk2scholars import replace_dict


def test_state_replace_dict():
    """Verifies state dictionary replacement works correctly"""
    existing = {"key1": "value1", "key2": "value2"}
    new = {"key3": "value3"}
    result = replace_dict(existing, new)
    assert result == new
    assert isinstance(result, dict)
