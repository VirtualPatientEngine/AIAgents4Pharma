"""
Tests for state management functionality.
"""

# import pytest
from ..state.state_talk2scholars import replace_dict

# from ..state.state_talk2scholars import Talk2Scholars
#
#
# @pytest.fixture
# def initial_state() -> Talk2Scholars:
#     """Create a base state for tests"""
#     return Talk2Scholars(
#         messages=[],
#         papers={},
#         multi_papers={},
#         is_last_step=False,
#         current_agent=None,
#         llm_model="gpt-4o-mini",
#         next="",
#     )


def test_state_replace_dict():
    """Verifies state dictionary replacement works correctly"""
    existing = {"key1": "value1", "key2": "value2"}
    new = {"key3": "value3"}
    result = replace_dict(existing, new)
    assert result == new
    assert isinstance(result, dict)
