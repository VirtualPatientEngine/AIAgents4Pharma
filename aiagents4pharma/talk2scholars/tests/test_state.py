"""
Tests for state management functionality.
"""

from ..state.state_talk2scholars import merge_dict, replace_dict


def test_state_replace_dict():
    """Verifies state dictionary replacement works correctly"""
    existing = {"key1": "value1", "key2": "value2"}
    new = {"key3": "value3"}
    result = replace_dict(existing, new)
    assert result == new
    assert isinstance(result, dict)


def test_state_merge_dict():
    """Verifies state dictionary merging works correctly"""
    existing = {"a": 1, "b": 2}
    new = {"b": 3, "c": 4}
    result = merge_dict(existing, new)
    # result should contain merged keys, with new values overriding existing ones
    assert result == {"a": 1, "b": 3, "c": 4}
    assert isinstance(result, dict)
    # original existing dict should be unchanged
    assert existing == {"a": 1, "b": 2}


def test_replace_dict_non_mapping():
    """Verifies replace_dict returns non-mapping values directly"""

    existing = {"key": "value"}
    # When new is not a dict, replace_dict should return new value unchanged
    new_value = "not_a_dict"
    result = replace_dict(existing, new_value)
    assert result == new_value
    # existing should remain unmodified when returning new directly
    assert existing == {"key": "value"}
