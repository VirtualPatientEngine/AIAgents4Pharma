"""Tests for Talk2KnowledgeGraphs state helpers."""

from __future__ import annotations

from ..states.state_talk2knowledgegraphs import add_data


def test_add_data_merges_by_name():
    """Ensure reducer merges entries by name and appends new ones."""
    left = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]
    right = [{"name": "b", "value": 3}, {"name": "c", "value": 4}]

    merged = add_data(left, right)

    assert merged[0]["value"] == 1
    assert merged[1]["value"] == 3
    assert merged[2]["name"] == "c"
