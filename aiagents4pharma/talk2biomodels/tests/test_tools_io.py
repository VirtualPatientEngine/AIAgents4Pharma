"""
Unit-mock coverage for IO-oriented tools (save_model, search_models).
"""

import pytest

from ..tools.save_model import SaveModelTool
from ..tools.search_models import SearchModelsTool

pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def test_save_model_tool_paths(tmp_path):
    """Exercise save_model success and failure branches."""
    tool = SaveModelTool()
    # Success branch
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"model_as_string": ["data"]},
                "path_to_folder": str(tmp_path),
                "output_filename": "file.xml",
            },
        )
    )
    assert (tmp_path / "file.xml").exists()
    assert "successfully" in cmd.update["messages"][0].content

    # Failure branch
    bad = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"model_as_string": ["data"]},
                "path_to_folder": "/nonexistent/path",
                "output_filename": "file.xml",
            },
        )
    )
    assert bad.update["messages"][0].content.startswith("Error: Path")


def test_search_models_tool(monkeypatch):
    """Exercise search_models tool path."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.search_models.biomodels.search_for_model",
        lambda query, num_results: [{"name": "Model1", "id": "1"}],
    )
    tool = SearchModelsTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {"tool_call_id": "tc", "query": "abc", "num_query": 1},
        )
    )
    assert "Found 1 manually curated models" in cmd.update["messages"][0].content
