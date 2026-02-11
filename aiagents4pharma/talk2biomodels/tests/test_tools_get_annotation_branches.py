"""
Extra branch coverage for get_annotation helpers.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from ..tools.get_annotation import extract_relevant_species_names

pytestmark = pytest.mark.unit_mock


class FakeCtx:
    """Hydra initialize context manager stub."""

    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def test_extract_relevant_species_names_success(monkeypatch):
    """Cover extract_relevant_species_names success path."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.hydra.initialize",
        lambda **kwargs: FakeCtx(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.hydra.compose",
        lambda **kwargs: SimpleNamespace(
            tools=SimpleNamespace(get_annotation=SimpleNamespace(prompt="prompt"))
        ),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.model_info.get_species",
        lambda model=None: pd.DataFrame(index=["S1", "S2"]),
    )
    llm = SimpleNamespace()
    llm.with_structured_output = lambda model: SimpleNamespace(
        invoke=lambda question: SimpleNamespace(relevant_species=["S1"])
    )
    species = extract_relevant_species_names(
        SimpleNamespace(copasi_model="copasi"),
        SimpleNamespace(user_question="uq"),
        {"llm_model": llm},
    )
    assert species == ["S1"]
