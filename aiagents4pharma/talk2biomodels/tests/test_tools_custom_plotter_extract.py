"""
Exercise extract_relevant_species success path to cover hydra/LLM wiring.
"""

from types import SimpleNamespace

import pytest

from ..tools.custom_plotter import extract_relevant_species

pytestmark = pytest.mark.unit_mock


class FakeCtx:
    """Hydra initialize context manager stub."""

    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def test_extract_relevant_species(monkeypatch):
    """Cover extract_relevant_species and prompt wiring."""
    # Patch hydra to return a dummy prompt
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.hydra.initialize",
        lambda **kwargs: FakeCtx(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.hydra.compose",
        lambda **kwargs: SimpleNamespace(
            tools=SimpleNamespace(
                custom_plotter=SimpleNamespace(system_prompt_custom_header="prompt")
            )
        ),
    )

    class DummyPrompt:
        """Minimal prompt with chain operator."""

        def __or__(self, other):
            return SimpleNamespace(invoke=lambda _input: SimpleNamespace(relevant_species=["A"]))

        def marker(self):
            """No-op method to satisfy lint for public methods."""
            return None

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.ChatPromptTemplate.from_messages",
        lambda msgs: DummyPrompt(),
    )
    assert DummyPrompt().marker() is None
    llm = SimpleNamespace()
    llm.with_structured_output = lambda model: SimpleNamespace()
    state = {"llm_model": llm}
    result = extract_relevant_species("question", ["A", "B"], state)
    assert result.relevant_species == ["A"]
