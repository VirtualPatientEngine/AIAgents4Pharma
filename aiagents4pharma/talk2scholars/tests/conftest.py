"""
Shared fixtures for Talk2Scholars test suite.
"""

import pytest
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ..state.state_talk2scholars import Talk2Scholars

@pytest.fixture(autouse=True)
def hydra_setup():
    """Setup and cleanup Hydra for tests."""
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        yield

@pytest.fixture
def mock_cfg() -> DictConfig:
    """Create a mock configuration for testing."""
    config = {
        "agents": {
            "talk2scholars": {
                "main_agent": {
                    "state_modifier": "Test prompt for main agent",
                    "temperature": 0,
                },
                "s2_agent": {
                    "temperature": 0,
                    "s2_agent": "Test prompt for s2 agent",
                },
            }
        },
        "tools": {
            "search": {
                "api_endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
                "default_limit": 2,
                "api_fields": ["paperId", "title", "abstract", "year", "authors"],
            }
        },
    }
    return OmegaConf.create(config)

@pytest.fixture
def initial_state() -> Talk2Scholars:
    """Create a base state for tests"""
    return Talk2Scholars(
        messages=[],
        papers={},
        multi_papers={},
        is_last_step=False,
        current_agent=None,
        llm_model="gpt-4o-mini",
        next="",
    )

# Fixed test data for deterministic results
MOCK_SEARCH_RESPONSE = {
    "data": [
        {
            "paperId": "123",
            "title": "Machine Learning Basics",
            "abstract": "An introduction to ML",
            "year": 2023,
            "citationCount": 100,
            "url": "https://example.com/paper1",
            "authors": [{"name": "Test Author"}],
        }
    ]
}

MOCK_STATE_PAPER = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "URL": "https://example.com/paper1",
    }
}
