"""Test configuration and fixtures"""

import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for tests"""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")
    yield


@pytest.fixture
def mock_response():
    """Fixture for mocked API responses"""
    return {
        "data": [
            {
                "paperId": "1234567890123456789012345678901234567890",
                "title": "Test Paper",
                "abstract": "Test abstract",
                "year": 2024,
            }
        ]
    }
