"""
Unit tests for main agent functionality.
Tests the supervisor agent's routing logic and state management.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=redefined-outer-name,too-few-public-methods
import random
from unittest.mock import Mock, patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END
from ..agents.main_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars
