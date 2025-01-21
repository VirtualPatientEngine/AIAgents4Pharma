# Expose main agent and sub-agents at package level
from agents.main_agent import get_app
from agents.s2_agent import s2_agent

__all__ = ["get_app", "s2_agent"]
