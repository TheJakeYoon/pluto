"""Agents package: insertion + education LangChain agents, helpers, metrics, routes."""

import langchain_py314_shim

langchain_py314_shim.install()

from .insertion_agent import InsertionAgent
from .education_agent import EducationAgent
from .routes import router as agents_router

__all__ = ["InsertionAgent", "EducationAgent", "agents_router"]
