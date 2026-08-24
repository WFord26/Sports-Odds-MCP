"""MCP tool registration functions for the Sports Data server, grouped by API."""

from .odds_tools import register_odds_tools
from .espn_tools import register_espn_tools
from .tennis_tools import register_tennis_tools
from .format_tools import register_format_tools
from .artifact_tools import register_artifact_tools
from .diagnostics_tools import register_diagnostics_tools

__all__ = [
    "register_odds_tools",
    "register_espn_tools",
    "register_tennis_tools",
    "register_format_tools",
    "register_artifact_tools",
    "register_diagnostics_tools",
]
