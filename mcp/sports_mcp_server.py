#!/usr/bin/env python3
"""
Sports Data MCP Server
=====================
Model Context Protocol server providing access to sports data from:
- The Odds API (betting odds, scores, events)
- ESPN API (teams, schedules, players, news)

Supports natural language queries for sports information.

Tool definitions live in sports_api/tools/, grouped by the API they call
(odds_tools, espn_tools, tennis_tools, format_tools, artifact_tools,
diagnostics_tools). This file is the composition root: it loads configuration,
builds the shared handlers and cache, and registers each tool group against
them.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from mcp.server import FastMCP

# Import API handlers
from sports_api.odds_api_handler import OddsAPIHandler
from sports_api.espn_api_handler import ESPNAPIHandler
from sports_api.tennis_api_handler import TennisAPIHandler
from sports_api.cache import ResponseCache
from sports_api.tools import (
    register_odds_tools,
    register_espn_tools,
    register_tennis_tools,
    register_format_tools,
    register_artifact_tools,
    register_diagnostics_tools,
)

# Configure logging before anything else emits a message. This server speaks
# JSON-RPC over stdout, so all diagnostics must go to stderr (logging's default).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from persistent config directory (survives updates)
# Check for user-specified config directory (set via manifest.json)
config_dir_env = os.getenv("SPORTS_MCP_CONFIG_DIR")
if config_dir_env:
    config_dir = Path(os.path.expandvars(config_dir_env))
else:
    # Fall back to script directory (for development/standalone)
    config_dir = Path(__file__).parent.absolute()

# Create config directory if it doesn't exist
config_dir.mkdir(parents=True, exist_ok=True)

env_path = config_dir / ".env"
script_dir = Path(__file__).parent.absolute()
env_example_path = script_dir / ".env.example"

# If .env doesn't exist, create from template (first install only)
if not env_path.exists() and env_example_path.exists():
    import shutil
    shutil.copy(env_example_path, env_path)
    logger.warning("First-time setup: created configuration file at %s", env_path)
    logger.warning(
        "IMPORTANT: edit that file and add your ODDS_API_KEY. "
        "Get your API key from: https://the-odds-api.com"
    )

if env_path.exists():
    load_dotenv(env_path)
    logger.info("Loaded .env from persistent config: %s", env_path)
else:
    # Fall back to current directory
    load_dotenv()
    logger.info("Loaded .env from current directory or environment")

# Initialize FastMCP server
mcp = FastMCP(
    "Sports Data MCP",
    dependencies=["aiohttp", "python-dotenv"]
)

# Initialize API handlers
odds_api_key_str = os.getenv("ODDS_API_KEY")
if not odds_api_key_str:
    logger.warning("ODDS_API_KEY not found. Odds API tools will not be available.")

# Parse API keys (supports comma-separated list for round-robin)
odds_api_keys = None
if odds_api_key_str:
    keys = [key.strip() for key in odds_api_key_str.split(",") if key.strip()]
    if len(keys) == 1:
        odds_api_keys = keys[0]  # Single key as string
    else:
        odds_api_keys = keys  # Multiple keys, drained one at a time
        logger.info(f"\U0001F3B2 Easter egg activated! {len(keys)} API keys loaded")

# Load bookmaker configuration
bookmakers_filter_str = os.getenv("BOOKMAKERS_FILTER", "").strip()
bookmakers_filter = [bm.strip() for bm in bookmakers_filter_str.split(",") if bm.strip()] if bookmakers_filter_str else None

bookmakers_limit_str = os.getenv("BOOKMAKERS_LIMIT", "5").strip()
try:
    bookmakers_limit = int(bookmakers_limit_str) if bookmakers_limit_str else 5
except ValueError:
    logger.warning(f"Invalid BOOKMAKERS_LIMIT value: {bookmakers_limit_str}. Using default: 5")
    bookmakers_limit = 5

if bookmakers_filter:
    logger.info(f"Bookmaker filter active: {', '.join(bookmakers_filter)} (limit: {bookmakers_limit})")
else:
    logger.info(f"No bookmaker filter set. Showing up to {bookmakers_limit} bookmakers per game.")

# ============================================================================
# RESPONSE CACHE
# ============================================================================
# One cache shared by both handlers, so the Odds and ESPN namespaces are
# bounded together rather than each growing to its own limit. Cache keys are
# namespaced ("odds"/"espn"), so there is no risk of collision.

def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid {name} value: {raw!r}. Using default: {default}")
        return default


cache_enabled = os.getenv("CACHE_ENABLED", "true").strip().lower() not in (
    "false", "0", "no", "off"
)
cache_max_entries = _env_int("CACHE_MAX_ENTRIES", 256)

response_cache = ResponseCache(max_entries=cache_max_entries, enabled=cache_enabled)

# Per-endpoint TTL overrides. Anything unset keeps the handler default.
odds_ttls = {
    key: _env_int(env_name, default)
    for key, env_name, default in (
        ("sports", "CACHE_TTL_SPORTS", 3600),
        ("odds", "CACHE_TTL_ODDS", 60),
        ("scores", "CACHE_TTL_SCORES", 60),
        ("event_odds", "CACHE_TTL_ODDS", 60),
    )
}
espn_ttls = {
    "scoreboard": _env_int("CACHE_TTL_SCOREBOARD", 30),
    "summary": _env_int("CACHE_TTL_SCOREBOARD", 30),
}
# Tennis live TTL is the free-tier survival knob: at 30s, following one match
# is ~360 requests against a free key's 100/day; a larger value keeps a day of
# casual score-checking inside the free allowance. See .env.example.
tennis_ttls = {
    "live": _env_int("CACHE_TTL_TENNIS_LIVE", 30),
    "fixtures": _env_int("CACHE_TTL_TENNIS_FIXTURES", 300),
    "player": _env_int("CACHE_TTL_TENNIS_PLAYER", 3600),
    # /usage is the free recovery read; keep it short so a park re-check is
    # never served from a raised (up to 900s) live cache. Independent of
    # CACHE_TTL_TENNIS_LIVE on purpose.
    "usage": _env_int("CACHE_TTL_TENNIS_USAGE", 15),
}

if cache_enabled:
    logger.info(
        f"Response cache enabled (max {cache_max_entries} entries, "
        f"odds TTL {odds_ttls['odds']}s, scoreboard TTL {espn_ttls['scoreboard']}s)"
    )
else:
    logger.info("Response cache DISABLED via CACHE_ENABLED. Every call hits the API.")

odds_handler = OddsAPIHandler(
    api_key=odds_api_keys,
    bookmakers_filter=bookmakers_filter,
    bookmakers_limit=bookmakers_limit,
    cache=response_cache,
    ttls=odds_ttls,
) if odds_api_keys else None
espn_handler = ESPNAPIHandler(cache=response_cache, ttls=espn_ttls)

# Live Tennis API handler, gated on TENNIS_API_KEY. Without a key we build no
# handler and register no tools, so a user who only wants US team sports never
# sees tennis tools that could only fail for them. The handler carries its own
# per-minute token bucket and daily-quota tracking rather than routing through
# the Odds key manager, which models a different quota shape.
tennis_api_key = os.getenv("TENNIS_API_KEY", "").strip()
if not tennis_api_key:
    logger.info(
        "TENNIS_API_KEY not set. Tennis tools are unavailable; other sports "
        "tools work normally. Get a free key at "
        "https://livetennisapi.com/subscribe/free to enable them."
    )
tennis_handler = (
    TennisAPIHandler(api_key=tennis_api_key, cache=response_cache, ttls=tennis_ttls)
    if tennis_api_key
    else None
)


# ============================================================================
# DASHBOARD TOOLS (optional)
# ============================================================================
# The BetTrack dashboard tools live in the `dashboard_api` package and are
# attached to this same server when DASHBOARD_API_KEY is configured. Without
# a key we register nothing at all, so a user who only wants sports data
# never sees dashboard tools they cannot call.
#
# This must run after load_dotenv() above, since the key normally comes from
# the persistent .env rather than the process environment.

dashboard_tool_names: list[str] = []

try:
    from dashboard_api import client as dashboard_client
    from dashboard_api.tools import register_dashboard_tools

    if dashboard_client.configure():
        dashboard_tool_names = register_dashboard_tools(mcp)
        logger.info(
            f"Dashboard connected at {dashboard_client.get_api_url()} "
            f"({len(dashboard_tool_names)} tools registered)"
        )
    else:
        logger.info(
            "DASHBOARD_API_KEY not set. Dashboard tools are unavailable; "
            "sports data tools work normally. Add DASHBOARD_API_KEY to your "
            ".env to enable bet tracking and analytics."
        )
except ImportError as exc:
    # A partial install should degrade to sports-only rather than refusing
    # to start and taking the sports tools down with it.
    logger.warning(f"Dashboard tools could not be loaded: {exc}")


# ============================================================================
# SPORTS DATA TOOLS
# ============================================================================

register_odds_tools(mcp, odds_handler)
register_espn_tools(mcp, espn_handler)
register_tennis_tools(mcp, tennis_handler)
register_format_tools(mcp, espn_handler, odds_handler)
register_artifact_tools(mcp, odds_handler)
register_diagnostics_tools(
    mcp, response_cache, odds_handler, dashboard_tool_names, tennis_handler
)


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Sports Data MCP Server...")
    logger.info(f"Odds API configured: {odds_handler is not None}")
    logger.info(f"ESPN API configured: True")
    logger.info(f"Tennis API configured: {tennis_handler is not None}")
    logger.info(f"Dashboard configured: {len(dashboard_tool_names) > 0}")
    logger.info(f"Response cache: {'enabled' if cache_enabled else 'disabled'}")
    if odds_handler and len(odds_handler.api_keys) > 1:
        logger.info(f"Odds API keys loaded: {len(odds_handler.api_keys)}")

    # Run the server
    mcp.run()
