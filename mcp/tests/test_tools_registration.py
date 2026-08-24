"""
Tests for sports_api/tools/*

These exercise the register_*_tools() functions that sports_mcp_server.py
calls to build the MCP surface. Each register_*_tools() only ever calls
`mcp.tool()` as a decorator, so a minimal FakeMCP (see conftest_fakemcp.py)
stands in for the real FastMCP instance — no need for the `mcp` SDK package,
which requires Python >=3.10.

Two things are worth catching here that the handler-level tests below don't:
  1. A tool silently missing after the module split (wrong name, forgotten
     registration call).
  2. A wrapper function passing the wrong args to its handler, or mishandling
     the handler-is-None case for tools gated on ODDS_API_KEY.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sports_api.tools import (
    register_odds_tools,
    register_espn_tools,
    register_tennis_tools,
    register_format_tools,
    register_artifact_tools,
    register_diagnostics_tools,
)


class FakeMCP:
    """Stands in for FastMCP: register_*_tools() only ever calls
    `mcp.tool()` as a decorator factory, so this records the decorated
    function under its own name without needing the real `mcp` SDK package
    (which requires Python >=3.10)."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


class TestRegisterOddsTools:
    def test_registers_expected_tools(self):
        mcp = FakeMCP()
        register_odds_tools(mcp, odds_handler=MagicMock())
        assert set(mcp.tools) == {
            "get_available_sports",
            "get_odds",
            "get_scores",
            "get_event_odds",
            "search_odds",
        }

    @pytest.mark.asyncio
    async def test_delegates_to_handler(self):
        handler = MagicMock()
        handler.get_odds = AsyncMock(return_value={"success": True, "data": []})
        mcp = FakeMCP()
        register_odds_tools(mcp, odds_handler=handler)

        result = await mcp.tools["get_odds"]("basketball_nba", markets="h2h")

        handler.get_odds.assert_awaited_once_with(
            sport="basketball_nba",
            regions="us",
            markets="h2h",
            odds_format="american",
            date_format="iso",
        )
        assert result == {"success": True, "data": []}

    @pytest.mark.asyncio
    async def test_missing_handler_returns_error_not_exception(self):
        mcp = FakeMCP()
        register_odds_tools(mcp, odds_handler=None)

        result = await mcp.tools["get_odds"]("basketball_nba")
        assert "error" in result

        result = await mcp.tools["search_odds"]("Lakers")
        assert "error" in result


class TestRegisterEspnTools:
    def test_registers_expected_tools(self):
        mcp = FakeMCP()
        register_espn_tools(mcp, espn_handler=MagicMock())
        assert set(mcp.tools) == {
            "get_espn_scoreboard",
            "get_espn_standings",
            "get_espn_teams",
            "get_espn_team_details",
            "get_espn_team_schedule",
            "get_espn_news",
            "search_espn",
            "get_espn_game_summary",
        }

    @pytest.mark.asyncio
    async def test_get_espn_scoreboard_streamlines_response(self):
        handler = MagicMock()
        handler.get_scoreboard = AsyncMock(return_value={
            "success": True,
            "data": {
                "leagues": [{"name": "NBA"}],
                "events": [
                    {
                        "id": "1",
                        "name": "Lakers vs Celtics",
                        "date": "2026-01-15",
                        "competitions": [{
                            "status": {"type": {"description": "Final", "completed": True}},
                            "competitors": [
                                {"homeAway": "home", "team": {"displayName": "Lakers", "abbreviation": "LAL"}, "score": "120"},
                                {"homeAway": "away", "team": {"displayName": "Celtics", "abbreviation": "BOS"}, "score": "118"},
                            ],
                        }],
                    }
                ],
            },
        })
        mcp = FakeMCP()
        register_espn_tools(mcp, espn_handler=handler)

        result = await mcp.tools["get_espn_scoreboard"]("basketball", "nba")

        assert result["success"] is True
        assert result["total_games"] == 1
        assert result["games"][0]["home_team"]["name"] == "Lakers"
        assert result["games"][0]["away_team"]["score"] == "118"

    @pytest.mark.asyncio
    async def test_failure_passthrough(self):
        handler = MagicMock()
        handler.get_scoreboard = AsyncMock(return_value={"success": False, "error": "boom"})
        mcp = FakeMCP()
        register_espn_tools(mcp, espn_handler=handler)

        result = await mcp.tools["get_espn_scoreboard"]("basketball", "nba")
        assert result == {"success": False, "error": "boom"}


class TestRegisterTennisTools:
    def test_no_key_registers_nothing(self):
        # Gated on TENNIS_API_KEY: with no handler, nothing is registered, so a
        # user without a tennis key is never shown tools that can only fail.
        mcp = FakeMCP()
        register_tennis_tools(mcp, tennis_handler=None)
        assert mcp.tools == {}

    def test_registers_expected_tools_with_handler(self):
        mcp = FakeMCP()
        register_tennis_tools(mcp, tennis_handler=MagicMock())
        assert set(mcp.tools) == {
            "get_tennis_scoreboard",
            "get_tennis_match_score",
            "get_tennis_fixtures",
            "get_tennis_player",
        }

    @pytest.mark.asyncio
    async def test_scoreboard_summarizes_and_derives_break_point(self):
        handler = MagicMock()
        handler.get_live_matches = AsyncMock(return_value={
            "success": True,
            "cached": False,
            "data": {
                "data": [
                    {
                        "id": "m_1",
                        "tour": "atp",
                        "status": "live",
                        "players": {
                            "p1": {"id": "p_a", "name": "Player A"},
                            "p2": {"id": "p_b", "name": "Player B"},
                        },
                        "score": {
                            "sets": [1, 0],
                            "games": [[6, 4], [3, 2]],
                            "points": ["30", "40"],  # receiver (p2) 40 vs server 30
                            "server": 1,
                        },
                    }
                ]
            },
        })
        mcp = FakeMCP()
        register_tennis_tools(mcp, tennis_handler=handler)

        result = await mcp.tools["get_tennis_scoreboard"](tour="atp")

        handler.get_live_matches.assert_awaited_once_with(tour="atp", status="live", limit=20)
        assert result["success"] is True
        assert result["count"] == 1
        match = result["matches"][0]
        assert match["players"]["p1"]["name"] == "Player A"
        assert match["score"]["server"] == 1
        # Derived from the documented rule: receiver at 40 vs server at 30.
        assert match["score"]["break_point"] is True

    @pytest.mark.asyncio
    async def test_scoreboard_rejects_unknown_tour(self):
        mcp = FakeMCP()
        register_tennis_tools(mcp, tennis_handler=MagicMock())

        result = await mcp.tools["get_tennis_scoreboard"](tour="cricket")
        assert result["success"] is False
        assert "cricket" in result["error"]

    @pytest.mark.asyncio
    async def test_failure_passthrough(self):
        handler = MagicMock()
        handler.get_player = AsyncMock(return_value={"success": False, "error": "boom"})
        mcp = FakeMCP()
        register_tennis_tools(mcp, tennis_handler=handler)

        result = await mcp.tools["get_tennis_player"]("p_x")
        assert result == {"success": False, "error": "boom"}


class TestRegisterFormatTools:
    def test_registers_expected_tools(self):
        mcp = FakeMCP()
        register_format_tools(mcp, espn_handler=MagicMock(), odds_handler=MagicMock())
        assert set(mcp.tools) == {
            "get_scoreboard",
            "get_formatted_scoreboard",
            "get_matchup_cards",
            "get_detailed_scoreboard",
            "get_formatted_standings",
            "get_team_reference",
            "find_team",
            "get_comprehensive_game_info",
        }

    @pytest.mark.asyncio
    async def test_get_scoreboard_rejects_unknown_league(self):
        mcp = FakeMCP()
        register_format_tools(mcp, espn_handler=MagicMock(), odds_handler=None)

        result = await mcp.tools["get_scoreboard"]("cricket")
        assert result["success"] is False
        assert "cricket" in result["error"]

    @pytest.mark.asyncio
    async def test_get_scoreboard_resolves_league_and_formats(self):
        handler = MagicMock()
        handler.get_scoreboard = AsyncMock(return_value={
            "success": True,
            "data": {"events": [{"home_team": "Lakers", "away_team": "Celtics"}]},
        })
        mcp = FakeMCP()
        register_format_tools(mcp, espn_handler=handler, odds_handler=None)

        result = await mcp.tools["get_scoreboard"]("nba")

        handler.get_scoreboard.assert_awaited_once_with(sport="basketball", league="nba", date=None, limit=15)
        assert result["success"] is True
        assert result["league"] == "NBA"
        assert result["game_count"] == 1

    def test_find_team_not_found(self):
        mcp = FakeMCP()
        register_format_tools(mcp, espn_handler=MagicMock(), odds_handler=None)

        result = mcp.tools["find_team"]("Nonexistent Team", "nba")
        assert result["success"] is False

    def test_find_team_found(self):
        mcp = FakeMCP()
        register_format_tools(mcp, espn_handler=MagicMock(), odds_handler=None)

        result = mcp.tools["find_team"]("Los Angeles Lakers", "nba")
        assert result["success"] is True
        assert result["team"]["abbr"] == "LAL"


class TestRegisterArtifactTools:
    def test_registers_expected_tools(self):
        mcp = FakeMCP()
        register_artifact_tools(mcp, odds_handler=MagicMock())
        assert set(mcp.tools) == {"get_odds_card_artifact", "get_visual_scoreboard"}

    @pytest.mark.asyncio
    async def test_missing_handler_returns_error(self):
        mcp = FakeMCP()
        register_artifact_tools(mcp, odds_handler=None)

        result = await mcp.tools["get_odds_card_artifact"]("Lakers")
        assert "error" in result

        result = await mcp.tools["get_visual_scoreboard"]()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_matching_games(self):
        handler = MagicMock()
        handler.search_odds = AsyncMock(return_value={"matching_games": []})
        mcp = FakeMCP()
        register_artifact_tools(mcp, odds_handler=handler)

        result = await mcp.tools["get_odds_card_artifact"]("Nuggets")
        assert result["success"] is False
        assert "Nuggets" in result["error"]


class TestRegisterDiagnosticsTools:
    def test_registers_expected_tools(self):
        mcp = FakeMCP()
        register_diagnostics_tools(mcp, response_cache=MagicMock(), odds_handler=None, dashboard_tool_names=[])
        assert set(mcp.tools) == {"get_api_status", "clear_cache"}

    def test_get_api_status_without_odds_handler(self):
        cache = MagicMock()
        cache.stats.return_value = {"hits": 0}
        mcp = FakeMCP()
        register_diagnostics_tools(mcp, response_cache=cache, odds_handler=None, dashboard_tool_names=["create_bet"])

        status = mcp.tools["get_api_status"]()
        assert status["odds_api"]["configured"] is False
        assert status["dashboard"]["tools_registered"] == 1
        # No tennis handler passed -> reported as not configured.
        assert status["tennis_api"]["configured"] is False

    def test_get_api_status_folds_in_tennis_quota(self):
        cache = MagicMock()
        cache.stats.return_value = {"hits": 0}
        tennis = MagicMock()
        tennis.get_quota_status.return_value = {
            "configured": True,
            "key": "twjp...key",
            "daily_exhausted": False,
        }
        mcp = FakeMCP()
        register_diagnostics_tools(
            mcp,
            response_cache=cache,
            odds_handler=None,
            dashboard_tool_names=[],
            tennis_handler=tennis,
        )

        status = mcp.tools["get_api_status"]()
        # Folded from tracked state, so it costs no upstream call.
        tennis.get_quota_status.assert_called_once_with()
        assert status["tennis_api"]["configured"] is True
        assert status["tennis_api"]["daily_exhausted"] is False

    def test_clear_cache(self):
        cache = MagicMock()
        cache.invalidate.return_value = 7
        mcp = FakeMCP()
        register_diagnostics_tools(mcp, response_cache=cache, odds_handler=None, dashboard_tool_names=[])

        result = mcp.tools["clear_cache"]()
        assert result == {
            "success": True,
            "entries_cleared": 7,
            "note": "The next call for each query will fetch live data.",
        }
