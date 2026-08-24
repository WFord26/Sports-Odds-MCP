"""
Live Tennis API MCP Tools
=========================
Live scores, fixtures, and player lookups backed by `TennisAPIHandler`.

Unlike the always-on ESPN tools, these are gated on `TENNIS_API_KEY`:
`register_tennis_tools(mcp, tennis_handler)` registers nothing at all when
`tennis_handler` is None, so a user who never sets a tennis key is not shown
tools that could only fail for them.

Tool names are prefixed (`get_tennis_*`) — the server's tool namespace is flat
and already deep.

Tier note (see the handler module docstring for the arithmetic): live scores
are free-tier by entitlement but bound by the free key's 100 requests/day, so
following a match to the end needs a Basic/Pro key or a raised
CACHE_TTL_TENNIS_LIVE. Fixtures and player lookups sit comfortably in the free
tier. Rank-ordered rankings and H2H are intentionally not in v1.
"""

from typing import Optional

from ..tennis_api_handler import derive_break_point


# Tour labels are cheap to normalise in one place.
_VALID_TOURS = {"atp", "wta", "challenger", "itf"}


def _summarize_match(match: dict) -> dict:
    """Reduce a raw match object to the essentials, deriving break-point state.

    Keeps the response small (the ESPN tools do the same) and adds a derived
    `break_point` flag from the documented rule when the server and points are
    known, so a client does not have to reconstruct it.
    """
    players = match.get("players", {}) or {}
    p1 = players.get("p1", {}) or {}
    p2 = players.get("p2", {}) or {}
    score = match.get("score", {}) or {}

    summary = {
        "id": match.get("id"),
        "tour": match.get("tour"),
        "status": match.get("status"),
        "players": {
            "p1": {"id": p1.get("id"), "name": p1.get("name")},
            "p2": {"id": p2.get("id"), "name": p2.get("name")},
        },
        "score": {
            "sets": score.get("sets"),
            "games": score.get("games"),
            "points": score.get("points"),
            "server": score.get("server"),
        },
    }

    tournament = match.get("tournament")
    if tournament:
        summary["tournament"] = {
            "name": tournament.get("name"),
            "surface": tournament.get("surface"),
            "round": tournament.get("round"),
        }

    # Prefer an explicit flag from the API; otherwise derive it.
    if "break_point" in score:
        summary["score"]["break_point"] = score.get("break_point")
    else:
        derived = derive_break_point(score)
        if derived is not None:
            summary["score"]["break_point"] = derived

    return summary


def register_tennis_tools(mcp, tennis_handler) -> None:
    """Attach the Live Tennis API tools to `mcp`, if a handler was built.

    Args:
        mcp: A FastMCP server instance.
        tennis_handler: A TennisAPIHandler instance, or None when
            TENNIS_API_KEY is not configured — in which case nothing is
            registered.
    """
    if tennis_handler is None:
        return

    @mcp.tool()
    async def get_tennis_scoreboard(
        tour: Optional[str] = None,
        status: str = "live",
        limit: int = 20,
    ) -> dict:
        """
        Get tennis matches and their current score (ATP, WTA, Challenger, ITF).

        Quota note: live scores are free-tier but the free key is 100
        requests/day. At the default 30s cache TTL, following a single match to
        the end exhausts a free key within that match. For low-cadence checks,
        fixtures and players this is comfortably free; for live match-following
        use a Basic/Pro key or raise CACHE_TTL_TENNIS_LIVE.

        Args:
            tour: Optional tour filter — atp, wta, challenger, or itf.
            status: Match status — live (default), upcoming, or completed.
            limit: Maximum number of matches to return (default: 20, max: 50).

        Returns:
            Dictionary with a streamlined `matches` list. Each match carries
            players, the set/games/points score object, the server (1|2|null),
            and a `break_point` flag (from the API, else derived).

        Example:
            get_tennis_scoreboard() -> live matches across all tours
            get_tennis_scoreboard(tour="atp") -> live ATP matches
        """
        if tour and tour.lower() not in _VALID_TOURS:
            return {
                "success": False,
                "error": f"Unknown tour '{tour}'. Use one of: {', '.join(sorted(_VALID_TOURS))}.",
            }

        limit = max(1, min(limit, 50))
        result = await tennis_handler.get_live_matches(tour=tour, status=status, limit=limit)
        if not result.get("success"):
            return result

        payload = result.get("data", {})
        matches = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(matches, list):
            matches = []

        summarized = [_summarize_match(m) for m in matches[:limit]]
        return {
            "success": True,
            "matches": summarized,
            "count": len(summarized),
            "status": status,
            "cached": result.get("cached", False),
        }

    @mcp.tool()
    async def get_tennis_match_score(match_id: str) -> dict:
        """
        Get one tennis match's current score — the lowest-latency score read.

        Args:
            match_id: The match id (from get_tennis_scoreboard or get_tennis_fixtures).

        Returns:
            Dictionary with the match's score object (sets, games, points,
            server) and a derived `break_point` flag when server and points
            are known.

        Example:
            get_tennis_match_score("m_8f2c1a") -> current score for that match
        """
        result = await tennis_handler.get_match_score(match_id)
        if not result.get("success"):
            return result

        payload = result.get("data", {})
        match = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(match, dict):
            return {"success": True, "match": match, "cached": result.get("cached", False)}

        return {
            "success": True,
            "match": _summarize_match(match),
            "cached": result.get("cached", False),
        }

    @mcp.tool()
    async def get_tennis_fixtures(
        tour: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """
        Get upcoming tennis fixtures (free tier).

        Args:
            tour: Optional tour filter — atp, wta, challenger, or itf.
            date: Optional date filter in YYYY-MM-DD.
            limit: Maximum number of fixtures to return (default: 20, max: 50).

        Returns:
            Dictionary with an `fixtures` list of upcoming matches (players,
            tournament, scheduled start when provided).

        Example:
            get_tennis_fixtures(tour="wta") -> upcoming WTA fixtures
            get_tennis_fixtures(date="2026-05-31") -> fixtures on that date
        """
        if tour and tour.lower() not in _VALID_TOURS:
            return {
                "success": False,
                "error": f"Unknown tour '{tour}'. Use one of: {', '.join(sorted(_VALID_TOURS))}.",
            }

        limit = max(1, min(limit, 50))
        result = await tennis_handler.get_fixtures(tour=tour, date=date, limit=limit)
        if not result.get("success"):
            return result

        payload = result.get("data", {})
        fixtures = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(fixtures, list):
            fixtures = []

        return {
            "success": True,
            "fixtures": fixtures[:limit],
            "count": len(fixtures[:limit]),
            "cached": result.get("cached", False),
        }

    @mcp.tool()
    async def get_tennis_player(player_id: str) -> dict:
        """
        Get a tennis player's profile, including current ranking and Elo (free tier).

        Args:
            player_id: The player id (from a match's players, or a fixture).

        Returns:
            Dictionary with the player's profile under `player`.

        Example:
            get_tennis_player("p_alcaraz") -> profile, ranking, Elo rating
        """
        result = await tennis_handler.get_player(player_id)
        if not result.get("success"):
            return result

        payload = result.get("data", {})
        player = payload.get("data", payload) if isinstance(payload, dict) else payload
        return {
            "success": True,
            "player": player,
            "cached": result.get("cached", False),
        }
