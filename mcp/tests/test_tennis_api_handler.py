"""
Tests for sports_api/tennis_api_handler.py

Mocks the underlying aiohttp session with aioresponses so no real Live Tennis
API requests are made. The response samples are real Live Tennis API shapes:
the {"data": ...} envelope, players.p1/p2, and the set/games/points/server
score object.

The two pieces of logic a reviewer cannot check by eye are covered explicitly:
the per-minute token bucket, and the daily-quota-exhausted short-circuit.
"""

import re
from datetime import datetime, timezone

import pytest
from aioresponses import aioresponses

from sports_api.cache import ResponseCache
from sports_api.tennis_api_handler import (
    TennisAPIHandler,
    TokenBucket,
    derive_break_point,
    _next_utc_midnight,
    _classify_rate_limit,
    _parse_retry_after,
)

TENNIS_URL_RE = re.compile(r"^https://api\.livetennisapi\.com/.*")

# A real-shape live match envelope: the API returns {"data": [...]} and each
# match carries players.p1/p2 and a score object of sets/games/points/server.
LIVE_MATCHES_ENVELOPE = {
    "data": [
        {
            "id": "m_8f2c1a",
            "tour": "atp",
            "status": "live",
            "tournament": {"name": "Roland Garros", "surface": "clay", "round": "QF"},
            "players": {
                "p1": {"id": "p_alcaraz", "name": "Carlos Alcaraz", "country": "ESP"},
                "p2": {"id": "p_sinner", "name": "Jannik Sinner", "country": "ITA"},
            },
            "score": {
                "sets": [1, 1],
                "games": [[6, 4], [3, 6], [2, 1]],
                "points": ["30", "40"],
                "server": 1,
            },
        }
    ]
}


@pytest.fixture
def handler():
    # A disabled cache forces every call through _fetch, which is what the
    # rate-limit and quota tests need to observe.
    return TennisAPIHandler(api_key="twjp_test_key", cache=ResponseCache(enabled=False))


class TestGetLiveMatches:
    @pytest.mark.asyncio
    async def test_success(self, handler):
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=LIVE_MATCHES_ENVELOPE, status=200)
            result = await handler.get_live_matches(tour="atp")

        assert result["success"] is True
        assert result["data"]["data"][0]["id"] == "m_8f2c1a"
        await handler.close()

    @pytest.mark.asyncio
    async def test_non_200_returns_failure(self, handler):
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, status=500, body="server error")
            result = await handler.get_live_matches()

        assert result["success"] is False
        assert "500" in result["error"]
        await handler.close()

    @pytest.mark.asyncio
    async def test_result_is_cached_on_second_call(self):
        handler = TennisAPIHandler(api_key="twjp_test_key")  # cache enabled
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=LIVE_MATCHES_ENVELOPE, status=200)
            first = await handler.get_live_matches(tour="atp")
            second = await handler.get_live_matches(tour="atp")

        assert first.get("cached") is False
        assert second.get("cached") is True
        await handler.close()

    @pytest.mark.asyncio
    async def test_auth_error_reports_rejected_key(self, handler):
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, status=401, body="unauthenticated")
            result = await handler.get_live_matches()

        assert result["success"] is False
        assert "rejected the key" in result["error"]
        await handler.close()

    @pytest.mark.asyncio
    async def test_request_exception_returns_failure(self, handler):
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, exception=ConnectionError("network down"))
            result = await handler.get_live_matches()

        assert result["success"] is False
        assert "network down" in result["error"]
        await handler.close()


class TestOtherEndpoints:
    @pytest.mark.asyncio
    async def test_get_match_score(self, handler):
        score_envelope = {"data": {"id": "m_8f2c1a", "score": {"server": 2, "points": ["0", "0"]}}}
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=score_envelope, status=200)
            result = await handler.get_match_score("m_8f2c1a")

        assert result["success"] is True
        assert result["data"]["data"]["id"] == "m_8f2c1a"
        await handler.close()

    @pytest.mark.asyncio
    async def test_get_fixtures(self, handler):
        fixtures_envelope = {
            "data": [
                {
                    "id": "m_upcoming1",
                    "tour": "wta",
                    "status": "upcoming",
                    "players": {"p1": {"name": "A"}, "p2": {"name": "B"}},
                    "start_time": "2026-05-31T09:00:00Z",
                }
            ]
        }
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=fixtures_envelope, status=200)
            result = await handler.get_fixtures(tour="wta", date="2026-05-31")

        assert result["success"] is True
        assert result["data"]["data"][0]["status"] == "upcoming"
        await handler.close()

    @pytest.mark.asyncio
    async def test_get_player(self, handler):
        player_envelope = {
            "data": {
                "id": "p_alcaraz",
                "name": "Carlos Alcaraz",
                "country": "ESP",
                "ranking": 2,
                "elo": 2185,
            }
        }
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=player_envelope, status=200)
            result = await handler.get_player("p_alcaraz")

        assert result["success"] is True
        assert result["data"]["data"]["ranking"] == 2
        await handler.close()


class TestTokenBucket:
    def test_starts_full_and_drains(self):
        bucket = TokenBucket(rate_per_minute=3, time_func=lambda: 1000.0)
        assert bucket.acquire() is True
        assert bucket.acquire() is True
        assert bucket.acquire() is True
        # Clock frozen, so no refill: the fourth request is denied.
        assert bucket.acquire() is False

    def test_refills_over_time(self):
        now = {"t": 1000.0}
        bucket = TokenBucket(rate_per_minute=60, capacity=1, time_func=lambda: now["t"])
        assert bucket.acquire() is True
        assert bucket.acquire() is False
        # 60/min == 1/sec, so one second restores exactly one token.
        now["t"] += 1.0
        assert bucket.acquire() is True

    def test_retry_after_is_positive_when_empty(self):
        bucket = TokenBucket(rate_per_minute=60, capacity=1, time_func=lambda: 1000.0)
        assert bucket.acquire() is True
        assert bucket.retry_after() > 0


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_minute_limit_short_circuits_without_upstream(self):
        # One token, frozen clock: the first request spends it, the second is
        # denied locally. Only one upstream response is registered, so a second
        # upstream call would raise — proving the limiter never made it.
        bucket = TokenBucket(rate_per_minute=1, capacity=1, time_func=lambda: 5000.0)
        handler = TennisAPIHandler(
            api_key="twjp_test_key",
            cache=ResponseCache(enabled=False),
            rate_limiter=bucket,
        )
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=LIVE_MATCHES_ENVELOPE, status=200)
            first = await handler.get_live_matches()
            second = await handler.get_live_matches()

        assert first["success"] is True
        assert second["success"] is False
        assert second["rate_limited"] is True
        assert second["retry_after_seconds"] > 0
        await handler.close()


class TestDailyQuota:
    @pytest.mark.asyncio
    async def test_429_parks_key_until_reset_and_next_call_short_circuits(self, handler):
        # First call gets a 429 (daily limit). Only one 429 is registered, so
        # if the second call reached the network aioresponses would raise —
        # the daily-exhausted short-circuit is what keeps it local.
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, status=429, body="daily limit exceeded")
            first = await handler.get_live_matches()
            second = await handler.get_live_matches()

        assert first["success"] is False
        assert first["quota"]["daily_exhausted"] is True
        assert second["success"] is False
        assert "Daily request quota exhausted" in second["error"]
        assert handler._daily_exhausted_until is not None
        assert handler._daily_exhausted_until > datetime.now(timezone.utc)
        await handler.close()

    @pytest.mark.asyncio
    async def test_quota_clears_after_reset(self, handler):
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, status=429, body="daily limit exceeded")
            await handler.get_live_matches()

        # Rewind the reset into the past; the next check should clear it.
        handler._daily_exhausted_until = datetime(2000, 1, 1, tzinfo=timezone.utc)
        assert handler._daily_is_exhausted() is False
        assert handler._daily_exhausted_until is None
        await handler.close()

    @pytest.mark.asyncio
    async def test_usage_endpoint_populates_quota_status(self, handler):
        usage_envelope = {
            "data": {
                "plan": "free",
                "requests_remaining": 37,
                "daily_limit": 100,
                "resets_at": "2026-05-31T00:00:00Z",
            }
        }
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=usage_envelope, status=200)
            await handler.get_usage()

        status = handler.get_quota_status()
        assert status["configured"] is True
        assert status["key"] == "twjp..._key"  # masked first4...last4
        assert status["usage"]["daily_remaining"] == 37
        assert status["usage"]["daily_limit"] == 100
        assert status["daily_exhausted"] is False
        await handler.close()

    @pytest.mark.asyncio
    async def test_usage_zero_remaining_marks_exhausted(self, handler):
        usage_envelope = {"data": {"requests_remaining": 0, "daily_limit": 100}}
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=usage_envelope, status=200)
            await handler.get_usage()

        assert handler._daily_is_exhausted() is True
        await handler.close()


class TestClassifyRateLimit:
    """FIX 1 — a 429 is minute-scale unless something says otherwise."""

    def test_ambiguous_429_defaults_to_minute(self):
        # No headers, generic body: must NOT be read as daily exhaustion.
        c = _classify_rate_limit({}, "429 Too Many Requests")
        assert c["scale"] == "minute"

    def test_empty_body_and_headers_defaults_to_minute(self):
        c = _classify_rate_limit(None, "")
        assert c["scale"] == "minute"
        assert c["signal"] == "default-minute"

    def test_daily_body_wording_is_daily(self):
        assert _classify_rate_limit({}, "Daily limit exceeded")["scale"] == "daily"
        assert _classify_rate_limit({}, "quota exceeded for today")["scale"] == "daily"
        assert _classify_rate_limit({}, "100 requests/day cap reached")["scale"] == "daily"

    def test_minute_body_wording_is_minute(self):
        assert _classify_rate_limit({}, "Per-minute rate limit exceeded")["scale"] == "minute"
        assert _classify_rate_limit({}, "Slow down")["scale"] == "minute"

    def test_scope_header_minute_beats_daily_body_noise(self):
        # An explicit scope header is the strongest signal.
        c = _classify_rate_limit({"X-RateLimit-Scope": "minute"}, "daily-ish text")
        assert c["scale"] == "minute"

    def test_scope_header_daily(self):
        assert _classify_rate_limit({"RateLimit-Scope": "day"}, "")["scale"] == "daily"

    def test_day_remaining_zero_is_daily(self):
        c = _classify_rate_limit({"X-RateLimit-Remaining-Day": "0"}, "")
        assert c["scale"] == "daily"

    def test_minute_remaining_zero_with_day_left_is_minute(self):
        c = _classify_rate_limit(
            {"X-RateLimit-Remaining-Minute": "0", "X-RateLimit-Remaining-Day": "58"}, ""
        )
        assert c["scale"] == "minute"

    def test_headers_are_case_insensitive(self):
        c = _classify_rate_limit({"x-ratelimit-remaining-day": "0"}, "")
        assert c["scale"] == "daily"

    def test_large_retry_after_is_daily(self):
        c = _classify_rate_limit({"Retry-After": "3600"}, "")
        assert c["scale"] == "daily"
        assert c["retry_after_seconds"] == 3600.0

    def test_short_retry_after_is_minute(self):
        c = _classify_rate_limit({"Retry-After": "12"}, "")
        assert c["scale"] == "minute"
        assert c["retry_after_seconds"] == 12.0


class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert _parse_retry_after("30") == 30.0

    def test_absent(self):
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None

    def test_http_date(self):
        now = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
        secs = _parse_retry_after("Sun, 31 May 2026 12:01:00 GMT", now=now)
        assert secs == 60.0


class TestMinuteScale429DoesNotPark:
    """FIX 1 — a minute-scale 429 must NOT park the daily quota."""

    @pytest.mark.asyncio
    async def test_minute_429_reports_rate_limited_and_leaves_daily_unparked(self, handler):
        with aioresponses() as mocked:
            mocked.get(
                TENNIS_URL_RE,
                status=429,
                headers={"X-RateLimit-Scope": "minute", "Retry-After": "20"},
                body="per-minute rate limit",
            )
            result = await handler.get_live_matches()

        assert result["success"] is False
        assert result["rate_limited"] is True
        assert result["retry_after_seconds"] >= 20
        # The daily quota must remain untouched: no park, no daily short-circuit.
        assert handler._daily_exhausted_until is None
        assert handler._daily_is_exhausted() is False
        assert "quota" not in result
        await handler.close()

    @pytest.mark.asyncio
    async def test_minute_429_drains_bucket_so_next_call_is_local(self, handler):
        # After a minute-scale 429 the local bucket is drained, so the very next
        # call is denied locally without a second upstream request. Only one 429
        # is registered; a second network hit would raise.
        with aioresponses() as mocked:
            mocked.get(
                TENNIS_URL_RE,
                status=429,
                headers={"X-RateLimit-Scope": "minute"},
                body="per-minute rate limit",
            )
            first = await handler.get_live_matches()
            second = await handler.get_live_matches()

        assert first["rate_limited"] is True
        assert second["rate_limited"] is True
        assert second["retry_after_seconds"] > 0
        # Still not a daily park.
        assert handler._daily_exhausted_until is None
        await handler.close()

    @pytest.mark.asyncio
    async def test_daily_429_still_parks(self, handler):
        with aioresponses() as mocked:
            mocked.get(
                TENNIS_URL_RE,
                status=429,
                headers={"X-RateLimit-Remaining-Day": "0"},
                body="daily quota exceeded",
            )
            result = await handler.get_live_matches()

        assert result["success"] is False
        assert result["quota"]["daily_exhausted"] is True
        assert handler._daily_exhausted_until is not None
        await handler.close()


class TestTokenBucketPenalize:
    def test_penalize_drains_to_empty(self):
        bucket = TokenBucket(rate_per_minute=60, capacity=10, time_func=lambda: 1000.0)
        assert bucket.acquire() is True
        bucket.penalize()
        assert bucket.acquire() is False

    def test_penalize_with_seconds_holds_empty_longer(self):
        now = {"t": 1000.0}
        bucket = TokenBucket(rate_per_minute=60, capacity=10, time_func=lambda: now["t"])
        # 60/min == 1 token/sec. Penalize for 5s -> ~5s until the first token.
        bucket.penalize(5.0)
        assert bucket.retry_after() >= 5.0
        now["t"] += 6.0
        assert bucket.acquire() is True


class TestUsageRecovery:
    """FIX 2 — /usage bypasses the daily gate, clears the park, is short-TTL."""

    def test_usage_ttl_is_short_and_independent_of_live(self):
        # Even when the live TTL is raised to the 900s survival knob, /usage
        # keeps its own short TTL so a recovery read is never stale.
        handler = TennisAPIHandler(api_key="twjp_test_key", ttls={"live": 900})
        assert handler.ttls["live"] == 900
        assert handler.ttls["usage"] == 15
        assert handler.ttls["usage"] < handler.ttls["live"]

    @pytest.mark.asyncio
    async def test_usage_reaches_api_while_daily_parked(self, handler):
        # Park the key as if a 429 exhausted the day...
        handler._mark_daily_exhausted()
        assert handler._daily_is_exhausted() is True

        # ...a normal call short-circuits locally (no upstream needed)...
        parked = await handler.get_live_matches()
        assert "Daily request quota exhausted" in parked["error"]

        # ...but /usage still reaches the API and reports real headroom, which
        # clears the park. Only the /usage response is registered; if the daily
        # gate had blocked it, no network call would occur and remaining would
        # never be read.
        usage_envelope = {"data": {"requests_remaining": 63, "daily_limit": 100}}
        with aioresponses() as mocked:
            mocked.get(TENNIS_URL_RE, payload=usage_envelope, status=200)
            result = await handler.get_usage()

        assert result["success"] is True
        assert handler._daily_is_exhausted() is False
        assert handler._daily_exhausted_until is None
        await handler.close()

    @pytest.mark.asyncio
    async def test_positive_remaining_clears_park(self, handler):
        handler._mark_daily_exhausted()
        handler._record_usage({"requests_remaining": 42, "daily_limit": 100})
        assert handler._daily_is_exhausted() is False

    @pytest.mark.asyncio
    async def test_zero_remaining_still_parks(self, handler):
        handler._record_usage({"requests_remaining": 0, "daily_limit": 100})
        assert handler._daily_is_exhausted() is True


class TestQuotaStatus:
    def test_reports_masked_key_and_no_upstream(self):
        handler = TennisAPIHandler(api_key="twjp_abcdefgh")
        status = handler.get_quota_status()
        assert status["configured"] is True
        assert status["key"] != "twjp_abcdefgh"
        assert status["key"].startswith("twjp")
        assert status["per_minute_limit"] == 30
        assert status["daily_exhausted"] is False


class TestNextUtcMidnight:
    def test_is_after_now_and_at_midnight(self):
        base = datetime(2026, 5, 31, 14, 30, tzinfo=timezone.utc)
        nxt = _next_utc_midnight(base)
        assert nxt == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        assert nxt > base


class TestDeriveBreakPoint:
    def test_receiver_advantage_is_break_point(self):
        # server=1, so receiver is p2 (index 1) holding AD.
        score = {"server": 1, "points": ["40", "AD"]}
        assert derive_break_point(score) is True

    def test_receiver_40_vs_server_30_is_break_point(self):
        score = {"server": 1, "points": ["30", "40"]}
        assert derive_break_point(score) is True

    def test_server_ahead_is_not_break_point(self):
        score = {"server": 1, "points": ["40", "30"]}
        assert derive_break_point(score) is False

    def test_deuce_is_not_break_point(self):
        score = {"server": 1, "points": ["40", "40"]}
        assert derive_break_point(score) is False

    def test_tiebreak_is_undefined(self):
        score = {"server": 1, "points": ["6", "5"], "tiebreak": True}
        assert derive_break_point(score) is None

    def test_null_server_is_undefined(self):
        score = {"server": None, "points": ["40", "30"]}
        assert derive_break_point(score) is None

    def test_missing_points_is_undefined(self):
        assert derive_break_point({"server": 1}) is None
