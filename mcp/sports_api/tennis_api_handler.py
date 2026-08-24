"""
Live Tennis API Handler
=======================
Handler for the Live Tennis API (https://livetennisapi.com), covering live
scores, fixtures, and player lookups for ATP, WTA, Challenger and ITF.

Vendor disclosure: this provider is authored by the Live Tennis API team, who
proposed it in issue #86. Judge it on the merits — it follows the same handler
pattern as `espn_api_handler.py` and `odds_api_handler.py` and shares the same
`ResponseCache`.

Tier contract (this is the whole point of the docstring — read it):

    Free tier for fixtures, player lookups and low-cadence score checks;
    paid tier for live match-following.

The constraint that makes that split real is the free key's daily quota, not
entitlement. Live scores ARE included on the free tier; the free key is just
30 requests/minute and **100 requests/day**. At the default 30s live cache TTL
a single three-hour match refreshes ~360 times — so following one match to the
end exhausts a free key's whole day inside that one match. A free key sustains
roughly one score check every 15 minutes (~96/day), plus fixtures and player
lookups, plus develop-and-test. Following live state for real is Basic
($9.99/mo, 60/min · 1k/day) or, for a full Slam board, Pro (300/min · 10k/day).

Raising `CACHE_TTL_TENNIS_LIVE` is the free-tier survival knob: a 900s TTL
keeps a whole day of casual score-checking inside the free 100/day allowance.

v1 scope is deliberately the surface a free key can actually exercise: live
scores, fixtures, and player lookups. Rank-ordered rankings (Pro) and H2H
(Basic) are intentionally left out rather than shipped as tools that fail on
the key our own setup docs tell people to get. Match prices are unaffected and
stay with the Odds API's `get_odds` and its `tennis_atp_*` passthrough.

Two things this handler does NOT reuse, on purpose:

* `key_manager.py` — it models a monthly, sticky-until-exhausted quota with no
  per-minute concept, and its marker-string matching would classify a 429
  burst as a drained key and park it for the process. Tennis is 30/min plus
  100/day, so this handler carries its own token-bucket minute limiter and
  tracks the daily quota honestly (from a 429 or from `/usage`), never routing
  through the manager.
* the team formatters in `formatter.py` — tennis output is player-shaped and
  set-by-set, formatted in the tool layer.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

import aiohttp

from .cache import ResponseCache, make_key

logger = logging.getLogger(__name__)

# Cache lifetimes by endpoint kind, in seconds. Live scores must stay fresh;
# fixtures and player profiles change slowly. `live` defaults to 30s to match
# the ESPN scoreboard, and the composition root lets CACHE_TTL_TENNIS_LIVE
# override it — that override is the free-tier survival knob (see module doc).
DEFAULT_TTLS = {
    "live": 30,       # live match list and single-match score
    "fixtures": 300,  # upcoming schedule
    "player": 3600,   # player profile, ranking, Elo
    "usage": 15,      # /usage recovery read — kept short on purpose (see below)
}

# /usage is the documented recovery call: it is free, it does not spend the
# daily allowance, and it is the only way to check a locally-guessed daily park
# against the API's own count. It gets its OWN short TTL rather than sharing
# `live` (which the composition root may raise to 900s as the survival knob), so
# a recovery check is never answered from a 15-minute-old cache.

# Free-tier limits, used as defaults for the minute limiter. Basic/Pro raise
# the per-minute rate; the daily allowance is enforced by the API and observed
# here rather than assumed.
FREE_TIER_RPM = 30
FREE_TIER_RPD = 100


def _next_utc_midnight(now: Optional[datetime] = None) -> datetime:
    """The next 00:00 UTC — when a daily quota resets."""
    now = now or datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)


# A 429 can mean either the per-minute OR the per-day window was exceeded. The
# process-local token bucket starts full, so a restart mid-minute — or a second
# client sharing the key — sends a request the bucket thinks is within budget
# but the API rejects as a MINUTE burst. Parking the daily quota on that wastes
# ~90 of the day's 100 requests until UTC midnight. So a 429 is classified
# before it is allowed to park the day: only a genuine daily 429 parks, and
# everything ambiguous is treated as a minute-scale back-off (a wrong minute
# back-off costs ~2 seconds; a wrong daily park costs the rest of the day).
_MINUTE_SCALE_MAX_SECONDS = 120


def _to_int(value) -> Optional[int]:
    """Best-effort int parse; returns None on anything non-numeric."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_retry_after(value: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    """Parse a Retry-After header to seconds-from-now.

    Handles both documented forms: delta-seconds ("120") and an HTTP-date
    ("Wed, 21 Oct 2026 07:28:00 GMT"). Returns None when absent or unparseable.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _classify_rate_limit(headers, body_text: str, now: Optional[datetime] = None) -> Dict[str, object]:
    """Classify a 429 as ``minute``-scale or ``daily``-scale.

    Signals are consulted strongest-first: an explicit rate-limit scope/window
    header, then per-window ``remaining`` headers, then the response body
    wording, then the magnitude of Retry-After. When nothing positively says
    "daily", the answer is ``minute`` — the safe default, because only a
    genuine daily 429 should park the key until reset.

    Args:
        headers: The response headers (aiohttp CIMultiDict or a plain dict).
        body_text: The 429 response body.
        now: Injectable clock for Retry-After date math (testing).

    Returns:
        ``{"scale", "retry_after_seconds", "signal"}``.
    """
    # Normalise to a case-insensitive lookup that works for both aiohttp's
    # CIMultiDict and a plain dict passed by a unit test.
    lowered: Dict[str, object] = {}
    if headers:
        try:
            items = list(headers.items())
        except AttributeError:
            items = []
        for k, v in items:
            lowered[str(k).lower()] = v

    def h(*names):
        for n in names:
            v = lowered.get(n.lower())
            if v is not None:
                return v
        return None

    retry_after = _parse_retry_after(h("Retry-After"), now=now)
    body = (body_text or "").lower()

    def result(scale: str, signal: str) -> Dict[str, object]:
        return {"scale": scale, "retry_after_seconds": retry_after, "signal": signal}

    # 1. Explicit scope/window header, e.g. "X-RateLimit-Scope: minute".
    scope = h("X-RateLimit-Scope", "RateLimit-Scope", "X-RateLimit-Window", "RateLimit-Window")
    if scope is not None:
        s = str(scope).lower()
        if any(w in s for w in ("day", "daily", "date")):
            return result("daily", f"scope={scope}")
        if any(w in s for w in ("min", "sec", "burst")):
            return result("minute", f"scope={scope}")

    # 2. Per-window remaining headers. A day window at 0 is daily; a minute
    #    window at 0 while the day still has room is minute-scale.
    day_remaining = _to_int(
        h("X-RateLimit-Remaining-Day", "X-RateLimit-Daily-Remaining", "X-RateLimit-Remaining-Daily")
    )
    min_remaining = _to_int(
        h("X-RateLimit-Remaining-Minute", "X-RateLimit-Remaining-Min")
    )
    if day_remaining is not None and day_remaining <= 0:
        return result("daily", "day-remaining<=0")
    if min_remaining is not None and min_remaining <= 0 and (day_remaining is None or day_remaining > 0):
        return result("minute", "minute-remaining<=0")

    # 3. Body wording. Check the daily vocabulary first so "daily rate limit"
    #    is read as daily rather than tripping the generic minute markers.
    if any(w in body for w in ("per day", "daily", "day limit", "quota exceeded", "requests/day")):
        return result("daily", "body=daily")
    if any(w in body for w in ("per minute", "per-minute", "minute", "per second", "rate limit", "too many", "slow down")):
        return result("minute", "body=minute")

    # 4. Retry-After magnitude: a back-off longer than a minute window is
    #    day-scale; a short one is a minute burst.
    if retry_after is not None and retry_after > _MINUTE_SCALE_MAX_SECONDS:
        return result("daily", "retry-after-large")

    # 5. Default: minute-scale (see module note above).
    return result("minute", "default-minute")


class TokenBucket:
    """A simple per-minute token bucket.

    The free tier is 30 requests/minute. Rather than route a 429 burst through
    the quota-model key manager (which would misclassify it as an exhausted
    key and park it), this gates outgoing requests locally: `acquire()` returns
    False when the minute's budget is spent, and the caller reports a
    rate-limited result without touching the network or the daily quota.

    `time_func` is injectable so the refill is testable without real sleeping.
    """

    def __init__(
        self,
        rate_per_minute: int = FREE_TIER_RPM,
        capacity: Optional[int] = None,
        time_func: Callable[[], float] = time.monotonic,
    ):
        self.rate_per_minute = max(1, rate_per_minute)
        self.rate_per_second = self.rate_per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else self.rate_per_minute)
        self._tokens = self.capacity
        self._time = time_func
        self._last = self._time()

    def _refill(self) -> None:
        now = self._time()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
            self._last = now

    def acquire(self, tokens: int = 1) -> bool:
        """Take `tokens` from the bucket. Returns False if not enough remain."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def retry_after(self, tokens: int = 1) -> float:
        """Seconds until `tokens` would be available, for an honest hint."""
        self._refill()
        deficit = tokens - self._tokens
        if deficit <= 0:
            return 0.0
        return round(deficit / self.rate_per_second, 1)

    def penalize(self, seconds: float = 0.0) -> None:
        """Empty the bucket, optionally holding it empty for `seconds`.

        Called when the API reports a minute-scale 429 the local bucket did not
        predict — a restart reset it to full, or a second client shares the key.
        Draining aligns the local view with reality so we stop hammering for the
        rest of the window instead of re-issuing on the next full-bucket tick.
        `seconds` (e.g. a Retry-After hint) pushes the bucket negative so the
        next token is not available until roughly that long from now.
        """
        self._refill()
        self._tokens = 0.0
        if seconds and seconds > 0:
            self._tokens = -max(0.0, float(seconds)) * self.rate_per_second


class TennisAPIHandler:
    """Handler for the Live Tennis API (free-tier surface)."""

    BASE_URL = "https://api.livetennisapi.com/api/public/v1"

    def __init__(
        self,
        api_key: str,
        cache: Optional[ResponseCache] = None,
        ttls: Optional[Dict[str, int]] = None,
        rate_per_minute: int = FREE_TIER_RPM,
        rate_limiter: Optional[TokenBucket] = None,
    ):
        """Initialize the Live Tennis API handler.

        Args:
            api_key: A Live Tennis API key (sent as the `X-API-Key` header).
            cache: Optional shared ResponseCache. One is created if omitted.
            ttls: Optional per-kind TTL overrides, merged over DEFAULT_TTLS.
            rate_per_minute: Minute budget for the token bucket (free = 30).
            rate_limiter: Inject a pre-built TokenBucket (used by tests).
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = cache if cache is not None else ResponseCache()
        self.ttls = {**DEFAULT_TTLS, **(ttls or {})}
        self._bucket = rate_limiter or TokenBucket(rate_per_minute=rate_per_minute)

        # Daily-quota state, learned from a 429 or a `/usage` read. `until` is a
        # UTC datetime while the key is known-exhausted, else None.
        self._daily_exhausted_until: Optional[datetime] = None
        # Last observed usage figures, surfaced by get_quota_status() with no
        # upstream call (mirrors how the Odds handler reports tracked quota).
        self._usage: Dict[str, object] = {}

    # -- quota / rate reporting ------------------------------------------------

    @staticmethod
    def _mask_key(key: str) -> str:
        """Show only the first and last 4 characters of a key."""
        if not key:
            return ""
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"

    def _daily_is_exhausted(self, now: Optional[datetime] = None) -> bool:
        """True while a known daily-quota exhaustion is still in effect."""
        if self._daily_exhausted_until is None:
            return False
        now = now or datetime.now(timezone.utc)
        if now >= self._daily_exhausted_until:
            # The reset time has passed; clear the flag and let requests flow.
            self._daily_exhausted_until = None
            return False
        return True

    def _mark_daily_exhausted(self) -> None:
        """Park the key until the next UTC midnight after a daily-limit hit."""
        self._daily_exhausted_until = _next_utc_midnight()
        logger.warning(
            "Live Tennis API daily quota exhausted; parking until %s UTC",
            self._daily_exhausted_until.isoformat(),
        )

    def get_quota_status(self) -> Dict:
        """Rate-limit and daily-quota health, with the key masked.

        This makes no upstream call: it reports the token bucket's current
        state plus the last observed daily usage, so folding it into
        get_api_status() stays quota-free.
        """
        self._bucket._refill()
        status = {
            "configured": True,
            "key": self._mask_key(self.api_key),
            "per_minute_limit": self._bucket.rate_per_minute,
            "minute_tokens_available": int(self._bucket._tokens),
            "daily_exhausted": self._daily_is_exhausted(),
        }
        if self._daily_exhausted_until is not None:
            status["daily_resets_at"] = self._daily_exhausted_until.isoformat()
        if self._usage:
            status["usage"] = dict(self._usage)
        return status

    def _record_usage(self, usage: Dict) -> None:
        """Fold a `/usage`-shaped block into tracked state.

        Tolerant of key spelling because the field names are the one part of
        the shape a docstring cannot pin down: accepts remaining/limit/reset
        under a few common names, ignores what it does not recognise.
        """
        if not isinstance(usage, dict):
            return

        def pick(*names):
            for n in names:
                if n in usage and usage[n] is not None:
                    return usage[n]
            return None

        remaining = pick("requests_remaining", "remaining", "daily_remaining")
        limit = pick("daily_limit", "limit", "requests_limit")
        reset = pick("resets_at", "reset", "reset_at")
        plan = pick("plan", "tier")

        recorded: Dict[str, object] = {}
        if remaining is not None:
            recorded["daily_remaining"] = remaining
        if limit is not None:
            recorded["daily_limit"] = limit
        if reset is not None:
            recorded["resets_at"] = reset
        if plan is not None:
            recorded["plan"] = plan
        if recorded:
            self._usage = recorded

        # The reported remaining count is authoritative in BOTH directions:
        #   * <= 0 is the same signal as a daily 429 — park the key.
        #   * > 0 means the key is usable right now, so clear any park. This is
        #     what makes /usage a real recovery call: if a 429 was misclassified
        #     as daily exhaustion, the API's own count releases the key.
        try:
            if remaining is not None:
                if int(remaining) <= 0:
                    self._mark_daily_exhausted()
                elif self._daily_exhausted_until is not None:
                    logger.info(
                        "Live Tennis API reports %s requests remaining; clearing daily park",
                        remaining,
                    )
                    self._daily_exhausted_until = None
        except (TypeError, ValueError):
            pass

    # -- HTTP plumbing ---------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session with the auth header attached."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={"X-API-Key": self.api_key})
        return self.session

    @staticmethod
    def _is_cacheable(result: Dict) -> bool:
        """Never cache a failure; a blip would become a full TTL of failure."""
        return bool(result.get("success"))

    async def _make_request(
        self,
        endpoint: str,
        params: Dict = None,
        cache_kind: str = "live",
        bypass_daily_gate: bool = False,
    ) -> Dict:
        """Make a request, serving from cache when possible.

        The API key is deliberately excluded from the cache key: the payload is
        identical whichever key fetched it, and keying on it would throw the
        cache away on rotation. Cache hits carry `"cached": True`.

        `bypass_daily_gate` lets the free `/usage` recovery read reach the API
        even while the key is locally parked (see get_usage / _fetch).
        """
        params = dict(params or {})
        key = make_key("tennis", endpoint, params)
        ttl = self.ttls.get(cache_kind, 60)

        result, was_hit = await self.cache.get_or_fetch(
            key,
            ttl,
            lambda: self._fetch(endpoint, params, bypass_daily_gate=bypass_daily_gate),
            should_cache=self._is_cacheable,
        )

        if was_hit:
            result = {**result, "cached": True}

        return result

    async def _fetch(self, endpoint: str, params: Dict, bypass_daily_gate: bool = False) -> Dict:
        """Perform one live request, gated by the daily quota and minute rate.

        Order matters: a known daily exhaustion short-circuits before the
        minute limiter, and both short-circuit before the network, so neither a
        drained day nor a spent minute costs a request or an upstream round
        trip.

        `bypass_daily_gate=True` skips only the daily short-circuit — the free
        `/usage` read must reach the API even while parked, because it is the
        only way to check the local park against the real count. The minute
        limiter still applies: it is a live request.
        """
        if not bypass_daily_gate and self._daily_is_exhausted():
            reset = self._daily_exhausted_until
            return {
                "success": False,
                "error": "Daily request quota exhausted for this Live Tennis API key",
                "quota": {
                    "daily_exhausted": True,
                    "resets_at": reset.isoformat() if reset else None,
                    "note": (
                        "Free tier is 100 requests/day. Raise CACHE_TTL_TENNIS_LIVE "
                        "or use a Basic/Pro key to follow live matches."
                    ),
                },
            }

        if not self._bucket.acquire():
            return {
                "success": False,
                "error": "Per-minute rate limit reached for this Live Tennis API key",
                "rate_limited": True,
                "retry_after_seconds": self._bucket.retry_after(),
            }

        url = f"{self.BASE_URL}{endpoint}"
        session = await self._get_session()

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # An envelope may carry a usage block (top-level or under
                    # `meta`); fold it into tracked quota if present.
                    if isinstance(data, dict):
                        usage = data.get("usage")
                        if usage is None and isinstance(data.get("meta"), dict):
                            usage = data["meta"].get("usage")
                        if usage:
                            self._record_usage(usage)
                    return {"success": True, "data": data, "cached": False}

                error_text = await response.text()

                if response.status == 429:
                    # The local minute bucket starts full, so a 429 here does
                    # NOT prove the day is drained — a restart or a second
                    # client on the same key produces a MINUTE-scale 429 the
                    # bucket never saw. Classify before parking: only a genuine
                    # daily 429 calls _mark_daily_exhausted(); a minute burst is
                    # a minute back-off that leaves the daily quota untouched.
                    classification = _classify_rate_limit(response.headers, error_text)
                    logger.warning(
                        "Live Tennis API 429 (%s): %s",
                        classification["signal"],
                        error_text,
                    )
                    if classification["scale"] == "daily":
                        self._mark_daily_exhausted()
                        return {
                            "success": False,
                            "error": "Live Tennis API daily quota limit hit (429)",
                            "details": error_text,
                            "quota": {
                                "daily_exhausted": True,
                                "resets_at": self._daily_exhausted_until.isoformat()
                                if self._daily_exhausted_until
                                else None,
                            },
                        }

                    # Minute-scale: align the local bucket with the API's view
                    # so we stop hammering, but leave the daily quota alone.
                    retry_after = classification["retry_after_seconds"]
                    self._bucket.penalize(retry_after or 0.0)
                    hint = self._bucket.retry_after()
                    if retry_after is not None:
                        hint = max(hint, round(float(retry_after), 1))
                    return {
                        "success": False,
                        "error": "Live Tennis API per-minute rate limit hit (429)",
                        "details": error_text,
                        "rate_limited": True,
                        "retry_after_seconds": hint,
                    }

                if response.status in (401, 403):
                    logger.error("Live Tennis API auth error %s", response.status)
                    return {
                        "success": False,
                        "error": f"Live Tennis API rejected the key (status {response.status})",
                        "details": error_text,
                    }

                logger.error("Live Tennis API error %s: %s", response.status, error_text)
                return {
                    "success": False,
                    "error": f"API returned status {response.status}",
                    "details": error_text,
                }
        except Exception as e:  # noqa: BLE001 - report, never crash the tool
            logger.error("Request failed: %s", str(e))
            return {"success": False, "error": str(e)}

    # -- endpoints (free-tier surface) ----------------------------------------

    async def get_live_matches(
        self,
        tour: Optional[str] = None,
        status: str = "live",
        limit: int = 20,
    ) -> Dict:
        """List matches and their current score.

        Args:
            tour: Optional tour filter — atp, wta, challenger, or itf.
            status: Match status — live (default), upcoming, or completed.
            limit: Maximum number of matches to return.

        Returns:
            Dictionary with the matches envelope under `data`.
        """
        params: Dict[str, object] = {"status": status, "limit": limit}
        if tour:
            params["tour"] = tour.lower()
        return await self._make_request("/matches", params, cache_kind="live")

    async def get_match_score(self, match_id: str) -> Dict:
        """Read a single match's live score — the lowest-latency score endpoint.

        Args:
            match_id: The match id.

        Returns:
            Dictionary with the score envelope under `data`.
        """
        return await self._make_request(f"/matches/{match_id}/score", cache_kind="live")

    async def get_fixtures(
        self,
        tour: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 20,
    ) -> Dict:
        """Upcoming matches, so callers know what is about to start.

        Args:
            tour: Optional tour filter — atp, wta, challenger, or itf.
            date: Optional date filter in YYYY-MM-DD.
            limit: Maximum number of fixtures to return.

        Returns:
            Dictionary with the fixtures envelope under `data`.
        """
        params: Dict[str, object] = {"limit": limit}
        if tour:
            params["tour"] = tour.lower()
        if date:
            params["date"] = date
        return await self._make_request("/fixtures", params, cache_kind="fixtures")

    async def get_player(self, player_id: str) -> Dict:
        """A player's profile, including current ranking and Elo rating.

        Args:
            player_id: The player id.

        Returns:
            Dictionary with the player envelope under `data`.
        """
        return await self._make_request(f"/players/{player_id}", cache_kind="player")

    async def get_usage(self) -> Dict:
        """Read the free-tier `/usage` endpoint and fold it into tracked quota.

        `/usage` is itself free and does not spend the daily allowance, so this
        can be called to refresh the numbers get_quota_status() reports — and,
        critically, to recover from a wrong daily park. It therefore bypasses
        the daily short-circuit (otherwise the one call that could correct a bad
        park would be blocked by that very park) and is cached under its own
        short `usage` TTL so recovery is never answered from a stale `live`
        cache. A positive remaining in the response clears the park, via
        _record_usage.
        """
        result = await self._make_request("/usage", cache_kind="usage", bypass_daily_gate=True)
        if result.get("success"):
            data = result.get("data")
            if isinstance(data, dict):
                self._record_usage(data.get("data") if isinstance(data.get("data"), dict) else data)
        return result

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


# -- domain helper -----------------------------------------------------------

def derive_break_point(score: Dict) -> Optional[bool]:
    """Derive whether the current point is a break point from a score object.

    Applies the documented rule: it is a break point when the receiver holds
    advantage, or the receiver has 40 while the server has 0/15/30. It never
    holds inside a tiebreak, and is undefined (None) when the server or the
    per-player points are unknown.

    `score` is the Live Tennis API score object: `server` is 1, 2 or null, and
    `points` is a two-element list [player1_point, player2_point] using the
    tennis point labels ("0", "15", "30", "40", "AD").
    """
    if not isinstance(score, dict):
        return None
    if score.get("tiebreak") or score.get("is_tiebreak"):
        return None

    server = score.get("server")
    points = score.get("points")
    if server not in (1, 2) or not isinstance(points, (list, tuple)) or len(points) < 2:
        return None

    server_idx = server - 1
    receiver_idx = 1 - server_idx
    server_pt = str(points[server_idx])
    receiver_pt = str(points[receiver_idx])

    if receiver_pt == "AD":
        return True
    if receiver_pt == "40" and server_pt in ("0", "15", "30"):
        return True
    return False
