"""Shared HTTP plumbing for source adapters.

Every outbound request in the project goes through here so that the custom
User-Agent, timeout and politeness delay are impossible to forget.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from .. import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_last_call: dict[str, float] = {}
MIN_INTERVAL_S = 1.0  # per-host politeness floor


def _throttle(host: str) -> None:
    with _lock:
        prev = _last_call.get(host, 0.0)
        wait = MIN_INTERVAL_S - (time.monotonic() - prev)
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """GET returning parsed JSON, or None on any failure.

    Source adapters must never raise into the agent loop — a dead source
    degrades the report, it does not abort the run.
    """
    host = httpx.URL(url).host or url
    _throttle(host)
    hdrs = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    hdrs.update(headers or {})
    try:
        resp = httpx.get(
            url,
            params=params,
            headers=hdrs,
            timeout=timeout or config.REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("GET %s failed: %s", url, exc)
        return None


class ProviderUnavailable(RuntimeError):
    """The provider refused the request for an account reason, not a data one.

    402 (out of credits), 401/403 (bad or expired key). These are NOT transient
    and must not be retried — but they also must not look like "no results".
    A Firecrawl credit exhaustion mid-session silently turned every parts
    search into an empty list, and the agent went on to produce a confident
    report with half its sources missing and nothing saying so.
    """


class TransientError(RuntimeError):
    """A request failed for a reason that will probably not recur.

    Raised for rate limits, timeouts and 5xx. It matters because callers cache
    their results: a transient failure that looks like "nothing found" gets
    stored as a permanent negative. Eighteen rapid searches tripped Firecrawl's
    rate limit and four vehicles were cached for 30 days as having no manual.
    """


_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    retries: int = 2,
) -> dict[str, Any] | None:
    host = httpx.URL(url).host or url
    hdrs = {"User-Agent": config.USER_AGENT, "Content-Type": "application/json"}
    hdrs.update(headers or {})

    last: Exception | None = None
    for attempt in range(retries + 1):
        _throttle(host)
        try:
            resp = httpx.post(
                url, json=payload, headers=hdrs,
                timeout=timeout or config.REQUEST_TIMEOUT,
            )
            if resp.status_code in (401, 402, 403):
                raise ProviderUnavailable(
                    f"HTTP {resp.status_code} from {host} "
                    f"({'out of credits' if resp.status_code == 402 else 'auth failed'})"
                )
            if resp.status_code in _TRANSIENT_STATUS:
                raise TransientError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except ProviderUnavailable:
            raise
        except (TransientError, httpx.TimeoutException, httpx.NetworkError) as exc:
            last = exc
            if attempt < retries:
                wait = 2 ** attempt
                log.warning("POST %s transient (%s) — retry in %ss", url, exc, wait)
                time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - genuine failure, no retry
            log.warning("POST %s failed: %s", url, exc)
            return None

    # Exhausted retries on a transient fault. Raise rather than return None so
    # callers do not record this as "no results".
    raise TransientError(f"POST {url} failed after {retries + 1} attempts: {last}")


def truncate(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
