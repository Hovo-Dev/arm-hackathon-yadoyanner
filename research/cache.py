"""On-disk cache for external lookups.

Why a cache rather than pre-built local data
--------------------------------------------
A local corpus has to guess which vehicles matter before anyone asks. That
guess cannot win: the fleet in Armenia spans decades of Japanese, European and
Russian imports, and 50 downloaded manuals covered 8 makes while producing four
provenance bugs — a French manual indexed as English, a Honda query reading
Nissan specs, motorcycle manuals filed under a car make, and a Civic manual
labelled as an Accord.

A cache inverts it. Nothing is stored until a real user asks for it, so
coverage is unbounded and storage is proportional to actual demand rather than
to a guess. The manual corpus that prompted this has since been removed
entirely; the cache now serves turn.am's category list and workshop lookups,
where the same reasoning applies — the directory is theirs, not ours to mirror.

It also makes demos reproducible: a warm cache answers with no network at all,
which matters when the venue wifi is bad or a key expires mid-presentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import ROOT

log = logging.getLogger(__name__)

CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_TTL_S = 30 * 24 * 3600  # manual specs do not change; 30 days is cautious


def _key(namespace: str, parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
    return f"{namespace}-{digest}"


def _path(namespace: str, parts: dict[str, Any]) -> Path:
    return CACHE_DIR / namespace / f"{_key(namespace, parts)}.json"


def get(namespace: str, parts: dict[str, Any],
        ttl_s: int = DEFAULT_TTL_S) -> Any | None:
    """Return a cached payload, or None on miss or expiry."""
    path = _path(namespace, parts)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Corrupt cache entry %s (%s) — ignoring.", path.name, exc)
        return None

    age = time.time() - entry.get("stored_at", 0)
    if age > ttl_s:
        log.info("Cache entry %s expired (%.0f days).", path.name, age / 86400)
        return None

    log.info("Cache HIT %s/%s", namespace, path.stem)
    return entry.get("payload")


def put(namespace: str, parts: dict[str, Any], payload: Any) -> None:
    path = _path(namespace, parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(
                {"stored_at": time.time(), "query": parts, "payload": payload},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - caching must never break a run
        log.warning("Could not write cache entry %s: %s", path.name, exc)


def stats() -> dict[str, Any]:
    if not CACHE_DIR.exists():
        return {"entries": 0, "bytes": 0, "namespaces": {}}
    per_ns: dict[str, int] = {}
    total = size = 0
    for path in CACHE_DIR.rglob("*.json"):
        ns = path.parent.name
        per_ns[ns] = per_ns.get(ns, 0) + 1
        total += 1
        size += path.stat().st_size
    return {"entries": total, "bytes": size, "namespaces": per_ns}


def clear(namespace: str | None = None) -> int:
    target = CACHE_DIR / namespace if namespace else CACHE_DIR
    if not target.exists():
        return 0
    removed = 0
    for path in target.rglob("*.json"):
        path.unlink()
        removed += 1
    return removed
