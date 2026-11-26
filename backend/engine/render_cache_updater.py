"""
Automated Background Cache Refresh Pipeline for CricOracle
========================================================
Keeps all local analytics snapshots synchronized automatically without manual scripts:
1. Render Analytics (https://cricket-analytics-ghpv.onrender.com/):
   - render_players_{test,odi,t20}.json
   - render_adaptability_{test,odi,t20}.json (with home-surface substitution)
   - render_player_lookup.json (candidates + full player roster)
   - backend/database/cricket_analytics_players_cache.json + SQLite players/aliases sync
   - Stale entry refresh on render_duels_cache.json
2. Cricsheet Recent Matches & Bayesian Optimization:
   - Triggers CricsheetAutoOptimizer to pull recent.zip and update DB/optimized_parameters.json
3. In-Memory Cache Invalidation:
   - Reloads in-memory caches in render_analytics_bridge and piml_engine after atomic disk writes.
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("cricoracle.cache_updater")

RENDER_BASE_URL = "https://cricket-analytics-ghpv.onrender.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR = Path(__file__).resolve().parent.parent / "database"
PLAYERS_CATALOG_CACHE = DB_DIR / "cricket_analytics_players_cache.json"
DUELS_CACHE_PATH = DATA_DIR / "render_duels_cache.json"
LOOKUP_CACHE_PATH = DATA_DIR / "render_player_lookup.json"
STATE_FILE_PATH = DATA_DIR / ".cache_refresh_state.json"

SURFACES = ("dusty", "green", "bouncy", "flat", "balanced")
RENDER_FORMATS = ("TEST", "ODI", "T20")

DEFAULT_RENDER_TTL_HOURS = float(os.environ.get("RENDER_CACHE_REFRESH_HOURS", "12"))
DEFAULT_CRICSHEET_TTL_HOURS = float(os.environ.get("CRICSHEET_CACHE_REFRESH_HOURS", "24"))

_refresh_lock = asyncio.Lock()
_scheduler_task: Optional[asyncio.Task] = None


def _atomic_write_json(target_path: Path, payload: Any) -> None:
    """Writes JSON atomically via temp file + os.replace so concurrent readers never observe partial writes."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=str(target_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_name, target_path)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise


def _read_json_safe(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return default


def _load_refresh_state() -> Dict[str, Any]:
    return _read_json_safe(STATE_FILE_PATH, {})


def _save_refresh_state(state: Dict[str, Any]) -> None:
    try:
        _atomic_write_json(STATE_FILE_PATH, state)
    except Exception as exc:
        logger.warning(f"Could not save cache refresh state: {exc}")


def _is_stale(iso_ts: Optional[str], ttl_hours: float) -> bool:
    if not iso_ts:
        return True
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) >= timedelta(hours=ttl_hours)
    except Exception:
        return True


async def _fetch_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 3,
    retry_delay: float = 3.0
) -> Optional[Dict[str, Any]]:
    for attempt in range(1, retries + 1):
        try:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 CricOracle/2.0"})
            if resp.status_code == 200:
                return resp.json()
            logger.debug(f"Render API {url} returned HTTP {resp.status_code} (attempt {attempt}/{retries})")
        except Exception as exc:
            logger.debug(f"Render API fetch error for {url} (attempt {attempt}/{retries}): {exc}")
        if attempt < retries:
            await asyncio.sleep(retry_delay * attempt)
    return None


def _enrich_adaptability_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Substitutes missing foreign surface stats with home-surface stats when available."""
    p = dict(rec)
    p_role = str(p.get("player_role") or "BATTER").upper()
    is_bowler = ("BOWL" in p_role) or (p_role in ("ALL_ROUNDER", "BOWLER"))
    for s in SURFACES:
        f_avg = p.get(f"{s}_avg")
        h_avg = p.get(f"{s}_home_avg")
        h_balls = p.get(f"{s}_home_balls") or 0
        if f_avg is None and h_avg is not None and h_balls > 0:
            p[f"{s}_avg"] = h_avg
            if p.get(f"{s}_sr") in (None, 0, 0.0) and p.get(f"{s}_home_sr"):
                p[f"{s}_sr"] = p.get(f"{s}_home_sr")
            p[f"{s}_source"] = "HOME_SURFACE_EXPERIENCE"
        elif f_avg is not None:
            p[f"{s}_source"] = "FOREIGN_DATA"
        else:
            p[f"{s}_source"] = "NO_EXPERIENCE"

        if is_bowler:
            f_econ = p.get(f"{s}_econ")
            h_econ = p.get(f"{s}_home_econ")
            if (f_econ is None or f_econ == 0.0) and h_econ is not None and h_econ > 0.0:
                p[f"{s}_econ"] = h_econ
                p[f"{s}_bowler_source"] = "HOME_SURFACE_EXPERIENCE"
    return p


async def _fetch_all_adaptability(client: httpx.AsyncClient, fmt: str) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    limit = 500
    offset = 0
    while True:
        url = f"{RENDER_BASE_URL}/api/adaptability/players?format={fmt}&limit={limit}&offset={offset}"
        data = await _fetch_json_with_retry(client, url)
        if not data or not isinstance(data.get("players"), list):
            break
        batch = data["players"]
        collected.extend(batch)
        total = int(data.get("total") or len(collected))
        if len(batch) < limit or len(collected) >= total:
            break
        offset += len(batch)
    return collected


async def refresh_render_snapshots(refresh_stale_duels_limit: int = 20) -> Dict[str, Any]:
    """
    Fetches fresh player profiles, 5-surface adaptability, full player catalog, and lookup tables
    from Render Analytics, merges safely with existing snapshots, updates SQLite DB, and reloads memory caches.
    """
    summary: Dict[str, Any] = {
        "players_by_format": {},
        "adaptability_by_format": {},
        "lookup_count": 0,
        "catalog_count": 0,
        "duels_refreshed": 0,
        "errors": []
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Cold-start friendly timeout (Render free/sleep instances can take ~5-8s on first request)
    timeout = httpx.Timeout(25.0, connect=10.0)
    lookup_merged: Dict[str, Any] = dict(_read_json_safe(LOOKUP_CACHE_PATH, {}))

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # 1. Per-format Player Profiles & 5-Way Pitch Adaptability
        for fmt in RENDER_FORMATS:
            p_path = DATA_DIR / f"render_players_{fmt.lower()}.json"
            existing_p: Dict[str, Any] = _read_json_safe(p_path, {})
            p_data = await _fetch_json_with_retry(client, f"{RENDER_BASE_URL}/api/players?format={fmt}&limit=3000")
            if p_data and isinstance(p_data.get("players"), list) and len(p_data["players"]) >= 20:
                fresh_p = {}
                for p in p_data["players"]:
                    pid = p.get("id")
                    if not pid:
                        continue
                    fresh_p[pid] = p
                    pname = str(p.get("name") or "").strip()
                    if pname:
                        lookup_merged[pname.lower()] = {
                            "id": pid,
                            "name": pname,
                            "country": p.get("country"),
                            "role": p.get("player_role") or p.get("role"),
                            "style": p.get("bowling_style") or p.get("batting_style")
                        }
                merged_p = {**existing_p, **fresh_p}
                if len(merged_p) >= max(20, int(len(existing_p) * 0.5)):
                    _atomic_write_json(p_path, merged_p)
                summary["players_by_format"][fmt] = len(merged_p)
            else:
                summary["errors"].append(f"players_{fmt}_fetch_failed")

            a_path = DATA_DIR / f"render_adaptability_{fmt.lower()}.json"
            existing_a: Dict[str, Any] = _read_json_safe(a_path, {})
            raw_adapt = await _fetch_all_adaptability(client, fmt)
            if len(raw_adapt) >= 20:
                fresh_a: Dict[str, Any] = {}
                for rec in raw_adapt:
                    enriched = _enrich_adaptability_record(rec)
                    pid = enriched.get("player_id")
                    pname = str(enriched.get("player_name") or "").strip()
                    if pid:
                        fresh_a[pid] = enriched
                    if pname:
                        fresh_a[pname.lower()] = enriched
                merged_a = {**existing_a, **fresh_a}
                if len(merged_a) >= max(20, int(len(existing_a) * 0.5)):
                    _atomic_write_json(a_path, merged_a)
                summary["adaptability_by_format"][fmt] = len(merged_a)
            else:
                summary["errors"].append(f"adaptability_{fmt}_fetch_failed")

        # 2. Full Global Players Catalog + Lookup Enrichment
        all_p_data = await _fetch_json_with_retry(client, f"{RENDER_BASE_URL}/api/players?limit=5000")
        existing_catalog = _read_json_safe(PLAYERS_CATALOG_CACHE, [])
        if all_p_data and isinstance(all_p_data.get("players"), list) and len(all_p_data["players"]) >= 100:
            raw_cat = all_p_data["players"]
            # Merge by ID so legacy profiles aren't lost
            by_id = {p["id"]: p for p in existing_catalog if isinstance(p, dict) and p.get("id")}
            for p in raw_cat:
                if p.get("id"):
                    by_id[p["id"]] = p
                    pname = str(p.get("name") or "").strip()
                    if pname:
                        lookup_merged[pname.lower()] = {
                            "id": p["id"],
                            "name": pname,
                            "country": p.get("country"),
                            "role": p.get("player_role") or p.get("role"),
                            "style": p.get("bowling_style") or p.get("batting_style")
                        }
            merged_cat = list(by_id.values())
            _atomic_write_json(PLAYERS_CATALOG_CACHE, merged_cat)
            summary["catalog_count"] = len(merged_cat)
        else:
            summary["errors"].append("global_players_catalog_fetch_failed")

        # 3. Enrich lookup candidates from /api/duel
        duel_meta = await _fetch_json_with_retry(
            client, f"{RENDER_BASE_URL}/api/duel?batter_id=08548b13&bowler_id=45a7e761&format=TEST"
        )
        if duel_meta:
            for item in (duel_meta.get("candidate_batters", []) + duel_meta.get("candidate_bowlers", [])):
                name = str(item.get("name") or "").strip()
                pid = item.get("id")
                if name and pid:
                    lookup_merged[name.lower()] = {
                        "id": pid,
                        "name": name,
                        "country": item.get("country"),
                        "role": item.get("role"),
                        "style": item.get("style")
                    }

        if lookup_merged:
            _atomic_write_json(LOOKUP_CACHE_PATH, lookup_merged)
            summary["lookup_count"] = len(lookup_merged)

        # 4. Bounded Refresh of Oldest / Unstamped Entries in render_duels_cache.json
        duels_cache: Dict[str, Any] = dict(_read_json_safe(DUELS_CACHE_PATH, {}))
        if duels_cache and refresh_stale_duels_limit > 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            stale_keys: List[str] = []
            for k, val in duels_cache.items():
                if isinstance(val, dict) and _is_stale(val.get("_updated_at"), 168.0):  # 7 days
                    stale_keys.append(k)
            refreshed_duels = 0
            for k in stale_keys[:refresh_stale_duels_limit]:
                entry = duels_cache.get(k) or {}
                b_id = entry.get("batter_id")
                bw_id = entry.get("bowler_id")
                parts = k.split("_")
                fmt_q = parts[-1] if len(parts) >= 3 else "T20"
                if not (b_id and bw_id):
                    continue
                d_data = await _fetch_json_with_retry(
                    client,
                    f"{RENDER_BASE_URL}/api/duel?batter_id={b_id}&bowler_id={bw_id}&format={fmt_q}",
                    retries=1
                )
                if d_data and d_data.get("success"):
                    duels_cache[k] = {
                        "batter_id": b_id,
                        "bowler_id": bw_id,
                        "batter_name": d_data.get("batter", {}).get("name") or entry.get("batter_name"),
                        "bowler_name": d_data.get("bowler", {}).get("name") or entry.get("bowler_name"),
                        "expected_runs_per_over": d_data.get("simulated_over", {}).get("expected_runs", entry.get("expected_runs_per_over", 5.5)),
                        "wicket_risk_pct": d_data.get("simulated_over", {}).get("wicket_risk_pct", entry.get("wicket_risk_pct", 20.0)),
                        "dot_pct": d_data.get("probabilities", {}).get("dot_pct", entry.get("dot_pct", 45.0)),
                        "boundary_pct": d_data.get("probabilities", {}).get("boundary_pct", entry.get("boundary_pct", 12.0)),
                        "rotation_pct": d_data.get("probabilities", {}).get("rotation_pct", entry.get("rotation_pct", 35.0)),
                        "batter_power_rating": d_data.get("batter", {}).get("power_rating", entry.get("batter_power_rating", 50.0)),
                        "bowler_power_rating": d_data.get("bowler", {}).get("power_rating", entry.get("bowler_power_rating", 50.0)),
                        "tactical_edge": d_data.get("simulated_over", {}).get("tactical_edge", entry.get("tactical_edge", "NEUTRAL")),
                        "_updated_at": now_iso,
                    }
                    refreshed_duels += 1
            if refreshed_duels > 0:
                _atomic_write_json(DUELS_CACHE_PATH, duels_cache)
            summary["duels_refreshed"] = refreshed_duels

    # 5. Upsert updated catalog into SQLite DB & invalidate in-memory caches
    try:
        from ..database.db import get_db
        from .cricket_analytics_matcher import sync_cricket_analytics_players_to_db
        conn = await get_db()
        try:
            summary["db_players_synced"] = await sync_cricket_analytics_players_to_db(conn)
        finally:
            await conn.close()
    except Exception as db_exc:
        logger.debug(f"DB player sync during cache refresh skipped: {db_exc}")

    try:
        from .render_analytics_bridge import invalidate_render_caches
        invalidate_render_caches()
    except Exception:
        pass
    try:
        from .piml_engine import invalidate_piml_render_cache
        invalidate_piml_render_cache()
    except Exception:
        pass

    return summary


async def refresh_all_caches_if_stale(force: bool = False) -> Dict[str, Any]:
    """
    Checks TTL timestamps and refreshes Render Analytics caches (>12h default)
    and Cricsheet historical calibration (>24h default) cleanly under an async lock.
    """
    async with _refresh_lock:
        state = _load_refresh_state()
        now_iso = datetime.now(timezone.utc).isoformat()
        results: Dict[str, Any] = {"render": "skipped_fresh", "cricsheet": "skipped_fresh"}

        if force or _is_stale(state.get("last_render_refresh_utc"), DEFAULT_RENDER_TTL_HOURS):
            logger.info("Auto-refreshing Render analytics snapshots from https://cricket-analytics-ghpv.onrender.com/ ...")
            r_sum = await refresh_render_snapshots()
            results["render"] = r_sum
            if not r_sum.get("errors") or r_sum.get("catalog_count", 0) > 0:
                state["last_render_refresh_utc"] = now_iso
                state["last_render_summary"] = r_sum

        if force or _is_stale(state.get("last_cricsheet_refresh_utc"), DEFAULT_CRICSHEET_TTL_HOURS):
            try:
                from .cricsheet_auto_updater import CricsheetAutoOptimizer
                logger.info("Auto-refreshing Cricsheet recent matches and recalibrating parameters...")
                c_sum = await CricsheetAutoOptimizer().execute_full_recalibration(max_matches=250)
                results["cricsheet"] = {
                    "status": c_sum.get("status"),
                    "total_matches_processed": c_sum.get("total_matches_processed", 0),
                    "senior_international_matches": c_sum.get("senior_international_matches"),
                    "database_updates": c_sum.get("database_updates"),
                }
                if c_sum.get("status") == "success" and int(c_sum.get("total_matches_processed") or 0) > 0:
                    state["last_cricsheet_refresh_utc"] = now_iso
                    state["last_cricsheet_summary"] = results["cricsheet"]
            except Exception as exc:
                logger.warning(f"Automated Cricsheet refresh skipped/failed cleanly: {exc}")
                results["cricsheet"] = {"status": "error", "detail": str(exc)}

        _save_refresh_state(state)
        return results


def get_cache_refresh_status() -> Dict[str, Any]:
    state = _load_refresh_state()
    return {
        "render_ttl_hours": DEFAULT_RENDER_TTL_HOURS,
        "cricsheet_ttl_hours": DEFAULT_CRICSHEET_TTL_HOURS,
        "render_stale": _is_stale(state.get("last_render_refresh_utc"), DEFAULT_RENDER_TTL_HOURS),
        "cricsheet_stale": _is_stale(state.get("last_cricsheet_refresh_utc"), DEFAULT_CRICSHEET_TTL_HOURS),
        "last_render_refresh_utc": state.get("last_render_refresh_utc"),
        "last_cricsheet_refresh_utc": state.get("last_cricsheet_refresh_utc"),
        "last_render_summary": state.get("last_render_summary"),
        "last_cricsheet_summary": state.get("last_cricsheet_summary"),
        "scheduler_active": _scheduler_task is not None and not _scheduler_task.done(),
    }


async def _background_refresh_loop() -> None:
    check_interval_sec = max(1800, int(min(DEFAULT_RENDER_TTL_HOURS, DEFAULT_CRICSHEET_TTL_HOURS) * 3600))
    # Small startup yield so port binds and health checks succeed immediately
    await asyncio.sleep(5)
    while True:
        try:
            await refresh_all_caches_if_stale(force=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Background cache refresh loop error: {exc}")
        try:
            await asyncio.sleep(check_interval_sec)
        except asyncio.CancelledError:
            raise


def start_background_cache_scheduler() -> asyncio.Task:
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_background_refresh_loop())
        logger.info(
            f"Automated cache refresh scheduler started (Render TTL={DEFAULT_RENDER_TTL_HOURS}h, Cricsheet TTL={DEFAULT_CRICSHEET_TTL_HOURS}h)."
        )
    return _scheduler_task


async def stop_background_cache_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
    _scheduler_task = None
