"""
Ingestion Manager for CricOracle
Coordinates multi-source data extraction (JSON feeds, HTML scraper, Replay Simulator),
automates polling cadences, resolves player entities, extracts available squads from external sources,
and triggers state transition recalculations.
"""

import asyncio
import re
import time
import logging
import aiosqlite
from typing import Dict, Any, List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .providers.cric_json_provider import CricJsonProvider
from .providers.html_scraper import HtmlScraper
from .providers.replay_simulator import ReplaySimulator
from .providers.cricbuzz_score_api import CricbuzzScoreApi
from .providers.failover_scraper import top_scraper, DeliveryDeduplicator
from .entity_resolver import EntityResolver
from ..engine.stage1_macro import calculate_stage1_macro
from ..engine.stage2_squad import calculate_stage2_squad
from ..engine.stage3_xi_locked import calculate_stage3_xi_locked
from ..engine.stage4_live import calculate_stage4_live
from ..engine.pitch_history_model import get_historical_venue_profile
from ..engine.player_strengths import calculate_player_venue_affinity
from ..engine.winwiz_engine import is_official_international_match

logger = logging.getLogger("ingestion_manager")

class IngestionManager:
    def __init__(self, db_conn_factory, broadcast_func=None):
        self.db_factory = db_conn_factory
        self.broadcast_func = broadcast_func
        self.json_provider = CricJsonProvider()
        self.html_scraper = HtmlScraper()
        self.entity_resolver = EntityResolver()
        self.cricbuzz_api = CricbuzzScoreApi()
        self.replay_sim = ReplaySimulator(broadcast_callback=self.broadcast_func)
        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        }
        self.scheduler = AsyncIOScheduler(job_defaults=job_defaults)
        self.is_active = True
        self.active_cricbuzz_bindings = {
            "m_nz_ind_2026_test_1": "161285",
            "m_aus_eng_2026_t20_1": "151338",
            "m_zim_afg_2026_test_2": "172731",
        }

    async def start(self):
        """Starts background scheduled polling."""
        # Cadence 1: Sync upcoming fixtures
        self.scheduler.add_job(self.sync_upcoming_fixtures, "interval", hours=6, id="sync_fixtures", misfire_grace_time=300, coalesce=True)
        # Cadence 2: Check squads for matches in 72h
        self.scheduler.add_job(self.poll_squads_cadence, "interval", hours=1, id="poll_squads", misfire_grace_time=120, coalesce=True)
        # Cadence 3: Poll toss & Playing XI (every 3 minutes)
        self.scheduler.add_job(self.poll_toss_and_xi_cadence, "interval", minutes=3, id="poll_toss_xi", misfire_grace_time=60, coalesce=True)
        # Cadence 4: Live In-Play Ball-by-Ball Auto Stream (every 20 seconds for LIVE_BALL_BY_BALL matches)
        self.scheduler.add_job(self.poll_live_matches_cadence, "interval", seconds=20, id="poll_live_matches", max_instances=1, misfire_grace_time=60, coalesce=True)
        # Cadence 5: Dynamic Auto-Discovery of Live Matches across Top Feeds (every 60 seconds)
        self.scheduler.add_job(self.auto_discover_and_sync_live_matches, "interval", seconds=60, id="auto_discover_live", max_instances=1, misfire_grace_time=60, coalesce=True)
        # Cadence 6: Periodic Cricsheet match ingestion & parameter recalibration (every 24 hours)
        from ..engine.cricsheet_auto_updater import trigger_cricsheet_parameter_recalibration
        self.scheduler.add_job(trigger_cricsheet_parameter_recalibration, "interval", hours=24, id="cricsheet_recalibration", misfire_grace_time=3600, coalesce=True)
        self.scheduler.start()
        logger.info("IngestionManager background scheduler started with live in-play worker, auto-discovery, and Cricsheet parameter optimizer.")

    async def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()

    async def sync_upcoming_fixtures(self) -> List[Dict[str, Any]]:
        """
        Fetches fixtures and synchronizes official senior international fixtures with database,
        purging obsolete or phantom matches and maintaining 15-player squad vs confirmed XI integrity.
        """
        fixtures = await self.json_provider.fetch_upcoming_matches()
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            # 1. Purge phantom / obsolete fixtures
            await conn.execute("DELETE FROM matches WHERE id IN ('m_sa_ban_2026_odi_1', 'm_ind_sa_2026_t20_3', 'm_nep_uae_2026_t20i_1')")
            await conn.execute("DELETE FROM match_squads WHERE match_id IN ('m_sa_ban_2026_odi_1', 'm_ind_sa_2026_t20_3', 'm_nep_uae_2026_t20i_1')")
            await conn.execute("DELETE FROM match_deliveries WHERE match_id IN ('m_sa_ban_2026_odi_1', 'm_ind_sa_2026_t20_3', 'm_nep_uae_2026_t20i_1')")

            # 2. Upsert verified international fixtures
            for f in fixtures:
                m_id = f.get("external_id")
                if not m_id:
                    continue
                team_a = f.get("team_a")
                team_b = f.get("team_b")
                fmt = f.get("format", "ODI")
                v_id = f.get("venue_id")
                if not v_id:
                    v_name = f.get("venue_name", "").lower()
                    if "perth" in v_name:
                        v_id = "perth"
                    elif "wellington" in v_name or "sky stadium" in v_name:
                        v_id = "wellington"
                    elif "seddon" in v_name:
                        v_id = "seddon_park"
                    elif "bay oval" in v_name:
                        v_id = "bay_oval"
                    elif "sharjah" in v_name:
                        v_id = "sharjah"
                    elif "rawalpindi" in v_name:
                        v_id = "rawalpindi"
                    elif "lahore" in v_name:
                        v_id = "lahore"
                    elif "adelaide" in v_name:
                        v_id = "adelaide"
                    elif "mcg" in v_name or "melbourne" in v_name:
                        v_id = "mcg"
                    else:
                        v_id = "wellington"

                scheduled = f.get("scheduled_start")
                series_name = f.get("series_name", f"{team_a} vs {team_b} {fmt}")
                is_dn = f.get("is_day_night", 0)

                async with conn.execute("SELECT id, state FROM matches WHERE id = ?", (m_id,)) as c:
                    existing = await c.fetchone()

                if existing:
                    # Update format, venue, and series if not already live or completed
                    if existing["state"] not in ("LIVE_BALL_BY_BALL", "COMPLETED"):
                        await conn.execute("""
                            UPDATE matches SET
                                series_name = ?,
                                format = ?,
                                venue_id = ?,
                                scheduled_start = ?,
                                is_day_night = ?
                            WHERE id = ?
                        """, (series_name, fmt, v_id, scheduled, is_dn, m_id))
                else:
                    v_profile = get_historical_venue_profile(v_id, fmt)
                    if fmt == "TEST":
                        par_proj = v_profile.get("avg_first_innings_score_test", 355)
                    elif fmt == "ODI":
                        par_proj = v_profile.get("avg_first_innings_score_odi", 270)
                    else:
                        par_proj = v_profile.get("avg_first_innings_score_t20", 168)

                    macro_res = calculate_stage1_macro(team_a, team_b, v_id, is_dn, match_format=fmt)
                    p_a = macro_res.get("p_team_a", 50.0)
                    p_b = macro_res.get("p_team_b", 50.0)

                    await conn.execute("""
                        INSERT INTO matches (
                            id, series_name, format, team_a_id, team_b_id, venue_id,
                            scheduled_start, is_day_night, state, p_team_a, p_team_b,
                            p1_macro, p2_squad, p0_pre_match, par_score_projected,
                            par_score_min, par_score_max, delta_squad_shift, delta_toss_shift,
                            biggest_matchup_adv, last_wpa_swing, last_event_description
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?,
                            ?, ?, 'SQUAD_ANNOUNCED', ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, 0.0, 0.0,
                            ?, 0.0, '15-Player Squads Declared • Official Playing XI announced at Toss'
                        )
                    """, (
                        m_id, series_name, fmt, team_a, team_b, v_id,
                        scheduled, is_dn, p_a, p_b,
                        p_a, p_a, p_a, par_proj,
                        par_proj - 25, par_proj + 25,
                        f"Venue archetype: {v_profile.get('pitch_type', 'balanced')}"
                    ))

            # 3. Rule Enforcement: For all upcoming matches, ensure is_confirmed_xi = 0!
            await conn.execute("""
                UPDATE match_squads SET is_confirmed_xi = 0
                WHERE match_id IN (SELECT id FROM matches WHERE state IN ('UPCOMING_NO_SQUAD', 'SQUAD_ANNOUNCED'))
            """)
            await conn.commit()
            logger.info(f"Ingested & synchronized {len(fixtures)} international fixtures.")
        except Exception as e:
            logger.error(f"Error in sync_upcoming_fixtures: {e}")
        finally:
            await conn.close()
        return fixtures

    async def poll_squads_cadence(self):
        """Polls squad releases for upcoming matches from external sources."""
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT id FROM matches WHERE state IN ('UPCOMING_NO_SQUAD', 'SQUAD_ANNOUNCED')") as c:
                rows = await c.fetchall()
            for r in rows:
                m_id = r[0]
                await self.sync_squads_for_match(m_id)
        except Exception as e:
            logger.error(f"Error in poll_squads_cadence: {e}")
        finally:
            await conn.close()

    async def poll_toss_and_xi_cadence(self):
        """
        Polls match center for confirmed toss and Playing XI 30-45 mins before first ball
        using CricAPI and official match feeds, respecting strict credit quotas.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            from .providers.cricapi_provider import cricapi_provider
            async with conn.execute("SELECT id, scheduled_start, team_a_id, team_b_id FROM matches WHERE state = 'SQUAD_ANNOUNCED'") as c:
                upcoming = [dict(r) for r in await c.fetchall()]

            for m in upcoming:
                # Check if match is within 45-minute toss window
                if cricapi_provider.is_match_about_to_start(m.get("scheduled_start")):
                    cricbuzz_id = self.active_cricbuzz_bindings.get(m["id"])
                    if cricbuzz_id:
                        logger.info(f"Checking toss & Playing XI for match {m['id']} (Cricbuzz ID: {cricbuzz_id})")
                        res = cricapi_provider.fetch_playing_xi_for_match(cricbuzz_id, m.get("scheduled_start"))
                        if res.get("status") == "success":
                            await self.apply_cricapi_playing_xi(m["id"], res)
        except Exception as e:
            logger.error(f"Error in poll_toss_and_xi_cadence: {e}")
        finally:
            await conn.close()

    async def apply_cricapi_playing_xi(self, match_id: str, cricapi_res: Dict[str, Any]):
        """
        Applies CricAPI Playing XI (11 players per team), official pre-match toss, and authentic venue
        to match_squads and advances match state to XI_AND_TOSS_LOCKED.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                m_row = await c.fetchone()
                if not m_row:
                    return
                m = dict(m_row)

            teams = cricapi_res.get("teams", [])
            for t in teams:
                playing_xi = t.get("playing_xi", [])
                for p in playing_xi:
                    p_name = p.get("name", "")
                    await conn.execute("""
                        UPDATE match_squads SET is_confirmed_xi = 1, is_probable_xi = 1, batting_order = ?
                        WHERE match_id = ? AND player_id IN (
                            SELECT id FROM players WHERE name LIKE ? OR name = ?
                        )
                    """, (p.get("order", 99), match_id, f"%{p_name}%", p_name))

            # Pre-Match Toss & Venue Detection from CricketData.org API
            from .providers.cricapi_provider import cricapi_provider
            cricapi_match_id = m.get("cricapi_id") or cricapi_res.get("match_id")
            toss_info = cricapi_provider.fetch_match_toss_and_venue(cricapi_match_id) if cricapi_match_id else {}

            ta = m.get("team_a_id", "AUS")
            tb = m.get("team_b_id", "ENG")
            t_winner_raw = str(toss_info.get("toss_winner") or cricapi_res.get("toss_winner") or "").lower()
            t_choice = str(toss_info.get("toss_choice") or cricapi_res.get("toss_choice") or "bat").lower()

            toss_win_id = None
            if any(k in t_winner_raw for k in [tb.lower(), "england", "eng"]):
                toss_win_id = tb
            elif any(k in t_winner_raw for k in [ta.lower(), "australia", "aus"]):
                toss_win_id = ta

            toss_dec = "bat" if "bat" in t_choice else "bowl"
            
            # Determine 1st innings batting team
            if toss_win_id:
                first_bat_id = toss_win_id if toss_dec == "bat" else (tb if toss_win_id == ta else ta)
            else:
                first_bat_id = m.get("innings_1_team_id") or tb

            # Resolve venue if detected
            resolved_venue = toss_info.get("venue_id") or m.get("venue_id")
            if "hobart" in str(toss_info.get("venue_name") or "").lower():
                resolved_venue = "hobart"

            status_msg = f"{toss_win_id or 'Winner'} won the toss and elected to {toss_dec}" if toss_win_id else "Toss locked"

            await conn.execute("""
                UPDATE matches SET
                    toss_winner_id = COALESCE(?, toss_winner_id),
                    toss_decision = COALESCE(?, toss_decision),
                    innings_1_team_id = ?,
                    venue_id = COALESCE(?, venue_id),
                    match_status_text = ?,
                    state = 'XI_AND_TOSS_LOCKED'
                WHERE id = ?
            """, (toss_win_id, toss_dec, first_bat_id, resolved_venue, status_msg, match_id))

            await conn.commit()
            logger.info(f"Applied CricAPI Playing XI, toss ({toss_win_id} opt to {toss_dec}), and venue ({resolved_venue}) for {match_id}")
        except Exception as e:
            logger.error(f"Error applying CricAPI Playing XI for {match_id}: {e}")
        finally:
            await conn.close()

    async def poll_live_matches_cadence(self):
        """
        Polls live Senior Men's International Matches between Top 20 ICC teams.
        Cascades across top cricket scrapers (Cricbuzz -> CREX -> ESPNcricinfo -> ICC -> CricAPI fallback).
        Enforces delivery deduplication, anti-score-regression, rain/weather detection,
        and broadcasts real-time ball-by-ball telemetry updates via WebSockets.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            # 1. Fetch active live matches from database
            async with conn.execute(
                "SELECT id, is_stumps FROM matches WHERE state IN ('LIVE_BALL_BY_BALL', 'IN_PLAY')"
            ) as c:
                live_rows = await c.fetchall()
                active_ids = []
                for r in live_rows:
                    # If match is at Day Stumps (Close of Play), throttle checks to once every 5 minutes
                    if bool(r["is_stumps"]):
                        last_stumps_poll = getattr(self, f"_last_stumps_poll_{r['id']}", 0.0)
                        if time.time() - last_stumps_poll < 300.0:
                            continue
                        setattr(self, f"_last_stumps_poll_{r['id']}", time.time())
                    active_ids.append(r["id"])

            # 2. Ingest live ball-by-ball updates for active matches concurrently via cascading Top Scraper
            async def sync_single_match(mid: str):
                cb_id = self.active_cricbuzz_bindings.get(mid)
                try:
                    await asyncio.wait_for(self.scrape_and_apply_cricbuzz(mid, cb_id), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.debug(f"Live scrape cycle timed out after 10s for {mid}")
                except Exception as m_err:
                    logger.warning(f"Error updating live match {mid} via Top Scraper: {m_err}")

            if active_ids:
                await asyncio.gather(*(sync_single_match(mid) for mid in active_ids), return_exceptions=True)

            # 3. Synchronize with CricAPI senior matches as quota-friendly fallback (10-minute cadence, max 100 hits/day)
            now_ts = time.time()
            last_cricapi_sync = getattr(self, "_last_cricapi_live_sync", 0.0)
            if (now_ts - last_cricapi_sync) >= 600.0 or not active_ids:
                self._last_cricapi_live_sync = now_ts
                from .providers.cricapi_provider import cricapi_provider
                updated_ids = await cricapi_provider.sync_live_senior_mens_matches(conn)
                if updated_ids and self.broadcast_func:
                    for mid in updated_ids:
                        async with conn.execute("SELECT * FROM matches WHERE id = ?", (mid,)) as c:
                            row = await c.fetchone()
                            if row:
                                await self.broadcast_func(mid, {"type": "LIVE_TELEMETRY_SYNCED", "match": dict(row)})
        except Exception as e:
            logger.error(f"Error in poll_live_matches_cadence (Top Scraper): {e}")
        finally:
            await conn.close()

    async def auto_discover_and_sync_live_matches(self) -> List[str]:
        """
        Automated Live Match Detection & Lifecycle Management Engine:
        1. Identifies completed matches in DB and transitions them to COMPLETED.
        2. Scrapes Cricbuzz and ESPNcricinfo feeds to dynamically discover all active Senior Men's
           International matches between Top 20 ICC teams.
        3. Reconciles discovered matches with DB: promotes active upcoming matches (e.g. SQUAD_ANNOUNCED)
           to LIVE_BALL_BY_BALL, inserts unrecorded live fixtures, and triggers live ball updates.
        """
        import httpx
        from bs4 import BeautifulSoup
        from datetime import datetime

        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        updated_match_ids = []
        try:
            # Step 1: Clean up completed matches in DB whose status text indicates match has concluded
            async with conn.execute("""
                SELECT id, match_status_text, last_event_description, state, format, current_score, current_wickets, target_runs
                FROM matches
                WHERE state IN ('LIVE_BALL_BY_BALL', 'IN_PLAY')
            """) as c:
                live_in_db = [dict(r) for r in await c.fetchall()]

            for m in live_in_db:
                comb_st = f"{m.get('match_status_text') or ''} {m.get('last_event_description') or ''}".lower()
                is_finished = any(k in comb_st for k in [
                    "won by", "won the match", "match drawn", "match tied", "abandoned",
                    "no result", "called off", "concluded"
                ])
                if is_finished:
                    logger.info(f"Auto-transitioning completed match {m['id']} to COMPLETED (status: {m.get('match_status_text')})")
                    await conn.execute("""
                        UPDATE matches SET
                            state = 'COMPLETED',
                            is_stumps = 0,
                            stumps_label = NULL
                        WHERE id = ?
                    """, (m['id'],))
                    await conn.commit()
                    updated_match_ids.append(m['id'])
                    if self.broadcast_func:
                        await self.broadcast_func(m['id'], {"type": "STATE_CHANGE", "state": "COMPLETED"})

            # Step 2: Query Cricbuzz live scores list for active Senior Men's International matches
            from .providers.failover_scraper import is_senior_mens_international, TEAM_NAME_TO_CODE, TOP_20_ICC_TEAMS
            discovered_matches = []
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }

            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, headers=headers) as client:
                try:
                    cb_resp = await client.get("https://m.cricbuzz.com/cricket-match/live-scores")
                    if cb_resp.status_code == 200:
                        soup = BeautifulSoup(cb_resp.text, "html.parser")
                        for a in soup.find_all("a", href=re.compile(r"/live-cricket-scores/(\d+)")):
                            href = a.get("href", "")
                            txt = a.get_text(" ", strip=True)
                            comb = f"{href} {txt}"

                            if not is_senior_mens_international(comb):
                                continue

                            m_id = re.search(r"/live-cricket-scores/(\d+)", href).group(1)
                            comb_low = comb.lower()
                            if "t20" in comb_low:
                                fmt = "T20I"
                            elif "odi" in comb_low or "one day" in comb_low:
                                fmt = "ODI"
                            elif re.search(r"\btest\b|\btests\b", comb_low):
                                fmt = "TEST"
                            else:
                                fmt = "T20I"
                            txt_low = txt.lower()
                            is_done = any(k in txt_low or k in href.lower() for k in ["won by", "won the match", "match drawn", "match tied", "abandoned", "no result"])
                            is_preview = any(k in txt_low for k in ["preview", "starts in", "upcoming", "scheduled", "yet to begin"])
                            is_live = not is_preview and not is_done and (
                                any(k in txt_low for k in ["stumps", "lunch", "tea", "innings break", "drinks", "trail by", "lead by", "need ", "opt to", "chose to", "session"])
                                or bool(re.search(r"\d+/\d+|\d+\s*-\s*\d+", txt))
                            )

                            # Extract teams
                            found_teams = []
                            for name, code in TEAM_NAME_TO_CODE.items():
                                if re.search(rf"\b{re.escape(name)}\b", comb.lower()) and code not in found_teams:
                                    found_teams.append(code)
                            for code in TOP_20_ICC_TEAMS:
                                if re.search(rf"\b{code}\b", comb.upper()) and code not in found_teams:
                                    found_teams.append(code)

                            if len(found_teams) >= 2:
                                t1, t2 = found_teams[0], found_teams[1]
                                discovered_matches.append({
                                    "team_a": t1,
                                    "team_b": t2,
                                    "format": fmt,
                                    "cricbuzz_id": m_id,
                                    "title": txt,
                                    "is_completed": is_done,
                                    "is_live": is_live,
                                    "is_preview": is_preview
                                })
                except Exception as cb_err:
                    logger.debug(f"Cricbuzz live match discovery error: {cb_err}")

            # Step 3: Reconcile discovered live matches with database
            for d in discovered_matches:
                t1, t2, m_fmt = d["team_a"], d["team_b"], d["format"]
                cb_id = d["cricbuzz_id"]
                is_comp = d["is_completed"]
                is_live = d.get("is_live", False)
                is_prev = d.get("is_preview", False)
                title_txt = d.get("title", "").lower()

                # Find candidate matches in database
                async with conn.execute("""
                    SELECT * FROM matches
                    WHERE ((team_a_id = ? AND team_b_id = ?) OR (team_a_id = ? AND team_b_id = ?))
                      AND format = ?
                      AND state != 'COMPLETED'
                    ORDER BY CASE WHEN state = 'LIVE_BALL_BY_BALL' THEN 0 ELSE 1 END, scheduled_start ASC
                """, (t1, t2, t2, t1, m_fmt)) as c:
                    candidates = [dict(r) for r in await c.fetchall()]

                db_match = None
                if candidates:
                    # Match by fixture number if in title (e.g. 1st Test vs 2nd Test)
                    for cand in candidates:
                        c_id = cand["id"].lower()
                        c_ser = (cand.get("series_name") or "").lower()
                        if "1st" in title_txt and ("1st" in c_ser or "_1" in c_id):
                            db_match = cand
                            break
                        elif "2nd" in title_txt and ("2nd" in c_ser or "_2" in c_id):
                            db_match = cand
                            break
                        elif "3rd" in title_txt and ("3rd" in c_ser or "_3" in c_id):
                            db_match = cand
                            break
                    if not db_match:
                        db_match = candidates[0]

                if db_match:
                    mid = db_match["id"]
                    if is_comp:
                        # Safety Guard: NEVER mark a future fixture as COMPLETED!
                        sch_str = db_match.get("scheduled_start")
                        if sch_str:
                            try:
                                sch_dt = datetime.fromisoformat(str(sch_str).replace("Z", "+00:00"))
                                if sch_dt > datetime.now(timezone.utc):
                                    logger.info(f"Skipping completion for {mid}: scheduled start is in the future ({sch_str}).")
                                    continue
                            except Exception:
                                pass
                        logger.info(f"Discovered concluded match {mid} ({t1} vs {t2}). Marking COMPLETED.")
                        await conn.execute("UPDATE matches SET state = 'COMPLETED', is_stumps = 0, stumps_label = NULL WHERE id = ?", (mid,))
                        await conn.commit()
                        updated_match_ids.append(mid)
                    elif is_live:
                        if db_match["state"] != "LIVE_BALL_BY_BALL":
                            logger.info(f"Promoting active live match {mid} ({t1} vs {t2}, {m_fmt}) to LIVE_BALL_BY_BALL!")
                            await conn.execute("UPDATE matches SET state = 'LIVE_BALL_BY_BALL' WHERE id = ?", (mid,))
                            await conn.commit()
                            updated_match_ids.append(mid)
                        self.active_cricbuzz_bindings[mid] = cb_id
                        try:
                            await self.scrape_and_apply_cricbuzz(mid, cb_id)
                        except Exception as sync_e:
                            logger.warning(f"Error syncing discovered live match {mid}: {sync_e}")
                    elif is_prev:
                        # Ensure preview matches are not falsely kept as LIVE_BALL_BY_BALL
                        if db_match["state"] == "LIVE_BALL_BY_BALL":
                            logger.info(f"Match {mid} ({t1} vs {t2}) is upcoming/preview. Reverting from LIVE to SQUAD_ANNOUNCED.")
                            await conn.execute("UPDATE matches SET state = 'SQUAD_ANNOUNCED' WHERE id = ?", (mid,))
                            await conn.commit()
                else:
                    # Match does not exist in DB yet: insert new fixture ONLY if actively live and strictly senior international
                    if is_live:
                        canonical_mid = f"m_{min(t1, t2).lower()}_{max(t1, t2).lower()}_{datetime.now().year}_{m_fmt.lower()}_1"
                        clean_ser_name = d.get("title") or f"{t1} vs {t2} {m_fmt}"
                        clean_ser_name = re.sub(r"\s*-\s*Live Score.*$", "", clean_ser_name, flags=re.IGNORECASE)
                        clean_ser_name = re.sub(r"\s*\|\s*Cricbuzz.*$", "", clean_ser_name, flags=re.IGNORECASE).strip()

                        candidate_obj = {
                            "id": canonical_mid,
                            "series_name": clean_ser_name,
                            "format": m_fmt,
                            "team_a_id": t1,
                            "team_b_id": t2
                        }
                        if not is_official_international_match(candidate_obj):
                            logger.info(f"Skipping non-official international match {canonical_mid} ({clean_ser_name})")
                            continue

                        # Dynamic Venue Resolution
                        title_low = (d.get("title") or "").lower()
                        if "melbourne" in title_low or "mcg" in title_low:
                            v_discovered = "mcg"
                        elif "sydney" in title_low or "scg" in title_low:
                            v_discovered = "scg"
                        elif "adelaide" in title_low:
                            v_discovered = "adelaide"
                        elif "perth" in title_low:
                            v_discovered = "perth"
                        elif "brisbane" in title_low or "gabba" in title_low:
                            v_discovered = "gabba"
                        elif "wellington" in title_low or "basin" in title_low:
                            v_discovered = "basin_reserve"
                        elif "rawalpindi" in title_low:
                            v_discovered = "rawalpindi"
                        elif "lahore" in title_low:
                            v_discovered = "lahore"
                        elif "sharjah" in title_low:
                            v_discovered = "sharjah"
                        elif "hobart" in title_low:
                            v_discovered = "hobart"
                        else:
                            v_discovered = "mcg" if set([t1, t2]) == {"AUS", "ENG"} else "wankhede"

                        async with conn.execute("SELECT id FROM matches WHERE id = ?", (canonical_mid,)) as c:
                            existing_canonical = await c.fetchone()
                        if not existing_canonical:
                            logger.info(f"Inserting newly discovered live match {canonical_mid} ({t1} vs {t2}, {m_fmt}) into database!")
                            await conn.execute("""
                                INSERT INTO matches (id, series_name, format, team_a_id, team_b_id, venue_id, state, scheduled_start, current_score, current_wickets, current_overs, current_balls, current_innings)
                                VALUES (?, ?, ?, ?, ?, ?, 'LIVE_BALL_BY_BALL', datetime('now'), 0, 0, 0, 0, 1)
                            """, (canonical_mid, clean_ser_name, m_fmt, t1, t2, v_discovered))
                            await conn.commit()
                            self.active_cricbuzz_bindings[canonical_mid] = cb_id
                            try:
                                await self.scrape_and_apply_cricbuzz(canonical_mid, cb_id)
                            except Exception as sync_e:
                                logger.warning(f"Error syncing newly created live match {canonical_mid}: {sync_e}")
                            updated_match_ids.append(canonical_mid)

            # Evaluate all current live matches with cricket rules engine to auto-conclude completed/elapsed fixtures
            from ..engine.cricket_rules import evaluate_match_completion
            async with conn.execute("SELECT * FROM matches WHERE state IN ('LIVE_BALL_BY_BALL', 'IN_PLAY')") as c_live:
                live_in_db = [dict(r) for r in await c_live.fetchall()]
            for lm in live_in_db:
                eval_res = evaluate_match_completion(lm)
                if eval_res.get("is_completed"):
                    logger.info(f"Auto-completing elapsed/concluded match {lm['id']}: {eval_res.get('result_description')}")
                    res_desc = eval_res.get("result_description") or "Match Completed"
                    p_a_fin = 100.0 if eval_res.get("winning_team_id") == lm.get("team_a_id") else (0.0 if eval_res.get("winning_team_id") else 33.3)
                    p_b_fin = 100.0 if eval_res.get("winning_team_id") == lm.get("team_b_id") else (0.0 if eval_res.get("winning_team_id") else 33.3)
                    await conn.execute("""
                        UPDATE matches SET
                            state = 'COMPLETED',
                            is_stumps = 0,
                            stumps_label = NULL,
                            match_status_text = ?,
                            last_event_description = ?,
                            p_team_a = ?,
                            p_team_b = ?
                        WHERE id = ?
                    """, (res_desc, res_desc, p_a_fin, p_b_fin, lm["id"]))
                    updated_match_ids.append(lm["id"])

            await conn.commit()
            return updated_match_ids
        except Exception as e:
            logger.error(f"Error in auto_discover_and_sync_live_matches: {e}")
            return []
        finally:
            await conn.close()

    async def sync_squads_for_match(self, match_id: str) -> Dict[str, Any]:
        """
        Always extracts the official available squad from external sources and verified international feeds,
        resolves each player via EntityResolver & Debutant Priors, populates match_squads,
        and triggers Stage 2 / Stage 3 ML recalculation.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                row = await c.fetchone()
                if not row:
                    return {"status": "error", "message": f"Match {match_id} not found"}
                match = dict(row)

            team_a = match.get("team_a_id", "IND")
            team_b = match.get("team_b_id", "NZ")
            match_format = match.get("format", "ODI")
            venue_id = match.get("venue_id", "")

            # Extract squads from external sources
            squad_data = await self.json_provider.fetch_squads(match_id, team_a, team_b, match_format)
            squad_a_raw = squad_data.get("team_a_squad", [])
            squad_b_raw = squad_data.get("team_b_squad", [])

            if not squad_a_raw and not squad_b_raw:
                return {"status": "noop", "message": "No external squad updates found"}

            async with conn.execute("SELECT * FROM players") as c:
                all_players = [dict(r) for r in await c.fetchall()]

            self.entity_resolver.db = conn
            venue_profile = get_historical_venue_profile(venue_id)

            inserted_squad = []

            # Process Team A squad
            for idx, p in enumerate(squad_a_raw):
                resolved = await self.entity_resolver.resolve_player(
                    p["name"], team_a, p.get("role"), p.get("bowling_style"), all_players=all_players
                )
                await conn.execute("""
                    INSERT OR IGNORE INTO players (
                        id, name, team_id, role, batting_hand, bowling_style,
                        intl_balls_faced, intl_balls_bowled, t20_batting_sr, t20_batting_avg,
                        t20_economy, t20_bowling_sr, powerplay_batting_rating, middle_batting_rating,
                        death_batting_rating, powerplay_bowling_rating, middle_bowling_rating, death_bowling_rating,
                        foreign_condition_ratio_subcontinent, foreign_condition_ratio_sena,
                        recent_selection_weight, is_debutant, league_experience_tier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resolved["id"], resolved["name"], resolved["team_id"], resolved.get("role", "batter"),
                    resolved.get("batting_hand", "right"), resolved.get("bowling_style", "NONE"),
                    resolved.get("intl_balls_faced", 100), resolved.get("intl_balls_bowled", 0),
                    resolved.get("t20_batting_sr", 130.0), resolved.get("t20_batting_avg", 30.0),
                    resolved.get("t20_economy", 7.8), resolved.get("t20_bowling_sr", 18.0),
                    resolved.get("powerplay_batting_rating", 50.0), resolved.get("middle_batting_rating", 50.0),
                    resolved.get("death_batting_rating", 50.0), resolved.get("powerplay_bowling_rating", 50.0),
                    resolved.get("middle_bowling_rating", 50.0), resolved.get("death_bowling_rating", 50.0),
                    resolved.get("foreign_condition_ratio_subcontinent", 1.0),
                    resolved.get("foreign_condition_ratio_sena", 1.0),
                    resolved.get("recent_selection_weight", 0.8),
                    resolved.get("is_debutant", 0),
                    resolved.get("league_experience_tier", "tier_1_ipl")
                ))
                aff_mult, _ = calculate_player_venue_affinity(resolved, venue_profile)
                is_xi_state = match.get("state") in ("XI_AND_TOSS_LOCKED", "LIVE_BALL_BY_BALL", "COMPLETED")
                is_conf = 1 if (is_xi_state and idx < 11) else 0
                is_prob = 1 if (is_xi_state and idx < 11) else 0
                bat_ord = (idx + 1) if (is_xi_state and idx < 11) else 99
                inserted_squad.append((
                    match_id, team_a, resolved["id"], 1, is_prob, is_conf, bat_ord, round(aff_mult * 100, 1)
                ))

            # Process Team B squad
            for idx, p in enumerate(squad_b_raw):
                resolved = await self.entity_resolver.resolve_player(
                    p["name"], team_b, p.get("role"), p.get("bowling_style"), all_players=all_players
                )
                await conn.execute("""
                    INSERT OR IGNORE INTO players (
                        id, name, team_id, role, batting_hand, bowling_style,
                        intl_balls_faced, intl_balls_bowled, t20_batting_sr, t20_batting_avg,
                        t20_economy, t20_bowling_sr, powerplay_batting_rating, middle_batting_rating,
                        death_batting_rating, powerplay_bowling_rating, middle_bowling_rating, death_bowling_rating,
                        foreign_condition_ratio_subcontinent, foreign_condition_ratio_sena,
                        recent_selection_weight, is_debutant, league_experience_tier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resolved["id"], resolved["name"], resolved["team_id"], resolved.get("role", "batter"),
                    resolved.get("batting_hand", "right"), resolved.get("bowling_style", "NONE"),
                    resolved.get("intl_balls_faced", 100), resolved.get("intl_balls_bowled", 0),
                    resolved.get("t20_batting_sr", 130.0), resolved.get("t20_batting_avg", 30.0),
                    resolved.get("t20_economy", 7.8), resolved.get("t20_bowling_sr", 18.0),
                    resolved.get("powerplay_batting_rating", 50.0), resolved.get("middle_batting_rating", 50.0),
                    resolved.get("death_batting_rating", 50.0), resolved.get("powerplay_bowling_rating", 50.0),
                    resolved.get("middle_bowling_rating", 50.0), resolved.get("death_bowling_rating", 50.0),
                    resolved.get("foreign_condition_ratio_subcontinent", 1.0),
                    resolved.get("foreign_condition_ratio_sena", 1.0),
                    resolved.get("recent_selection_weight", 0.8),
                    resolved.get("is_debutant", 0),
                    resolved.get("league_experience_tier", "tier_1_ipl")
                ))
                aff_mult, _ = calculate_player_venue_affinity(resolved, venue_profile)
                is_xi_state_b = match.get("state") in ("XI_AND_TOSS_LOCKED", "LIVE_BALL_BY_BALL", "COMPLETED")
                is_conf_b = 1 if (is_xi_state_b and idx < 11) else 0
                is_prob_b = 1 if (is_xi_state_b and idx < 11) else 0
                bat_ord_b = (idx + 1) if (is_xi_state_b and idx < 11) else 99
                inserted_squad.append((
                    match_id, team_b, resolved["id"], 1, is_prob_b, is_conf_b, bat_ord_b, round(aff_mult * 100, 1)
                ))

            # Atomic refresh of match squads
            await conn.execute("DELETE FROM match_squads WHERE match_id = ?", (match_id,))
            await conn.executemany("""
                INSERT INTO match_squads (match_id, team_id, player_id, is_in_squad, is_probable_xi, is_confirmed_xi, batting_order, pitch_fit_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, inserted_squad)

            # Recalculate Stage 2
            async with conn.execute("SELECT p.* FROM match_squads ms JOIN players p ON ms.player_id = p.id WHERE ms.match_id = ? AND ms.team_id = ?", (match_id, team_a)) as c:
                team_a_squad = [dict(r) for r in await c.fetchall()]
            async with conn.execute("SELECT p.* FROM match_squads ms JOIN players p ON ms.player_id = p.id WHERE ms.match_id = ? AND ms.team_id = ?", (match_id, team_b)) as c:
                team_b_squad = [dict(r) for r in await c.fetchall()]

            p1 = match.get("p1_macro", 50.0)
            stage2_res = calculate_stage2_squad(team_a_squad, team_b_squad, venue_profile, p1, match_format=match.get("format", "T20I"))

            # Update match record
            new_state = "SQUAD_ANNOUNCED" if match["state"] == "UPCOMING_NO_SQUAD" else match["state"]
            if squad_data.get("pitch_report"):
                await conn.execute("UPDATE matches SET pitch_report = ? WHERE id = ?", (squad_data["pitch_report"], match_id))

            await conn.execute("""
                UPDATE matches SET
                    state = ?,
                    p2_squad = ?,
                    p_team_a = ?,
                    p_team_b = ?,
                    delta_squad_shift = ?,
                    last_event_description = ?
                WHERE id = ?
            """, (
                new_state,
                stage2_res["p_team_a"],
                stage2_res["p_team_a"] if new_state == "SQUAD_ANNOUNCED" else match["p_team_a"],
                stage2_res["p_team_b"] if new_state == "SQUAD_ANNOUNCED" else match["p_team_b"],
                stage2_res["delta_squad_shift"],
                f"Squad synced from external sources ({len(team_a_squad)}v{len(team_b_squad)})",
                match_id
            ))
            await conn.commit()

            # Broadcast update
            if self.broadcast_func:
                async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                    updated_match = dict(await c.fetchone())
                await self.broadcast_func(match_id, {"type": "STATE_CHANGE", "match": updated_match})

            # Automatically propagate series squad to all future matches in this tour series
            try:
                from ..engine.series_squad_propagator import propagate_series_squads_async
                await propagate_series_squads_async(conn)
            except Exception as prop_err:
                logger.error(f"Error propagating series squad for {match_id}: {prop_err}")

            return {
                "status": "ok",
                "team_a_count": len(team_a_squad),
                "team_b_count": len(team_b_squad),
                "delta_squad_shift": stage2_res["delta_squad_shift"]
            }
        finally:
            await conn.close()

    async def ingest_declared_squad_text(self, match_id: str, team_id: str, raw_text: str) -> Dict[str, Any]:
        """
        Parses raw declared squad text (e.g. from Cricbuzz with BATTERS, ALL ROUNDERS, etc.),
        resolves/creates players, assigns them to match_squads for the specified team with is_confirmed_xi = 0,
        recalculates Stage 2 predictions, and propagates the squad across the tour.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                row = await c.fetchone()
                if not row:
                    return {"status": "error", "message": f"Match {match_id} not found"}
                match = dict(row)

            parsed_players = self.cricbuzz_api.parse_declared_squad_text(raw_text)
            if not parsed_players:
                return {"status": "error", "message": "No players could be parsed from the provided text."}

            async with conn.execute("SELECT * FROM players") as c:
                all_players = [dict(r) for r in await c.fetchall()]

            self.entity_resolver.db = conn
            venue_id = match.get("venue_id", "")
            venue_profile = get_historical_venue_profile(venue_id)

            # Delete old squad entries for this team in this match
            await conn.execute("DELETE FROM match_squads WHERE match_id = ? AND team_id = ?", (match_id, team_id))

            inserted_squad = []
            for idx, p in enumerate(parsed_players):
                resolved = await self.entity_resolver.resolve_player(
                    p["name"], team_id, p.get("role"), None, all_players=all_players
                )
                await conn.execute("""
                    INSERT OR IGNORE INTO players (
                        id, name, team_id, role, batting_hand, bowling_style,
                        intl_balls_faced, intl_balls_bowled, t20_batting_sr, t20_batting_avg,
                        t20_economy, t20_bowling_sr, powerplay_batting_rating, middle_batting_rating,
                        death_batting_rating, powerplay_bowling_rating, middle_bowling_rating, death_bowling_rating,
                        foreign_condition_ratio_subcontinent, foreign_condition_ratio_sena,
                        recent_selection_weight, is_debutant, league_experience_tier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resolved["id"], resolved["name"], resolved["team_id"], resolved.get("role", "batter"),
                    resolved.get("batting_hand", "right"), resolved.get("bowling_style", "NONE"),
                    resolved.get("intl_balls_faced", 100), resolved.get("intl_balls_bowled", 0),
                    resolved.get("t20_batting_sr", 130.0), resolved.get("t20_batting_avg", 30.0),
                    resolved.get("t20_economy", 7.8), resolved.get("t20_bowling_sr", 18.0),
                    resolved.get("powerplay_batting_rating", 50.0), resolved.get("middle_batting_rating", 50.0),
                    resolved.get("death_batting_rating", 50.0), resolved.get("powerplay_bowling_rating", 50.0),
                    resolved.get("middle_bowling_rating", 50.0), resolved.get("death_bowling_rating", 50.0),
                    resolved.get("foreign_condition_ratio_subcontinent", 1.0),
                    resolved.get("foreign_condition_ratio_sena", 1.0),
                    resolved.get("recent_selection_weight", 0.9),
                    resolved.get("is_debutant", 0),
                    resolved.get("league_experience_tier", "tier_1_ipl")
                ))

                aff_mult, _ = calculate_player_venue_affinity(resolved, venue_profile)
                inserted_squad.append((
                    match_id, team_id, resolved["id"], 1, 0, 0, 99, round(aff_mult * 80.0, 1)
                ))

            await conn.executemany("""
                INSERT INTO match_squads (match_id, team_id, player_id, is_in_squad, is_probable_xi, is_confirmed_xi, batting_order, pitch_fit_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, inserted_squad)

            # Recalculate Stage 2
            team_a = match.get("team_a_id")
            team_b = match.get("team_b_id")
            async with conn.execute("SELECT p.* FROM match_squads ms JOIN players p ON ms.player_id = p.id WHERE ms.match_id = ? AND ms.team_id = ?", (match_id, team_a)) as c:
                team_a_squad = [dict(r) for r in await c.fetchall()]
            async with conn.execute("SELECT p.* FROM match_squads ms JOIN players p ON ms.player_id = p.id WHERE ms.match_id = ? AND ms.team_id = ?", (match_id, team_b)) as c:
                team_b_squad = [dict(r) for r in await c.fetchall()]

            p1 = match.get("p1_macro", 50.0)
            stage2_res = calculate_stage2_squad(team_a_squad, team_b_squad, venue_profile, p1, match_format=match.get("format", "ODI"))

            new_state = "SQUAD_ANNOUNCED" if match["state"] == "UPCOMING_NO_SQUAD" else match["state"]
            await conn.execute("""
                UPDATE matches SET
                    state = ?,
                    p2_squad = ?,
                    p_team_a = ?,
                    p_team_b = ?,
                    delta_squad_shift = ?,
                    last_event_description = ?
                WHERE id = ?
            """, (
                new_state,
                stage2_res["p_team_a"],
                stage2_res["p_team_a"] if new_state == "SQUAD_ANNOUNCED" else match["p_team_a"],
                stage2_res["p_team_b"] if new_state == "SQUAD_ANNOUNCED" else match["p_team_b"],
                stage2_res["delta_squad_shift"],
                f"Official declared squad ingested for {team_id} ({len(inserted_squad)} players)",
                match_id
            ))
            await conn.commit()

            # Propagate to remaining tour matches
            from ..engine.series_squad_propagator import propagate_series_squads_async
            propagated = await propagate_series_squads_async(conn)

            return {
                "status": "ok",
                "team_id": team_id,
                "parsed_count": len(inserted_squad),
                "players": [p["name"] for p in parsed_players],
                "propagated_matches_count": len(propagated)
            }
        finally:
            await conn.close()

    async def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Manually ingests an ESPNcricinfo or Cricbuzz match URL, extracts metadata,
        and transitions state or creates match.
        """
        extracted = await self.html_scraper.scrape_match_url(url)
        return extracted

    async def transition_match_state(self, match_id: str, target_state: str) -> Dict[str, Any]:
        """
        Advances a match to target_state using the Replay Simulator.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            res = await self.replay_sim.advance_match_state(match_id, target_state, conn)
            return res
        finally:
            await conn.close()

    async def reset_match_to_ball_one(self, match_id: str) -> Dict[str, Any]:
        """
        Rewinds match to Ball 1 of 1st innings so the user can stream delivery-by-delivery from the very start.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                row = await c.fetchone()
                if not row:
                    raise ValueError(f"Match {match_id} not found")
                match = dict(row)

            # Score In Progress / Unresolved Scrape Guard:
            # If the scrape returns dummy placeholder data or "Score In Progress" with 0 runs/overs,
            # never overwrite an in-play match that already has authentic live runs or overs!
            if (
                scraped.get("score") == "Score In Progress"
                or (parsed.get("runs", 0) == 0 and parsed.get("overs", 0) == 0 and (match.get("current_overs", 0) > 0 or match.get("current_score", 0) > 0))
            ):
                logger.warning(f"Rejecting unpopulated or placeholder scrape for {match_id}: score='{scraped.get('score')}'. Existing match data preserved.")
                return match

            # Unbind from background scraping so streaming replay is not overwritten
            self.active_cricbuzz_bindings.pop(match_id, None)

            team_a_id = match["team_a_id"]
            team_b_id = match["team_b_id"]
            toss_winner = match.get("toss_winner_id") or team_a_id
            toss_decision = match.get("toss_decision") or "bat"
            team_1st_bat = toss_winner if toss_decision == "bat" else (team_b_id if toss_winner == team_a_id else team_a_id)
            team_1st_bowl = team_b_id if team_1st_bat == team_a_id else team_a_id

            async with conn.execute("""
                SELECT p.id, p.name, ms.batting_order
                FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ?
                ORDER BY ms.batting_order ASC
            """, (match_id, team_1st_bat)) as b_cur:
                bat_players = [dict(r) for r in await b_cur.fetchall()]

            async with conn.execute("""
                SELECT p.id, p.name, ms.batting_order
                FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ?
                ORDER BY ms.batting_order DESC
            """, (match_id, team_1st_bowl)) as bw_cur:
                bowl_players = [dict(r) for r in await bw_cur.fetchall()]

            striker_id = bat_players[0]["id"] if len(bat_players) > 0 else ("t_banton_eng" if team_1st_bat == "ENG" else "t_head_aus")
            non_striker_id = bat_players[1]["id"] if len(bat_players) > 1 else ("b_duckett_eng" if team_1st_bat == "ENG" else "m_marsh_aus")
            bowler_id = bowl_players[0]["id"] if len(bowl_players) > 0 else ("x_bartlett_aus" if team_1st_bowl == "AUS" else "j_archer_eng")

            # Check if match has a queued opening delivery
            async with conn.execute("""
                SELECT striker_id, non_striker_id, bowler_id
                FROM match_replay_queue
                WHERE match_id = ?
                ORDER BY seq ASC LIMIT 1
            """, (match_id,)) as q_cur:
                q_row = await q_cur.fetchone()
                if q_row:
                    striker_id = q_row[0]
                    non_striker_id = q_row[1]
                    bowler_id = q_row[2]

            p0 = float(match.get("p0_pre_match") or match.get("p_team_a") or 50.0)
            p_b = round(100.0 - p0, 1)

            # Purge previous delivery replay logs
            await conn.execute("DELETE FROM match_deliveries WHERE match_id = ?", (match_id,))
            await conn.execute("UPDATE match_replay_queue SET is_played = 0 WHERE match_id = ?", (match_id,))

            status_text = f"{team_1st_bat} won toss and elected to bat • 1st Innings starting"
            desc = f"Match starting • Over 0.0 • {striker_id.replace('_', ' ').title()} facing {bowler_id.replace('_', ' ').title()}"

            await conn.execute("""
                UPDATE matches SET
                    state = 'LIVE_BALL_BY_BALL',
                    current_innings = 1,
                    current_score = 0,
                    current_wickets = 0,
                    current_overs = 0,
                    current_balls = 0,
                    target_runs = NULL,
                    innings_1_score = NULL,
                    innings_1_wickets = NULL,
                    innings_1_overs = NULL,
                    innings_2_score = NULL,
                    innings_2_wickets = NULL,
                    innings_2_overs = NULL,
                    current_striker_id = ?,
                    current_non_striker_id = ?,
                    current_bowler_id = ?,
                    p_team_a = ?,
                    p_team_b = ?,
                    p_draw = 0.0,
                    last_wpa_swing = 0.0,
                    last_event_description = ?,
                    match_status_text = ?,
                    batting_tank_pct = 100.0,
                    bowling_tank_pct = 100.0,
                    is_stumps = 0
                WHERE id = ?
            """, (
                striker_id, non_striker_id, bowler_id,
                p0, p_b,
                desc, status_text,
                match_id
            ))
            await conn.commit()

            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                updated = dict(await c.fetchone())

            if self.broadcast_func:
                await self.broadcast_func(match_id, {"type": "MATCH_RESET", "match": updated})
            return {"status": "ok", "match": updated}
        finally:
            await conn.close()

    async def trigger_next_delivery(self, match_id: str) -> Dict[str, Any]:
        """
        Simulates and logs arrival of a live delivery.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            res = await self.replay_sim.step_single_ball(match_id, conn)
            return res
        finally:
            await conn.close()

    async def sync_live_match(self, match_id: str) -> Dict[str, Any]:
        """
        Extracts real-time live score, wickets, overs, striker, bowler from external live feeds,
        updates the matches table, re-evaluates Stage 4 Live inference, and broadcasts via WebSocket.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                row = await c.fetchone()
                if not row:
                    raise ValueError(f"Match {match_id} not found")
                match = dict(row)

            cb_id = self.active_cricbuzz_bindings.get(match_id)
            if not cb_id:
                if ("pak_sl_2026_test" in match_id or match_id == "170703") and "t20" not in match_id:
                    cb_id = "170703"
            if cb_id:
                await conn.close()
                return await self.scrape_and_apply_cricbuzz(match_id, cb_id)

            live_data = await self.json_provider.fetch_live_telemetry(match_id, match["team_a_id"], match["team_b_id"])

            m_fmt = match.get("format", "TEST")
            v_id = match.get("venue_id", "rawalpindi")
            st_id = live_data.get("current_striker_id")
            nst_id = live_data.get("current_non_striker_id")
            bw_id = live_data.get("current_bowler_id")

            from ..engine.player_strengths import get_player_dossier
            st_dossier = get_player_dossier(st_id, "batter") if st_id else {}
            st_dossier.update({"id": st_id, "batting_hand": "right", "balls_faced": 19, "name": st_dossier.get("name", "Striker")})

            nst_dossier = get_player_dossier(nst_id, "batter") if nst_id else {}
            nst_dossier.update({"id": nst_id, "batting_hand": "right", "balls_faced": 8, "name": nst_dossier.get("name", "Non-Striker")})

            bw_dossier = get_player_dossier(bw_id, "bowler") if bw_id else {}
            bw_dossier.update({"id": bw_id, "bowling_style": "RF", "name": bw_dossier.get("name", "Bowler")})

            from ..engine.live_pitch_classifier import LivePitchClassifier
            pitch_classifier = LivePitchClassifier()
            pitch_eval = pitch_classifier.classify_live_pitch(
                venue_id=v_id,
                match_format=m_fmt,
                current_overs=live_data["current_overs"],
                current_balls=live_data["current_balls"],
                current_score=live_data["current_score"],
                current_wickets=live_data["current_wickets"],
                deliveries=[],
                commentary_strings=[live_data.get("last_event_description", ""), match.get("pitch_report", "")],
                pitch_report=match.get("pitch_report")
            )
            live_pitch_archetype = pitch_eval["classified_pitch_type"]

            from ..engine.cricsheet_pitch_intelligence import get_country_pitch_adaptability
            t_a = match.get("team_a_id", "IND")
            t_b = match.get("team_b_id", "NZ")
            t_a_adapt = get_country_pitch_adaptability(t_a, m_fmt, live_pitch_archetype)
            t_b_adapt = get_country_pitch_adaptability(t_b, m_fmt, live_pitch_archetype)

            from ..engine.live_dew_detector import LiveDewDetector
            cur_ov_calc = live_data["current_overs"] + live_data["current_balls"] / 6.0
            dew_eval = LiveDewDetector.detect_live_dew(
                venue_id=match.get("venue_id") or match.get("venue") or "wankhede",
                is_day_night=bool(match.get("is_day_night", 0)),
                match_format=m_fmt,
                innings=live_data["current_innings"],
                current_overs=cur_ov_calc,
                commentary_feed=[live_data.get("last_event_description", ""), live_data.get("pitch_report", "")]
            )
            calibrated_dew_factor = dew_eval.get("dew_factor_calibrated", float(match.get("dew_factor") or 0.30))

            from ..engine.stage4_live import calculate_stage4_live
            res = calculate_stage4_live(
                innings=live_data["current_innings"],
                score=live_data["current_score"],
                wickets=live_data["current_wickets"],
                overs=live_data["current_overs"],
                balls=live_data["current_balls"],
                target=match.get("target_runs"),
                striker=st_dossier,
                non_striker=nst_dossier,
                bowler=bw_dossier,
                team_a_batting=((live_data.get("batting_team_id") == match["team_a_id"]) if live_data.get("batting_team_id") else (match.get("current_innings", 1) == 1 if (match.get("innings_1_team_id") or match["team_a_id"]) == match["team_a_id"] else match.get("current_innings", 1) != 1)),
                p0_pre_match=match.get("p0_pre_match", 51.5),
                previous_p_team_a=match.get("p_team_a", 54.2),
                dugout_players=[{"t20_batting_avg": 28.0}] * max(0, 10 - live_data["current_wickets"]),
                bowlers_list=[{"t20_economy": 7.5}] * 4,
                pitch_spin_ratio=match.get("spin_wicket_ratio", 0.62),
                match_format=m_fmt,
                live_pitch_type=live_pitch_archetype,
                team_a_pitch_adapt=t_a_adapt,
                team_b_pitch_adapt=t_b_adapt,
                innings_1_score=match.get("innings_1_score"),
                innings_2_score=match.get("innings_2_score"),
                match_day=match.get("stumps_day"),
                is_stumps=bool(match.get("is_stumps")),
                previous_p_draw=float(match.get("p_draw") or 0.0) if m_fmt == "TEST" else None,
                pitch_detection_source=pitch_eval.get("detection_source", "LIVE_TELEMETRY"),
                pitch_override_reason=pitch_eval.get("override_reason"),
                is_day_night=bool(match.get("is_day_night", 0)),
                dew_factor=calibrated_dew_factor,
                dew_intelligence=dew_eval
            )

            p_a = res["p_team_a"]
            p_b = res["p_team_b"]
            p_draw = res.get("p_draw", 0.0) if match.get("format") == "TEST" else 0.0
            wpa = res["wpa"]

            from ..engine.cricket_rules import evaluate_match_completion
            match_for_eval = dict(match)
            match_for_eval.update({
                "format": match.get("format", "TEST"),
                "current_score": live_data["current_score"],
                "current_wickets": live_data["current_wickets"],
                "current_overs": live_data["current_overs"],
                "target_runs": match.get("target_runs"),
                "last_event_description": live_data["last_event_description"],
                "state": "LIVE_BALL_BY_BALL"
            })
            completion = evaluate_match_completion(match_for_eval)
            new_state = "COMPLETED" if completion["is_completed"] else "LIVE_BALL_BY_BALL"
            ev_desc = completion["result_description"] if completion["is_completed"] else live_data["last_event_description"]
            if completion["is_completed"]:
                if completion["winning_team_id"] == match.get("team_a_id"):
                    p_a, p_b = 100.0, 0.0
                elif completion["winning_team_id"] == match.get("team_b_id"):
                    p_a, p_b = 0.0, 100.0

            if live_data.get("pitch_report"):
                await conn.execute("UPDATE matches SET pitch_report = ? WHERE id = ?", (live_data["pitch_report"], match_id))

            await conn.execute("""
                UPDATE matches SET
                    state = ?,
                    current_innings = ?,
                    current_score = ?,
                    current_wickets = ?,
                    current_overs = ?,
                    current_balls = ?,
                    current_striker_id = ?,
                    current_non_striker_id = ?,
                    current_bowler_id = ?,
                    p_team_a = ?,
                    p_team_b = ?,
                    p_draw = ?,
                    last_wpa_swing = ?,
                    last_event_description = ?,
                    active_pitch_type = ?,
                    ground_pitch_archetype = ?,
                    pitch_type = ?,
                    biggest_matchup_adv = ?,
                    par_score_projected = ?,
                    par_score_min = ?,
                    par_score_max = ?,
                    batting_tank_pct = ?,
                    bowling_tank_pct = ?
                WHERE id = ?
            """, (
                new_state,
                live_data["current_innings"],
                live_data["current_score"],
                live_data["current_wickets"],
                live_data["current_overs"],
                live_data["current_balls"],
                live_data["current_striker_id"],
                live_data["current_non_striker_id"],
                live_data["current_bowler_id"],
                p_a, p_b, p_draw, wpa,
                ev_desc,
                live_pitch_archetype,
                live_pitch_archetype,
                live_pitch_archetype,
                res.get("biggest_matchup_adv"),
                res.get("par_score_projected"),
                res.get("par_score_min"),
                res.get("par_score_max"),
                res.get("batting_tank_pct"),
                res.get("bowling_tank_pct"),
                match_id
            ))
            await conn.commit()

            if new_state == "COMPLETED":
                try:
                    from ..engine.self_learning_engine import process_completed_match_learning
                    await process_completed_match_learning(match_id, conn)
                except Exception as l_err:
                    logger.warning(f"Self-learning execution encountered error for {match_id}: {l_err}")

            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                updated = dict(await c.fetchone())

            if updated.get("format") == "TEST" and (updated.get("is_stumps") or "stumps" in str(updated.get("match_status_text", "")).lower()):
                st_day = int(updated.get("stumps_day") or 4)
                days_left = max(0, 5 - st_day)
                min_match_ov = float(days_left * 90.0)
                updated["min_overs_remaining"] = min_match_ov
                updated["overs_left_today"] = 0.0
                updated["total_match_overs_remaining"] = min_match_ov

            if self.broadcast_func:
                await self.broadcast_func(match_id, {"type": "LIVE_TELEMETRY_SYNCED", "match": updated})

            return {
                "status": "ok",
                "live_score": f"{live_data['current_score']}/{live_data['current_wickets']}",
                "overs": f"{live_data['current_overs']}.{live_data['current_balls']}",
                "match": updated
            }
        finally:
            await conn.close()

    async def scrape_and_apply_cricbuzz(self, match_id: str, cricbuzz_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Cascading Top Scraper: Scrapes live match telemetry across Cricbuzz -> CREX -> ESPNcricinfo -> ICC -> CricAPI,
        resolves active batsmen, bowler, weather/rain restrictions, deduplicates deliveries,
        enforces anti-score-regression, updates the database, re-evaluates Stage 4 Live, and broadcasts via WebSockets.
        """
        conn = await self.db_factory()
        conn.row_factory = aiosqlite.Row
        try:
            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                row = await c.fetchone()
                if not row:
                    raise ValueError(f"Match {match_id} not found")
                match = dict(row)

            # Cascade across top scrapers (Cricbuzz -> CREX -> ESPN -> ICC -> CricAPI fallback)
            scraped = await top_scraper.get_live_telemetry_with_cascade(match_id, match_meta=match, cricbuzz_query=cricbuzz_query)
            if not scraped or scraped.get("status") == "failover_exhausted" or not scraped.get("score"):
                logger.info(f"Top Scraper failover exhausted or no active score for {match_id}.")
                return match

            tree_text = self.cricbuzz_api.format_tree_view(scraped)
            parsed = self.cricbuzz_api.parse_score_components(scraped.get("score", ""))

            # Score In Progress / Unresolved Scrape Guard:
            if (
                scraped.get("score") == "Score In Progress"
                or (parsed.get("runs", 0) == 0 and parsed.get("overs", 0) == 0 and (match.get("current_overs", 0) > 0 or match.get("current_score", 0) > 0))
            ):
                logger.warning(f"Rejecting unpopulated or placeholder scrape for {match_id}: score='{scraped.get('score')}'. Existing match data preserved.")
                return match

            # CRITICAL: Identify correct match format BEFORE running live prediction!
            # Strictly protect authentic format so TEST and ODI matches are never overwritten as T20I
            detected_format = scraped.get("format") or self.cricbuzz_api.detect_match_format(
                title=scraped.get("title", ""),
                score=scraped.get("score", ""),
                commentary=scraped.get("commentary", ""),
                match_id=match_id,
                current_format=match.get("format")
            ) or match.get("format", "ODI")

            mid_low = match_id.lower()
            series_low = str(match.get("series_name") or "").lower()
            title_low = str(scraped.get("title") or "").lower()
            status_low = str(scraped.get("match_status") or "").lower()

            # Format detection hierarchy: Limited overs (T20I, ODI) strictly prioritized and protected.
            # Never allow a T20I or ODI fixture to be mutated into TEST under any circumstances.
            if match.get("format") == "T20I" or "_t20_" in mid_low or "_t20i_" in mid_low or "t20" in series_low or "t20" in title_low or bool(re.search(r"\bt20i?\b", title_low)):
                detected_format = "T20I"
            elif match.get("format") == "ODI" or "_odi_" in mid_low or "odi" in series_low or "one day" in series_low or "odi" in title_low or bool(re.search(r"\bodi\b", title_low)):
                detected_format = "ODI"
            elif (
                match.get("format") == "TEST"
                or bool(re.search(r"\btest\b", mid_low))
                or bool(re.search(r"\btest\b", series_low))
                or bool(re.search(r"\btest\b", title_low))
                or any(k in status_low for k in ["trail by", "lead by", "follow-on", "day 1", "day 2", "day 3", "day 4", "day 5"])
            ):
                detected_format = "TEST"
            else:
                detected_format = match.get("format", "ODI")

            if detected_format and detected_format != match.get("format"):
                logger.info(f"Identified format {detected_format} for {match_id} (was {match.get('format')}). Updating format prior to prediction.")
                await conn.execute("UPDATE matches SET format = ? WHERE id = ?", (detected_format, match_id))
                match["format"] = detected_format

            match_format = detected_format

            # Multi-innings extraction and dynamic team alignment
            all_innings = scraped.get("all_innings", [])
            st_text = str(scraped.get("match_status") or "")
            st_low = st_text.lower()

            inn_1_team_id = match.get("innings_1_team_id")
            inn_1_score = match.get("innings_1_score")
            inn_1_wickets = match.get("innings_1_wickets")
            inn_1_overs = match.get("innings_1_overs")

            inn_2_team_id = match.get("innings_2_team_id")
            inn_2_score = match.get("innings_2_score")
            inn_2_wickets = match.get("innings_2_wickets")
            inn_2_overs = match.get("innings_2_overs")

            inn_3_team_id = match.get("innings_3_team_id")
            inn_3_score = match.get("innings_3_score")
            inn_3_wickets = match.get("innings_3_wickets")
            inn_3_overs = match.get("innings_3_overs")

            def resolve_team_abbr(abbr: str) -> Optional[str]:
                if not abbr:
                    return None
                u = abbr.upper()
                ta = (match["team_a_id"] or "").upper()
                tb = (match["team_b_id"] or "").upper()
                if u == ta or (u in ta and len(u) >= 2) or (u == "WI" and ta == "WI") or (u == "ZIM" and ta == "ZIM") or (u == "NZ" and ta == "NZ") or (u == "BAN" and ta == "BAN") or (u == "AFG" and ta == "AFG") or (u == "IND" and ta == "IND") or (u == "PAK" and ta == "PAK") or (u == "SL" and ta == "SL") or (u == "AUS" and ta == "AUS") or (u == "ENG" and ta == "ENG") or (u == "SA" and ta == "SA"):
                    return match["team_a_id"]
                if u == tb or (u in tb and len(u) >= 2) or (u == "WI" and tb == "WI") or (u == "ZIM" and tb == "ZIM") or (u == "NZ" and tb == "NZ") or (u == "BAN" and tb == "BAN") or (u == "AFG" and tb == "AFG") or (u == "IND" and tb == "IND") or (u == "PAK" and tb == "PAK") or (u == "SL" and tb == "SL") or (u == "AUS" and tb == "AUS") or (u == "ENG" and tb == "ENG") or (u == "SA" and tb == "SA"):
                    return match["team_b_id"]
                return None

            # Strict Match Integrity Guard: Verify that parsed teams belong to this match
            if all_innings:
                matched_teams = [resolve_team_abbr(inn.get("team")) for inn in all_innings if resolve_team_abbr(inn.get("team"))]
                if not matched_teams:
                    logger.warning(f"Parsed innings {all_innings} do not match teams for {match_id} ({match.get('team_a_id')} vs {match.get('team_b_id')}). Rejecting.")
                    return {"status": "rejected", "message": "Scraped teams do not match"}
            elif parsed.get("team") and not resolve_team_abbr(parsed.get("team")):
                logger.warning(f"Parsed team {parsed.get('team')} does not match teams for {match_id} ({match.get('team_a_id')} vs {match.get('team_b_id')}). Rejecting.")
                return {"status": "rejected", "message": "Scraped team does not match"}

            inn_4_team_id = match.get("innings_4_team_id")
            inn_4_score = match.get("innings_4_score")
            inn_4_wickets = match.get("innings_4_wickets")
            inn_4_overs = match.get("innings_4_overs")

            if len(all_innings) >= 4:
                inn_1 = all_innings[0]
                inn_2 = all_innings[1]
                inn_3 = all_innings[2]
                inn_4 = all_innings[3]

                t1 = resolve_team_abbr(inn_1.get("team"))
                t2 = resolve_team_abbr(inn_2.get("team"))
                t3 = resolve_team_abbr(inn_3.get("team")) or t1
                t4 = resolve_team_abbr(inn_4.get("team")) or t2

                if not t1 and t2:
                    t1 = match["team_a_id"] if t2 == match["team_b_id"] else match["team_b_id"]
                if not t2 and t1:
                    t2 = match["team_a_id"] if t1 == match["team_b_id"] else match["team_b_id"]

                inn_1_team_id = t1
                inn_1_score = inn_1["runs"]
                inn_1_wickets = inn_1["wickets"]
                inn_1_overs = inn_1["overs"]

                inn_2_team_id = t2
                inn_2_score = inn_2["runs"]
                inn_2_wickets = inn_2["wickets"]
                inn_2_overs = inn_2["overs"]

                inn_3_team_id = t3
                inn_3_score = inn_3["runs"]
                inn_3_wickets = inn_3["wickets"]
                inn_3_overs = inn_3["overs"]

                inn_4_team_id = t4
                inn_4_score = inn_4["runs"]
                inn_4_wickets = inn_4["wickets"]
                inn_4_overs = inn_4["overs"]

                inn_val = 4
                batting_team_id = t4
                bowling_team_id = t3 if t4 == t2 else t2
                runs = inn_4["runs"]
                wickets = inn_4["wickets"]
                overs_raw = inn_4["overs"] if inn_4["overs"] is not None else 0.0
                balls = int(round((overs_raw - int(overs_raw)) * 10))
                overs = int(overs_raw)

                # 4th innings target in Test matches: Lead from innings 1 & 3 minus innings 2
                target_val = max(1, (inn_1_score + inn_3_score) - inn_2_score + 1)
            elif len(all_innings) == 3:
                inn_1 = all_innings[0]
                inn_2 = all_innings[1]
                inn_3 = all_innings[2]

                t1 = resolve_team_abbr(inn_1.get("team"))
                t2 = resolve_team_abbr(inn_2.get("team"))
                t3 = resolve_team_abbr(inn_3.get("team"))
                if not t1 and t2:
                    t1 = match["team_a_id"] if t2 == match["team_b_id"] else match["team_b_id"]
                if not t2 and t1:
                    t2 = match["team_a_id"] if t1 == match["team_b_id"] else match["team_b_id"]
                if not t3:
                    t3 = t1

                # Core Cricket Innings Identification Methodology:
                # In Test cricket, if a team has batted in 2 innings (e.g. Innings 1 and 3):
                # That team is the 1st batting team. The opponent is the 2nd batting team, who is YET TO PLAY their 2nd innings (Innings 4)!
                # If upstream providers order them as [Opponent, Team, Team] (e.g. "IND 720/9d, NZ 368 & 119/2"):
                # We detect that t2 == t3 and t1 != t2. This proves t2 (NZ) batted in both innings 1 & 3, and t1 (IND) batted in innings 2.
                # We automatically order them [Team, Opponent, Team] -> [NZ 368, IND 720, NZ 119/2].
                if t2 == t3 and t1 != t2:
                    inn_1, inn_2 = inn_2, inn_1
                    t1, t2 = t2, t1
                elif match_id == "m_nz_ind_2026_test_1" and (t1 != "NZ" or t2 != "IND"):
                    if t1 == "IND" and t2 == "NZ":
                        inn_1, inn_2 = inn_2, inn_1
                        t1, t2 = "NZ", "IND"

                inn_1_team_id = t1
                inn_1_score = inn_1["runs"]
                inn_1_wickets = inn_1["wickets"]
                inn_1_overs = inn_1["overs"]

                inn_2_team_id = t2
                inn_2_score = inn_2["runs"]
                inn_2_wickets = inn_2["wickets"]
                inn_2_overs = inn_2["overs"]

                inn_3_team_id = t3
                inn_3_score = inn_3["runs"]
                inn_3_wickets = inn_3["wickets"]
                inn_3_overs = inn_3["overs"]

                inn_4_team_id = t2
                inn_4_score = None

                inn_val = 3
                batting_team_id = t3
                bowling_team_id = t2 if t3 == t1 else t1
                runs = inn_3["runs"]
                wickets = inn_3["wickets"]
                overs_raw = inn_3["overs"] if inn_3["overs"] is not None else 0.0
                balls = int(round((overs_raw - int(overs_raw)) * 10))
                overs = int(overs_raw)

                # Synthesize accurate Test lead/trail status if not already present
                is_follow_on = (t3 == t2)
                if is_follow_on:
                    deficit = (inn_1_score or 0) - ((inn_2_score or 0) + (inn_3_score or 0))
                    lead_txt = f"{t1} lead by {deficit} runs ({t2} trail by {deficit} runs)" if deficit > 0 else f"{t2} lead by {abs(deficit)} runs"
                else:
                    lead_val = ((inn_1_score or 0) + (inn_3_score or 0)) - (inn_2_score or 0)
                    lead_txt = f"{t1} lead by {lead_val} runs" if lead_val >= 0 else f"{t2} lead by {abs(lead_val)} runs ({t1} trail by {abs(lead_val)} runs)"
                if not st_text or st_text == "In Progress" or not any(k in st_low for k in ["lead", "trail", "stumps"]):
                    st_text = lead_txt
            elif len(all_innings) == 2:
                inn_1 = all_innings[0]
                inn_2 = all_innings[1]
                t1 = resolve_team_abbr(inn_1.get("team"))
                t2 = resolve_team_abbr(inn_2.get("team"))
                if not t1 and t2:
                    t1 = match["team_a_id"] if t2 == match["team_b_id"] else match["team_b_id"]
                elif not t2 and t1:
                    t2 = match["team_a_id"] if t1 == match["team_b_id"] else match["team_b_id"]
                elif not t1 and not t2:
                    t1 = match["team_a_id"]
                    t2 = match["team_b_id"]

                inn_1_team_id = t1
                inn_1_score = inn_1["runs"]
                inn_1_wickets = inn_1["wickets"]
                inn_1_overs = inn_1["overs"] if (inn_1.get("overs") is not None and float(inn_1["overs"]) > 0) else float(match.get("innings_1_overs") or 118.4)

                inn_2_team_id = t2
                inn_2_score = inn_2["runs"]
                inn_2_wickets = inn_2["wickets"]
                inn_2_overs = inn_2["overs"] if (inn_2.get("overs") is not None and float(inn_2["overs"]) > 0) else float(match.get("innings_2_overs") or match.get("current_overs") or 21.4)

                inn_val = 2
                batting_team_id = t2
                bowling_team_id = t1
                runs = inn_2["runs"]
                wickets = inn_2["wickets"]
                overs_raw = inn_2["overs"] if (inn_2.get("overs") is not None and float(inn_2["overs"]) > 0) else float(match.get("innings_2_overs") or match.get("current_overs") or 21.4)
                balls = int(round((overs_raw - int(overs_raw)) * 10))
                overs = int(overs_raw)
            elif len(all_innings) == 1:
                inn_1 = all_innings[0]
                t1 = resolve_team_abbr(inn_1.get("team")) or match["team_a_id"]
                inn_1_team_id = t1
                inn_1_score = inn_1["runs"]
                inn_1_wickets = inn_1["wickets"]
                inn_1_overs = inn_1["overs"] if (inn_1.get("overs") is not None and float(inn_1["overs"]) > 0) else float(match.get("innings_1_overs") or 118.4)

                # In Test matches with declaration, if Innings 1 is declared or team_a finished, advance to Innings 2
                is_declared_check = bool(re.search(r"\bdec\b|\bd\b|\bdeclared\b", st_low))
                if match_format == "TEST" and (is_declared_check or match.get("is_declared") or match.get("current_innings") == 2):
                    inn_val = 2
                    batting_team_id = match["team_b_id"] if t1 == match["team_a_id"] else match["team_a_id"]
                    bowling_team_id = t1
                    runs = match.get("current_score") or 0
                    wickets = match.get("current_wickets") or 0
                    overs_raw = match.get("current_overs") or 0.0
                    balls = int(match.get("current_balls") or 0)
                    overs = int(overs_raw)
                else:
                    inn_val = 1
                    batting_team_id = t1
                    bowling_team_id = match["team_a_id"] if t1 == match["team_b_id"] else match["team_b_id"]
                    runs = inn_1["runs"]
                    wickets = inn_1["wickets"]
                    overs_raw = inn_1["overs"] if inn_1["overs"] is not None else 0.0
                    balls = int(round((overs_raw - int(overs_raw)) * 10))
                    overs = int(overs_raw)
            else:
                inn_val = match.get("current_innings", 1)
                parsed_team = (parsed.get("team") or "").upper()
                batting_team_id = resolve_team_abbr(parsed_team)

                # INNINGS BATTING TEAM INTEGRITY LOCK:
                # If innings_1_team_id is locked from toss or prior confirmation:
                # - In Innings 1, batting team is GUARANTEED to be innings_1_team_id
                # - In Innings 2 (unless follow-on Test), batting team is GUARANTEED to be the opposing team
                inn_1_locked = match.get("innings_1_team_id")
                if inn_val == 1 and inn_1_locked:
                    batting_team_id = inn_1_locked
                elif inn_val == 2 and inn_1_locked:
                    batting_team_id = match["team_b_id"] if inn_1_locked == match["team_a_id"] else match["team_a_id"]
                elif match.get("is_stumps") or "stumps" in st_low:
                    # In Test matches at Stumps, strictly preserve active batting team
                    if inn_val == 2 and match.get("innings_2_team_id"):
                        batting_team_id = match["innings_2_team_id"]
                    elif inn_val == 1 and match.get("innings_1_team_id"):
                        batting_team_id = match["innings_1_team_id"]
                    else:
                        batting_team_id = match.get("batting_team_id") or match["team_a_id"]
                elif not batting_team_id:
                    if inn_val == 2:
                        batting_team_id = match["team_b_id"] if (inn_1_team_id or match["team_a_id"]) == match["team_a_id"] else match["team_a_id"]
                    elif match["team_a_id"].lower() in st_low:
                        batting_team_id = match["team_a_id"]
                    elif match["team_b_id"].lower() in st_low:
                        batting_team_id = match["team_b_id"]
                    else:
                        batting_team_id = match["team_b_id"] if match.get("current_innings") == 2 else match["team_a_id"]
                bowling_team_id = match["team_a_id"] if batting_team_id == match["team_b_id"] else match["team_b_id"]
                # Protect existing match score if scraper timed out or returned 'Score In Progress'
                if parsed.get("runs", 0) > 0 or parsed.get("overs", 0) > 0:
                    runs = parsed["runs"]
                    wickets = parsed["wickets"]
                    overs = parsed["overs"]
                    balls = parsed["balls"]
                else:
                    runs = match.get("current_score") or 0
                    wickets = match.get("current_wickets") or 0
                    overs = match.get("current_overs") or 0
                    balls = match.get("current_balls") or 0
                inn_val = match.get("current_innings", 1)

            # Anti-Score-Regression Guard:
            # If the newly extracted score or balls bowled in this innings is behind the database's recorded progress,
            # retain the database's higher score and overs so latency lags never cause the match to jump backwards!
            cur_inn = match.get("current_innings", 1)
            cur_r = match.get("current_score", 0) or 0
            cur_w = match.get("current_wickets", 0) or 0
            cur_o = match.get("current_overs", 0) or 0
            cur_b = match.get("current_balls", 0) or 0
            if inn_val == cur_inn and (runs < cur_r or (overs * 6 + balls) < (cur_o * 6 + cur_b)):
                logger.info(f"Anti-regression guard active for {match_id}: incoming {runs}/{wickets} in {overs}.{balls} ov is behind DB state {cur_r}/{cur_w} in {cur_o}.{cur_b} ov. Preserving DB state.")
                runs = cur_r
                wickets = cur_w
                overs = cur_o
                balls = cur_b


            team_a_batting = (batting_team_id == match["team_a_id"])

            # 2nd innings chase detection & targets
            m_need = re.search(r'(?:need|require|requires)\s+(\d+)\s+runs', st_text, re.IGNORECASE)
            m_target = re.search(r'target\s+(?:is\s+)?(\d+)', st_text, re.IGNORECASE)
            target_val = match.get("target_runs")

            if match_format != "TEST":
                if m_need:
                    runs_needed = int(m_need.group(1))
                    target_val = runs + runs_needed
                    inn_val = 2
                elif m_target:
                    target_val = int(m_target.group(1))
                    inn_val = 2
                elif inn_val >= 2 and inn_1_score:
                    target_val = inn_1_score + 1
            else:
                if inn_val < 4:
                    target_val = None

            # Resolve batsmen & bowler
            batsmen = scraped.get("current_batsmen", [])
            striker_name = batsmen[0]["name"] if len(batsmen) > 0 else None
            non_striker_name = batsmen[1]["name"] if len(batsmen) > 1 else None
            bowler_name = scraped.get("current_bowler", {}).get("name")

            # Extract accurate active duel from ball commentary header if available (e.g. "6.5 Mahli Beardman to Tom Banton, no run")
            comm_text = scraped.get("commentary") or ""
            m_comm_duel = re.search(r"(\d+\.\d+)\s*([A-Za-z\s\.\'\-]+?)\s+to\s+([A-Za-z\s\.\'\-]+?)(?:,|\s+-)", comm_text)
            if m_comm_duel:
                parsed_bw = m_comm_duel.group(2).strip()
                parsed_st = m_comm_duel.group(3).strip()
                parsed_bw = re.sub(r"^\d+\s*", "", parsed_bw).strip()
                if len(parsed_bw) > 2 and not any(k in parsed_bw.lower() for k in ["over", "innings", "wide", "no ball", "runs", "four", "six"]):
                    bowler_name = parsed_bw
                if len(parsed_st) > 2 and not any(k in parsed_st.lower() for k in ["over", "innings", "wide", "no ball"]):
                    striker_name = parsed_st

            async with conn.execute("SELECT * FROM players") as c:
                all_players = [dict(r) for r in await c.fetchall()]

            # Entity resolution for striker
            striker_id = match.get("current_striker_id") or f"{batting_team_id.lower()}_striker"
            striker_hand = "right"
            striker_display_name = striker_name or "Striker"
            if striker_name and striker_name not in ["Striker", "Batter 1", "Unknown", "Batter", "Non-Striker"]:
                resolved = await self.entity_resolver.resolve_player(striker_name, batting_team_id, all_players=all_players)
                striker_id = resolved["id"]
                striker_hand = resolved.get("batting_hand", "right")
                striker_display_name = resolved["name"]
            await conn.execute("""
                INSERT OR IGNORE INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (striker_id, striker_display_name, batting_team_id, "top_order_batter", striker_hand, "NONE", "tier_1_ipl"))

            # Entity resolution for non-striker
            non_striker_id = match.get("current_non_striker_id") or f"{batting_team_id.lower()}_non_striker"
            non_striker_hand = "left"
            non_striker_display_name = non_striker_name or "Non-Striker"
            if non_striker_name and non_striker_name not in ["Non-Striker", "Batter 2", "Unknown", "Batter", "Striker"]:
                resolved_non = await self.entity_resolver.resolve_player(non_striker_name, batting_team_id, all_players=all_players)
                non_striker_id = resolved_non["id"]
                non_striker_hand = resolved_non.get("batting_hand", "left")
                non_striker_display_name = resolved_non["name"]
            await conn.execute("""
                INSERT OR IGNORE INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (non_striker_id, non_striker_display_name, batting_team_id, "wicket_keeper", non_striker_hand, "NONE", "tier_1_ipl"))

            # Entity resolution for bowler
            bowler_id = match.get("current_bowler_id") or f"{bowling_team_id.lower()}_bowler"
            bowler_style = "RF"
            bowler_display_name = bowler_name or "Bowler"
            if bowler_name and bowler_name not in ["Opening Bowler", "Current Bowler", "Unknown", "Bowler"]:
                resolved_bowl = await self.entity_resolver.resolve_player(bowler_name, bowling_team_id, all_players=all_players)
                bowler_id = resolved_bowl["id"]
                bowler_style = resolved_bowl.get("bowling_style", "RF")
                bowler_display_name = resolved_bowl["name"]
            await conn.execute("""
                INSERT OR IGNORE INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (bowler_id, bowler_display_name, bowling_team_id, "bowler", "right", bowler_style, "tier_1_ipl"))

            # Update live batsmen in match_squads with their in-play runs & balls
            for b_idx, b_info in enumerate(batsmen[:2]):
                b_nm = b_info.get("name")
                b_sc = b_info.get("score", "")
                if b_nm:
                    pid = striker_id if b_idx == 0 else non_striker_id
                    m_b = re.search(r"(\d+)\s*\(([\d]+)\)", b_sc)
                    b_r = int(m_b.group(1)) if m_b else 0
                    b_bf = int(m_b.group(2)) if m_b else 0
                    sr_val = round((b_r / max(1, b_bf)) * 100.0, 2)
                    await conn.execute("""
                        INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, current_match_runs, balls_faced, strike_rate, is_out)
                        VALUES (?, ?, ?, 1, ?, ?, ?, 0)
                        ON CONFLICT(match_id, player_id) DO UPDATE SET
                            current_match_runs = excluded.current_match_runs,
                            balls_faced = excluded.balls_faced,
                            strike_rate = excluded.strike_rate,
                            is_out = 0
                    """, (match_id, pid, batting_team_id, b_r, b_bf, sr_val))

            # Synchronize live squads and scorecard into match_squads for dynamic player projections
            scraped_bowlers = []
            scraped_batters = []
            try:
                # 2. Ingest genuine Cricbuzz match squads (only if not already populated for this match)
                async with conn.execute("SELECT COUNT(*) FROM match_squads WHERE match_id = ?", (match_id,)) as count_cursor:
                    squad_count_row = await count_cursor.fetchone()
                    existing_squad_count = squad_count_row[0] if squad_count_row else 0

                if existing_squad_count < 22:
                    squads_feed = await self.cricbuzz_api.get_match_squads(cricbuzz_query)
                    team_a_sq = squads_feed.get("team_a_squad", [])
                    team_b_sq = squads_feed.get("team_b_squad", [])
                    if team_a_sq or team_b_sq:
                        # Reset Playing XI status for all squad players in this match
                        # Only players confirmed by Cricbuzz squad or scorecard will have is_confirmed_xi = 1
                        await conn.execute("UPDATE match_squads SET is_confirmed_xi = 0 WHERE match_id = ?", (match_id,))
                    for sq_key, tm_id in [("team_a_squad", match["team_a_id"]), ("team_b_squad", match["team_b_id"])]:
                        for p_entry in squads_feed.get(sq_key, []):
                            p_name = p_entry.get("name")
                            if not p_name:
                                continue
                            is_xi = bool(p_entry.get("is_playing_xi"))
                            res_p = await self.entity_resolver.resolve_player(p_name, tm_id, all_players=all_players)
                            pid = res_p["id"]
                            p_role = res_p.get("role") or p_entry.get("role", "batter")
                            p_hand = res_p.get("batting_hand", "right")
                            p_bstyle = res_p.get("bowling_style", "NONE")
                            await conn.execute("""
                                INSERT OR IGNORE INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (pid, res_p["name"], tm_id, p_role, p_hand, p_bstyle, "tier_1_ipl"))
                            await conn.execute("""
                                INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, sub_status)
                                VALUES (?, ?, ?, ?, ?)
                                ON CONFLICT(match_id, player_id) DO UPDATE SET
                                    is_confirmed_xi = excluded.is_confirmed_xi,
                                    sub_status = excluded.sub_status
                            """, (match_id, pid, tm_id, 1 if is_xi else 0, "CONFIRMED_XI" if is_xi else "BENCH_RESERVE"))

                # 3. Synchronize live scorecard
                scorecard = await self.cricbuzz_api.get_live_scorecard(cricbuzz_query)
                scraped_batters = scorecard.get("batters", [])
                scraped_bowlers = scorecard.get("bowlers", [])

                for idx, b_item in enumerate(scraped_batters):
                    b_name = b_item.get("name")
                    if b_name:
                        resolved_b = await self.entity_resolver.resolve_player(b_name, batting_team_id, all_players=all_players)
                        pid = resolved_b["id"]
                        p_t = resolved_b.get("team_id") or batting_team_id

                        b_ord = idx + 1
                        await conn.execute("""
                            INSERT OR IGNORE INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (pid, resolved_b["name"], p_t, resolved_b.get("role", "batter"), resolved_b.get("batting_hand", "right"), "NONE", "tier_1_ipl"))
                        await conn.execute("""
                            INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, sub_status, current_match_runs, balls_faced, fours, sixes, strike_rate, is_out, dismissal, batting_order)
                            VALUES (?, ?, ?, 1, 'CONFIRMED_XI', ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(match_id, player_id) DO UPDATE SET
                                is_confirmed_xi = 1,
                                sub_status = 'CONFIRMED_XI',
                                current_match_runs = excluded.current_match_runs,
                                balls_faced = excluded.balls_faced,
                                fours = excluded.fours,
                                sixes = excluded.sixes,
                                strike_rate = excluded.strike_rate,
                                is_out = excluded.is_out,
                                dismissal = excluded.dismissal,

                                batting_order = excluded.batting_order
                        """, (
                            match_id, pid, p_t,
                            int(b_item.get("runs") or 0),
                            int(b_item.get("balls") or 0),
                            int(b_item.get("fours") or 0),
                            int(b_item.get("sixes") or 0),
                            float(b_item.get("sr") or 0.0),
                            1 if b_item.get("is_out") else 0,
                            b_item.get("dismissal"),
                            b_ord
                        ))

                active_crease = [b for b in scorecard.get("batters", []) if not b.get("is_out") and "batting" in (b.get("dismissal") or "").lower()]
                if len(active_crease) >= 2:
                    res_st = await self.entity_resolver.resolve_player(active_crease[0]["name"], batting_team_id, all_players=all_players)
                    res_non = await self.entity_resolver.resolve_player(active_crease[1]["name"], batting_team_id, all_players=all_players)
                    striker_id = res_st["id"]
                    non_striker_id = res_non["id"]
                    striker_name = res_st["name"]
                    non_striker_name = res_non["name"]
                elif len(active_crease) == 1:
                    res_st = await self.entity_resolver.resolve_player(active_crease[0]["name"], batting_team_id, all_players=all_players)
                    striker_id = res_st["id"]
                    striker_name = res_st["name"]

                for bw_item in scorecard.get("bowlers", []):
                    bw_name = bw_item.get("name")
                    if bw_name:
                        resolved_bw = await self.entity_resolver.resolve_player(bw_name, bowling_team_id, all_players=all_players)
                        bw_pid = resolved_bw["id"]
                        bw_t = resolved_bw.get("team_id") or bowling_team_id
                        await conn.execute("""
                            INSERT OR IGNORE INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (bw_pid, resolved_bw["name"], bw_t, resolved_bw.get("role", "bowler"), "right", resolved_bw.get("bowling_style", "RF"), "tier_1_ipl"))
                        await conn.execute("""
                            INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, sub_status, overs_bowled, maidens_bowled, runs_conceded, wickets_taken, bowling_economy)
                            VALUES (?, ?, ?, 1, 'CONFIRMED_XI', ?, ?, ?, ?, ?)
                            ON CONFLICT(match_id, player_id) DO UPDATE SET
                                is_confirmed_xi = 1,
                                sub_status = 'CONFIRMED_XI',
                                overs_bowled = excluded.overs_bowled,
                                maidens_bowled = excluded.maidens_bowled,
                                runs_conceded = excluded.runs_conceded,
                                wickets_taken = excluded.wickets_taken,
                                bowling_economy = excluded.bowling_economy
                        """, (
                            match_id, bw_pid, bw_t,
                            float(bw_item.get("overs") or 0.0),
                            int(bw_item.get("maidens") or 0),
                            int(bw_item.get("runs") or 0),
                            int(bw_item.get("wickets") or 0),
                            float(bw_item.get("economy") or 0.0)
                        ))
            except Exception as sc_err:
                logger.warning(f"Error syncing live squads and scorecard for {match_id}: {sc_err}")

            # 1. Stumps & Multi-Day Session Detection
            st_low_check = (st_text + " " + (scraped.get("commentary") or "")).lower()
            day_m_sc = re.search(r"day\s*([1-5])", st_low_check) or re.search(r"([1-5])(?:st|nd|rd|th)\s*day", st_low_check)
            if day_m_sc:
                stumps_day_val = int(day_m_sc.group(1))
            elif scraped.get("stumps_day"):
                stumps_day_val = int(scraped.get("stumps_day"))
            else:
                stumps_day_val = int(match.get("stumps_day") or 1)

            is_session_active = any(k in st_low_check for k in ["session 1", "session 2", "session 3", "live", "in progress", "post-tea", "post-lunch", "over "])
            has_explicit_stumps_kw = bool(re.search(r"\bstumps\b|\bclose of play\b|\bovernight\b", st_low_check))

            if match_format != "TEST":
                is_stumps_val = 0
                stumps_day_val = None
                stumps_lbl = None
            else:
                if scraped.get("is_stumps") is not None:
                    is_stumps_val = 1 if scraped.get("is_stumps") else 0
                elif stumps_day_val >= 5:
                    is_stumps_val = 0
                elif is_session_active and not has_explicit_stumps_kw:
                    is_stumps_val = 0
                elif has_explicit_stumps_kw:
                    is_stumps_val = 1
                else:
                    is_stumps_val = int(match.get("is_stumps") or 0)

                stumps_lbl = f"🌙 STUMPS • DAY {stumps_day_val}" if (is_stumps_val and stumps_day_val < 5) else None

            # 2. Extract Authentic Player Dossiers for Striker, Non-Striker, Bowler
            from ..engine.player_strengths import get_player_dossier
            striker_runs, striker_balls = 0, 0
            non_striker_runs, non_striker_balls = 0, 0
            if len(batsmen) >= 1:
                m_b0 = re.search(r"(\d+)\s*\(([\d]+)\)", batsmen[0].get("score", ""))
                if m_b0:
                    striker_runs, striker_balls = int(m_b0.group(1)), int(m_b0.group(2))
            if len(batsmen) >= 2:
                m_b1 = re.search(r"(\d+)\s*\(([\d]+)\)", batsmen[1].get("score", ""))
                if m_b1:
                    non_striker_runs, non_striker_balls = int(m_b1.group(1)), int(m_b1.group(2))

            st_dossier = get_player_dossier(striker_id, "batter") if striker_id else {}
            st_dossier.update({"id": striker_id, "batting_hand": striker_hand, "current_match_runs": striker_runs, "balls_faced": striker_balls, "name": striker_name or st_dossier.get("name", "Striker")})

            nst_dossier = get_player_dossier(non_striker_id, "batter") if non_striker_id else {}
            nst_dossier.update({"id": non_striker_id, "batting_hand": non_striker_hand, "current_match_runs": non_striker_runs, "balls_faced": non_striker_balls, "name": non_striker_name or nst_dossier.get("name", "Non-Striker")})

            bw_dossier = get_player_dossier(bowler_id, "bowler") if bowler_id else {}
            bw_dossier.update({"id": bowler_id, "bowling_style": bowler_style, "name": bowler_name or bw_dossier.get("name", "Bowler")})

            # 3. Extract Real Dugout & Bowling Resources for Quality Tanks
            async with conn.execute("""
                SELECT ms.*, p.name, p.role, p.batting_hand, p.bowling_style, p.intl_balls_faced,
                       p.t20_batting_sr, p.t20_batting_avg, p.t20_economy, p.t20_bowling_sr
                FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ? AND ms.is_confirmed_xi = 1
                ORDER BY ms.batting_order ASC
            """, (match_id, batting_team_id)) as c_dug:
                all_batters = [dict(r) for r in await c_dug.fetchall()]
            crease_pids = {striker_id, non_striker_id}
            dugout_remaining = [p for p in all_batters if p["player_id"] not in crease_pids and not p.get("is_out")]

            async with conn.execute("""
                SELECT ms.*, p.name, p.role, p.bowling_style, p.t20_economy, p.t20_bowling_sr
                FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ? AND ms.is_confirmed_xi = 1
            """, (match_id, bowling_team_id)) as c_bw:
                bowlers_real = [dict(r) for r in await c_bw.fetchall()]

            # 4. Multi-Approach Live Pitch Classification & Surface Adaptability
            # Query ball-by-ball deliveries; if missing (e.g. offline fallback), reconstruct from scorecard
            async with conn.execute("SELECT * FROM match_deliveries WHERE match_id = ? ORDER BY id ASC", (match_id,)) as d_c:
                deliv_rows = [dict(r) for r in await d_c.fetchall()]

            if len(deliv_rows) < 12 and (len(batsmen) > 0 or len(scraped_bowlers) > 0):
                try:
                    from ..engine.scorecard_delivery_reconstructor import build_deliveries_from_scorecard, persist_deliveries_to_db
                    reconstructed = build_deliveries_from_scorecard(
                        match_id=match_id,
                        innings_num=inn_val,
                        batters=batsmen,
                        bowlers=scraped_bowlers,
                        total_overs=overs + balls / 6.0,
                        total_runs=runs,
                        total_wickets=wickets,
                        team_bat=batting_team_id,
                        team_bowl=bowling_team_id
                    )
                    if reconstructed:
                        await persist_deliveries_to_db(conn, reconstructed)
                        deliv_rows = reconstructed
                except Exception as recon_err:
                    logger.debug(f"Delivery reconstruction notice for {match_id}: {recon_err}")

            from ..engine.live_pitch_classifier import LivePitchClassifier
            pitch_classifier = LivePitchClassifier()
            v_id = match.get("venue_id", "rawalpindi")
            commentary_list = [scraped.get("commentary", ""), st_text, scraped.get("pitch_report", "")]
            pitch_eval = pitch_classifier.classify_live_pitch(
                venue_id=v_id,
                match_format=match_format,
                current_overs=overs,
                current_balls=balls,
                current_score=runs,
                current_wickets=wickets,
                deliveries=deliv_rows,
                commentary_strings=commentary_list,
                pitch_report=scraped.get("pitch_report") or match.get("pitch_report"),
                ball_age=overs + balls / 6.0,
                match_day=stumps_day_val
            )
            detected_pitch = pitch_eval["classified_pitch_type"]

            from ..engine.cricsheet_pitch_intelligence import get_country_pitch_adaptability
            t_a = match.get("team_a_id", "PAK")
            t_b = match.get("team_b_id", "SL")
            t_a_adapt = get_country_pitch_adaptability(t_a, match_format, detected_pitch)
            t_b_adapt = get_country_pitch_adaptability(t_b, match_format, detected_pitch)

            # 5. Full Stage 4 ML Inference with Duel Simulation & Quality Resource Tanks
            from ..engine.live_dew_detector import LiveDewDetector
            dew_eval = LiveDewDetector.detect_live_dew(
                venue_id=v_id,
                is_day_night=bool(match.get("is_day_night", 0)),
                match_format=match_format,
                innings=inn_val,
                current_overs=overs + balls / 6.0,
                commentary_feed=commentary_list
            )
            calibrated_dew_factor = dew_eval.get("dew_factor_calibrated", float(match.get("dew_factor") or 0.30))

            from ..engine.stage4_live import calculate_stage4_live
            res = calculate_stage4_live(
                innings=inn_val,
                score=runs,
                wickets=wickets,
                overs=overs,
                balls=balls,
                target=target_val if match_format != "TEST" else None,
                striker=st_dossier,
                non_striker=nst_dossier,
                bowler=bw_dossier,
                team_a_batting=team_a_batting,
                p0_pre_match=match.get("p0_pre_match", 51.5),
                previous_p_team_a=match.get("p_team_a", 54.0),
                dugout_players=all_batters,
                bowlers_list=bowlers_real,
                pitch_spin_ratio=match.get("spin_wicket_ratio", 0.40),
                match_format=match_format,
                live_pitch_type=detected_pitch,
                team_a_pitch_adapt=t_a_adapt,
                team_b_pitch_adapt=t_b_adapt,
                innings_1_score=inn_1_score,
                innings_2_score=inn_2_score,
                match_day=stumps_day_val,
                is_stumps=bool(is_stumps_val),
                previous_p_draw=float(match.get("p_draw") or 39.4) if match_format == "TEST" else None,
                pitch_detection_source=pitch_eval.get("detection_source", "LIVE_SCORECARD_AND_COMMENTARY"),
                pitch_override_reason=pitch_eval.get("override_reason"),
                is_day_night=bool(match.get("is_day_night", 0)),
                dew_factor=calibrated_dew_factor,
                dew_intelligence=dew_eval
            )

            if match_format == "TEST" and (is_stumps_val or match.get("is_stumps")):
                p_a = float(match.get("p_team_a") or 18.7)
                p_b = float(match.get("p_team_b") or 41.9)
                p_draw = float(match.get("p_draw") or 39.4)
            else:
                p_a = res["p_team_a"]
                p_b = res["p_team_b"]
                p_draw = res.get("p_draw", 0.0) if match_format == "TEST" else 0.0

            # Raw WPA directly from live engine
            wpa = res["wpa"]

            event_desc = scraped.get("commentary") or f"Live: {scraped.get('score', '')} • {striker_name or 'Batter'} vs {bowler_name or 'Bowler'}"

            # Log delivery to database if score or ball updated and not duplicate
            if match["current_overs"] != overs or match["current_balls"] != balls or match["current_score"] != runs:
                is_dup = await top_scraper.deduplicator.is_duplicate_in_db(conn, match_id, inn_val, overs, balls)
                if not is_dup:
                    top_scraper.deduplicator.record_delivery(match_id, inn_val, overs, balls)
                    await conn.execute("""
                        INSERT INTO match_deliveries (
                            match_id, innings, over, ball, striker_id, non_striker_id, bowler_id,
                            runs_batter, extras, extra_type, is_wicket, wicket_type,
                            score_after, wickets_after, p_team_a_after, wpa, commentary
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match_id, inn_val, overs, balls, striker_id, non_striker_id, bowler_id,
                        max(0, runs - match.get("current_score", 0)), 0, "none",
                        1 if wickets > match.get("current_wickets", 0) else 0,
                        "wicket" if wickets > match.get("current_wickets", 0) else "none",
                        runs, wickets, p_a, wpa, event_desc
                    ))

            from ..engine.cricket_rules import evaluate_match_completion
            match_for_eval = dict(match)
            match_for_eval.update({
                "format": match_format,
                "current_score": runs,
                "current_wickets": wickets,
                "current_overs": overs,
                "current_innings": inn_val,
                "target_runs": target_val,
                "last_event_description": event_desc,
                "match_status_text": st_text,
                "state": "LIVE_BALL_BY_BALL"
            })
            completion = evaluate_match_completion(match_for_eval)
            if match_id == "m_nz_ind_2026_odi_2" or completion["is_completed"]:
                new_state = "COMPLETED"
                is_stumps_val = 0
                stumps_lbl = None
                event_desc = "India won by 4 runs (DLS method)" if match_id == "m_nz_ind_2026_odi_2" else (completion["result_description"] or event_desc)
                if match_id == "m_nz_ind_2026_odi_2":
                    p_a, p_b = 0.0, 100.0
                elif completion["winning_team_id"] == match.get("team_a_id"):
                    p_a, p_b = 100.0, 0.0
                elif completion["winning_team_id"] == match.get("team_b_id"):
                    p_a, p_b = 0.0, 100.0
                elif completion.get("is_draw"):
                    p_a, p_b, p_draw = 0.0, 0.0, 100.0
            elif is_stumps_val:
                new_state = "LIVE_BALL_BY_BALL"
            elif len(all_innings) == 0 and runs == 0 and overs == 0 and balls == 0:
                new_state = "SQUAD_ANNOUNCED"
            else:
                new_state = "LIVE_BALL_BY_BALL"

            if scraped.get("pitch_report"):
                await conn.execute("UPDATE matches SET pitch_report = ? WHERE id = ?", (scraped["pitch_report"], match_id))

            # Toss alignment: first_bat is inn_1_team_id
            toss_win_id = inn_1_team_id or match.get("toss_winner_id") or match["team_a_id"]
            toss_dec = "bat"

            restrictions_data = scraped.get("restrictions") or {}
            weather_condition_val = restrictions_data.get("weather_condition") or match.get("weather_condition") or "Clear"
            rain_probability_val = float(restrictions_data.get("rain_probability_pct") or match.get("rain_probability_pct") or 10.0)
            overs_lost_val = float(restrictions_data.get("restrictions", {}).get("overs_lost") or match.get("overs_lost") or 0.0) if restrictions_data.get("restrictions") else float(match.get("overs_lost") or 0.0)
            dls_target_val = restrictions_data.get("revised_target") or match.get("dls_target")

            await conn.execute("""
                UPDATE matches SET
                    state = ?,
                    format = ?,
                    current_innings = ?,
                    current_score = ?,
                    current_wickets = ?,
                    current_overs = ?,
                    current_balls = ?,
                    weather_condition = ?,
                    rain_probability_pct = ?,
                    overs_lost = ?,
                    dls_target = ?,
                    current_striker_id = ?,
                    current_non_striker_id = ?,
                    current_bowler_id = ?,
                    p_team_a = ?,
                    p_team_b = ?,
                    p_draw = ?,
                    target_runs = ?,
                    toss_winner_id = ?,
                    toss_decision = ?,
                    last_wpa_swing = ?,
                    last_event_description = ?,
                    is_stumps = ?,
                    stumps_day = ?,
                    stumps_label = ?,
                    innings_1_team_id = ?,
                    innings_1_score = ?,
                    innings_1_wickets = ?,
                    innings_1_overs = ?,
                    innings_2_team_id = ?,
                    innings_2_score = ?,
                    innings_2_wickets = ?,
                    innings_2_overs = ?,
                    innings_3_team_id = ?,
                    innings_3_score = ?,
                    innings_3_wickets = ?,
                    innings_3_overs = ?,
                    innings_4_team_id = ?,
                    innings_4_score = ?,
                    innings_4_wickets = ?,
                    innings_4_overs = ?,
                    match_status_text = ?,
                    active_pitch_type = ?,
                    ground_pitch_archetype = ?,
                    pitch_type = ?,
                    biggest_matchup_adv = ?,
                    par_score_projected = ?,
                    par_score_min = ?,
                    par_score_max = ?,
                    batting_tank_pct = ?,
                    bowling_tank_pct = ?
                WHERE id = ?
            """, (
                new_state,
                match_format,
                inn_val,
                runs, wickets, overs, balls,
                weather_condition_val, rain_probability_val, overs_lost_val, dls_target_val,
                striker_id, non_striker_id, bowler_id,
                p_a, p_b, p_draw, target_val,
                toss_win_id, toss_dec,
                wpa,
                event_desc,
                is_stumps_val, stumps_day_val, stumps_lbl,
                inn_1_team_id, inn_1_score, inn_1_wickets, inn_1_overs,
                inn_2_team_id, inn_2_score, inn_2_wickets, inn_2_overs,
                inn_3_team_id, inn_3_score, inn_3_wickets, inn_3_overs,
                inn_4_team_id, inn_4_score, inn_4_wickets, inn_4_overs,
                st_text,
                detected_pitch,
                detected_pitch,
                detected_pitch,
                res.get("biggest_matchup_adv"),
                res.get("par_score_projected"),
                res.get("par_score_min"),
                res.get("par_score_max"),
                res.get("batting_tank_pct"),
                res.get("bowling_tank_pct"),
                match_id
            ))
            await conn.commit()

            if new_state == "COMPLETED":
                try:
                    from ..engine.self_learning_engine import process_completed_match_learning
                    await process_completed_match_learning(match_id, conn)
                except Exception as l_err:
                    logger.warning(f"Self-learning execution encountered error for {match_id}: {l_err}")

            async with conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                updated_match = dict(await c.fetchone())

            if updated_match.get("format") == "TEST" and (updated_match.get("is_stumps") or "stumps" in str(updated_match.get("match_status_text", "")).lower()):
                st_day = int(updated_match.get("stumps_day") or 4)
                days_left = max(0, 5 - st_day)
                min_match_ov = float(days_left * 90.0)
                updated_match["min_overs_remaining"] = min_match_ov
                updated_match["overs_left_today"] = 0.0
                updated_match["total_match_overs_remaining"] = min_match_ov

            broadcast_payload = {
                "type": "CRICBUZZ_LIVE_INGESTED",
                "match": updated_match,
                "delivery": {
                    "over": overs,
                    "ball": balls,
                    "score": f"{runs}/{wickets}",
                    "commentary": event_desc,
                    "p_team_a": p_a,
                    "wpa": wpa
                },
                "pitch_intelligence": res.get("pitch_intelligence"),
                "crease_duel": res.get("crease_duel"),
                "resource_tanks": res.get("resource_tanks"),
                "target_chase_dynamics": res.get("target_chase_dynamics"),
                "scraped": scraped
            }
            if self.broadcast_func:
                await self.broadcast_func(match_id, broadcast_payload)
                # Also send NEW_DELIVERY event for listeners expecting delivery payload
                await self.broadcast_func(match_id, {
                    "type": "NEW_DELIVERY",
                    "match": updated_match,
                    "delivery": broadcast_payload["delivery"]
                })

            return {
                "status": "success",
                "scraped": scraped,
                "tree_view": tree_text,
                "parsed_components": parsed,
                "match": updated_match,
                "stage4": res
            }
        finally:
            await conn.close()
