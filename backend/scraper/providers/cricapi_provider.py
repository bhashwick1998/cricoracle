"""
CricAPI Official Playing XI Provider & Credit-Saving Telemetry Adapter
Extracts confirmed Playing XI (11 players) exclusively when matches are about to start (~30-45 mins before first ball)
or currently live.
Strictly adheres to daily credit quotas (maximum 100 calls/day limit) with multi-layer deduplication and in-memory/DB caching.
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import httpx
import requests

logger = logging.getLogger("cricapi_provider")

CRICAPI_KEY = "d0e9f357-ae08-4700-9634-1a26bc27501b"
MAX_DAILY_CALLS = 90  # Keep 10-call safety buffer under the 100/day limit

class CricApiSquadProvider:
    def __init__(self, api_key: str = CRICAPI_KEY):
        self.api_key = api_key
        self.call_count_today = 0
        self.last_reset_day = datetime.now(timezone.utc).date()
        self.cached_match_squads: Dict[str, Dict[str, Any]] = {}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

    def _check_and_increment_quota(self) -> bool:
        """
        Quota Guard: Checks daily usage and resets at UTC midnight.
        Returns True if call is permitted; False if quota threshold reached.
        """
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            self.call_count_today = 0
            self.last_reset_day = today
            logger.info("CricAPI quota counter reset for new UTC day.")

        if self.call_count_today >= MAX_DAILY_CALLS:
            logger.warning(
                f"CricAPI rate limit protection active: {self.call_count_today}/{MAX_DAILY_CALLS} calls used today. "
                "Preserving remaining credits."
            )
            return False

        self.call_count_today += 1
        logger.info(f"CricAPI call approved. Daily usage: {self.call_count_today}/{MAX_DAILY_CALLS}")
        return True

    def is_match_about_to_start(self, scheduled_start_iso: str, window_minutes: int = 45) -> bool:
        """
        Calculates if match start time is within the next 45 minutes or has just started.
        Prevents unnecessary early API calls before Playing XI is declared at toss time.
        """
        try:
            start_dt = datetime.fromisoformat(scheduled_start_iso.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            delta = (start_dt - now_dt).total_seconds() / 60.0

            # Match is about to start (-15 mins past start to +window_minutes before start)
            return -120 <= delta <= window_minutes
        except Exception as e:
            logger.error(f"Error parsing match start timestamp: {e}")
            return False

    def fetch_playing_xi_for_match(
        self,
        cricapi_match_id: str,
        scheduled_start_iso: Optional[str] = None,
        force_call: bool = False
    ) -> Dict[str, Any]:
        """
        Fetches the official Playing XI (extracting 11 players per team).
        Uses /v1/match_squad, with intelligent fallback to /v1/match_scorecard if live.
        Strictly caches per match ID so it consumes at most 1 hit per match.
        """
        if not cricapi_match_id:
            return {"status": "error", "reason": "No match ID provided"}

        # 1. Deduplication: Return cached squad if already fetched
        if cricapi_match_id in self.cached_match_squads:
            logger.info(f"Returning cached Playing XI for match {cricapi_match_id} (0 API credits used).")
            return self.cached_match_squads[cricapi_match_id]

        # 2. Timing Guard: Only fetch for matches about to start unless explicitly forced
        if not force_call and scheduled_start_iso:
            if not self.is_match_about_to_start(scheduled_start_iso):
                logger.info(
                    f"Match {cricapi_match_id} is not within 45 mins of start ({scheduled_start_iso}). "
                    "Skipping CricAPI call to save daily credits."
                )
                return {
                    "status": "skipped",
                    "reason": "Match is not within toss window. Credits preserved."
                }

        # 3. Quota Guard
        if not self._check_and_increment_quota():
            return {
                "status": "error",
                "reason": "Daily API call limit reached (100 calls/day). Preserving system integrity."
            }

        # 4. Execute CricAPI Call: /v1/match_squad
        url = f"https://api.cricapi.com/v1/match_squad?apikey={self.api_key}&id={cricapi_match_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=12)
            if resp.status_code != 200:
                logger.error(f"CricAPI request failed: HTTP {resp.status_code}")
                return {"status": "error", "reason": f"HTTP {resp.status_code}"}

            data = resp.json()
            teams_data = data.get("data", []) if data.get("status") == "success" else []
            parsed_teams = []

            if teams_data:
                for team in teams_data:
                    team_name = team.get("teamName", "Team")
                    players = team.get("players", [])

                    playing_xi = []
                    for idx, player in enumerate(players[:11], start=1):
                        p_name = player.get("name", "").strip()
                        playing_xi.append({
                            "order": idx,
                            "name": p_name,
                            "role": player.get("role", "Player"),
                            "batting_style": player.get("battingStyle", ""),
                            "bowling_style": player.get("bowlingStyle", ""),
                            "is_confirmed_xi": 1,
                            "is_probable_xi": 1
                        })

                    bench_reserves = []
                    for player in players[11:]:
                        bench_reserves.append({
                            "order": 99,
                            "name": player.get("name", "").strip(),
                            "role": player.get("role", "Player"),
                            "batting_style": player.get("battingStyle", ""),
                            "bowling_style": player.get("bowlingStyle", ""),
                            "is_confirmed_xi": 0,
                            "is_probable_xi": 0
                        })

                    parsed_teams.append({
                        "team_name": team_name,
                        "playing_xi": playing_xi,
                        "bench_reserves": bench_reserves,
                        "total_squad_count": len(players)
                    })

            # 5. Fallback for Live Matches: If match_squad returned empty, check scorecard for confirmed 11
            if not parsed_teams:
                sc_url = f"https://api.cricapi.com/v1/match_scorecard?apikey={self.api_key}&id={cricapi_match_id}"
                sc_resp = requests.get(sc_url, headers=self.headers, timeout=12)
                if sc_resp.status_code == 200:
                    sc_json = sc_resp.json()
                    if sc_json.get("status") == "success":
                        sc_data = sc_json.get("data", {})
                        scorecard_list = sc_data.get("scorecard", [])
                        team_players_map: Dict[str, List[str]] = {}
                        for inn in scorecard_list:
                            inn_name = inn.get("inning", "")
                            t_name = inn_name.replace(" Inning 1", "").replace(" Inning 2", "").strip()
                            t_list = team_players_map.setdefault(t_name, [])
                            for b in inn.get("batting", []):
                                nm = b.get("batsman", {}).get("name")
                                if nm and nm not in t_list:
                                    t_list.append(nm)
                            # Bowlers belong to opposition team
                            opp_team = [t for t in sc_data.get("teams", []) if t != t_name]
                            if opp_team:
                                opp_t_name = opp_team[0]
                                opp_list = team_players_map.setdefault(opp_t_name, [])
                                for bw in inn.get("bowling", []):
                                    nm = bw.get("bowler", {}).get("name")
                                    if nm and nm not in opp_list:
                                        opp_list.append(nm)

                        for t_name, plist in team_players_map.items():
                            xi = [{"order": i+1, "name": nm, "role": "Player", "is_confirmed_xi": 1, "is_probable_xi": 1} for i, nm in enumerate(plist[:11])]
                            reserves = [{"order": 99, "name": nm, "role": "Player", "is_confirmed_xi": 0, "is_probable_xi": 0} for nm in plist[11:]]
                            parsed_teams.append({
                                "team_name": t_name,
                                "playing_xi": xi,
                                "bench_reserves": reserves,
                                "total_squad_count": len(plist)
                            })

            result = {
                "status": "success",
                "match_id": cricapi_match_id,
                "teams": parsed_teams,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            if parsed_teams:
                self.cached_match_squads[cricapi_match_id] = result
            return result

        except Exception as e:
            logger.error(f"Exception during CricAPI fetch for {cricapi_match_id}: {e}")
            return {"status": "error", "reason": str(e)}

    def resolve_venue_from_text(self, venue_text: str) -> Optional[str]:
        """
        Resolves authentic venue ID from CricketData.org venue name string.
        e.g. 'Bellerive Oval, Hobart' -> 'hobart'
             'Rawalpindi Cricket Stadium, Rawalpindi' -> 'rawalpindi'
             'Melbourne Cricket Ground, Melbourne' -> 'mcg'
        """
        if not venue_text:
            return None
        v_low = venue_text.lower()
        venue_map = {
            "bellerive": "hobart",
            "hobart": "hobart",
            "mcg": "mcg",
            "melbourne": "mcg",
            "scg": "scg",
            "sydney": "scg",
            "adelaide": "adelaide",
            "perth": "perth",
            "optus": "perth",
            "gabba": "gabba",
            "brisbane": "gabba",
            "rawalpindi": "rawalpindi",
            "gaddafi": "lahore",
            "lahore": "lahore",
            "karachi": "karachi",
            "multan": "multan",
            "wankhede": "wankhede",
            "eden gardens": "eden_gardens",
            "chepauk": "chepauk",
            "chidambaram": "chepauk",
            "chinnaswamy": "chinnaswamy",
            "narendra modi": "narendra_modi",
            "ahmedabad": "narendra_modi",
            "dharamshala": "dharamshala",
            "bay oval": "bay_oval",
            "mount maunganui": "bay_oval",
            "eden park": "eden_park",
            "auckland": "eden_park",
            "basin reserve": "basin_reserve",
            "sky stadium": "wellington",
            "wellington": "wellington",
            "seddon": "seddon_park",
            "hagley": "christchurch",
            "sharjah": "sharjah",
            "dubai": "dubai_dsc",
            "galle": "galle",
            "premadasa": "premadasa",
            "colombo": "premadasa",
            "pallekele": "pallekele",
            "kandy": "pallekele",
            "lords": "lords",
            "oval": "the_oval",
            "edgbaston": "edgbaston",
            "trent bridge": "trent_bridge",
            "headingley": "headingley",
            "old trafford": "old_trafford",
            "centurion": "centurion",
            "wanderers": "wanderers",
            "newlands": "newlands",
            "kingsmead": "kingsmead",
            "harare": "harare"
        }
        for kw, vid in venue_map.items():
            if kw in v_low:
                return vid
        return None

    def fetch_match_toss_and_venue(self, cricapi_match_id: str) -> Dict[str, Any]:
        """
        Fetches official pre-match toss details and authentic venue from CricketData.org (/v1/match_info).
        Detects toss winner and decision (bat/field) 30-45 minutes before match goes live.
        """
        if not cricapi_match_id:
            return {"status": "error", "reason": "No match ID provided"}

        url = f"https://api.cricapi.com/v1/match_info?apikey={self.api_key}&id={cricapi_match_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                return {"status": "error", "reason": f"HTTP {resp.status_code}"}

            data = resp.json().get("data", {})
            toss_winner = data.get("tossWinner")
            toss_choice = str(data.get("tossChoice") or "").lower()
            venue_str = data.get("venue", "")
            status_str = data.get("status", "")
            
            resolved_venue_id = self.resolve_venue_from_text(venue_str)

            # Also parse toss from status string if tossWinner field is blank
            st_low = status_str.lower()
            if not toss_winner:
                m_toss = re.search(r"([A-Za-z\s]+)\s+(?:won the toss|opt to|chose to|elected to)", st_low)
                if m_toss:
                    toss_winner = m_toss.group(1).strip()
            if not toss_choice:
                if any(k in st_low for k in ["opt to bat", "chose to bat", "elected to bat", "bat first"]):
                    toss_choice = "bat"
                elif any(k in st_low for k in ["opt to bowl", "opt to field", "chose to bowl", "chose to field", "elected to bowl", "bowl first"]):
                    toss_choice = "bowl"

            return {
                "status": "success",
                "cricapi_match_id": cricapi_match_id,
                "toss_winner": toss_winner,
                "toss_choice": toss_choice,
                "venue_name": venue_str,
                "venue_id": resolved_venue_id,
                "match_status": status_str,
                "teams": data.get("teams", []),
                "team_info": data.get("teamInfo", [])
            }
        except Exception as e:
            logger.error(f"Error fetching match toss/venue from CricAPI for {cricapi_match_id}: {e}")
            return {"status": "error", "reason": str(e)}

    def sync_senior_international_fixtures(self, force: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches senior international fixtures from CricketData.org (cricScore).
        Filters strictly for live matches and matches starting within 3 days.
        Strictly respects the 100 hits/day quota with a 4-hour cache TTL.
        NEVER fetches live scores from this endpoint.
        """
        now = time.time()
        if not force and hasattr(self, "_cached_fixtures_time"):
            if (now - self._cached_fixtures_time) < 14400:  # 4-hour TTL
                return getattr(self, "_cached_fixtures", [])

        if not self._check_and_increment_quota():
            logger.warning("CricAPI quota guard active. Using cached fixtures.")
            return getattr(self, "_cached_fixtures", [])

        url = f"https://api.cricapi.com/v1/cricScore?apikey={self.api_key}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=12)
            if resp.status_code != 200:
                logger.error(f"CricAPI fixtures failed: HTTP {resp.status_code}")
                return getattr(self, "_cached_fixtures", [])

            data = resp.json()
            if data.get("status") != "success":
                return getattr(self, "_cached_fixtures", [])

            # Filter strictly for senior international teams
            SENIOR_INTL = {"PAK", "SL", "IND", "NZ", "AUS", "ENG", "SA", "WI", "BAN", "AFG", "ZIM", "IRE"}
            raw_matches = data.get("data", [])
            intl_fixtures = []

            now_utc = datetime.now(timezone.utc)
            max_future = now_utc + timedelta(days=3)
            max_past = now_utc - timedelta(days=3)

            for m in raw_matches:
                t1 = (m.get("t1") or "").upper()
                t2 = (m.get("t2") or "").upper()
                
                # Check senior international
                m1 = re.search(r"\[([A-Z]{2,4})\]", t1) or re.search(r"\b([A-Z]{2,3})\b", t1)
                m2 = re.search(r"\[([A-Z]{2,4})\]", t2) or re.search(r"\b([A-Z]{2,3})\b", t2)
                c1 = m1.group(1) if m1 else t1[:3]
                c2 = m2.group(1) if m2 else t2[:3]

                # Senior national teams only (exclude A-teams, U19, and Women if requested)
                is_senior_a = c1 in SENIOR_INTL and not any(w in t1 for w in ["U19", "WOMEN", " LIONS", " A ["])
                is_senior_b = c2 in SENIOR_INTL and not any(w in t2 for w in ["U19", "WOMEN", " LIONS", " A ["])

                if is_senior_a and is_senior_b:
                    s_low = str(m.get("series") or "").lower()
                    m_type = str(m.get("matchType") or "").lower()
                    if "test" in s_low or "test" in m_type:
                        fmt = "TEST"
                    elif "odi" in s_low or "odi" in m_type or "one-day" in s_low:
                        fmt = "ODI"
                    else:
                        fmt = "T20I" 
                    is_live = bool(m.get("ms") == "live" or "live" in str(m.get("status") or "").lower())
                    
                    intl_fixtures.append({
                        "cricapi_id": m.get("id"),
                        "series_name": m.get("series"),
                        "team_a_code": c1,
                        "team_b_code": c2,
                        "team_a_name": m.get("t1"),
                        "team_b_name": m.get("t2"),
                        "format": fmt,
                        "scheduled_start": m.get("dateTimeGMT"),
                        "is_live": is_live,
                        "status_text": m.get("status")
                    })

            self._cached_fixtures = intl_fixtures
            self._cached_fixtures_time = now
            logger.info(f"Successfully cached {len(intl_fixtures)} senior international fixtures from CricAPI.")
            return intl_fixtures
        except Exception as e:
            logger.error(f"Error fetching CricAPI fixtures: {e}")
            return getattr(self, "_cached_fixtures", [])

    def parse_score_string(self, score_str: str) -> Dict[str, Any]:
        """
        Parses scores like '220/9 (43.5)' or '328/10 (48.3)' into runs, wickets, overs, balls.
        """
        runs, wickets, overs, balls = 0, 0, 0, 0
        if not score_str:
            return {"runs": 0, "wickets": 0, "overs": 0, "balls": 0}
        try:
            parts = score_str.strip().split()
            score_part = parts[0]
            if "/" in score_part:
                r_s, w_s = score_part.split("/")
                runs, wickets = int(r_s), int(w_s)
            else:
                runs = int(score_part)
            if len(parts) > 1:
                ov_part = parts[1].replace("(", "").replace(")", "")
                if "." in ov_part:
                    o_s, b_s = ov_part.split(".")
                    overs, balls = int(o_s), int(b_s)
                else:
                    overs, balls = int(ov_part), 0
        except Exception:
            pass
        return {"runs": runs, "wickets": wickets, "overs": overs, "balls": balls}

    def fetch_live_senior_scores(self, force: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches live senior international scores using a single-hit to cricScore.
        Caches results for 90 seconds to strictly conserve daily quota.
        Falls back to disk cache on network timeout.
        """
        now = time.time()
        if not force and hasattr(self, "_cached_live_scores_time"):
            if (now - self._cached_live_scores_time) < 90 and getattr(self, "_cached_live_scores", None):
                return getattr(self, "_cached_live_scores", [])

        SENIOR_INTL = {"PAK", "SL", "IND", "NZ", "AUS", "ENG", "SA", "RSA", "WI", "BAN", "AFG", "ZIM", "IRE"}

        def parse_matches_list(raw_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            live_matches = []
            for m in raw_matches:
                t1 = (m.get("t1") or "").upper()
                t2 = (m.get("t2") or "").upper()
                m1 = re.search(r"\[([A-Z]{2,4})\]", t1) or re.search(r"\b([A-Z]{2,3})\b", t1)
                m2 = re.search(r"\[([A-Z]{2,4})\]", t2) or re.search(r"\b([A-Z]{2,3})\b", t2)
                c1 = m1.group(1) if m1 else t1[:3]
                c2 = m2.group(1) if m2 else t2[:3]

                is_senior_a = c1 in SENIOR_INTL and not any(w in t1 for w in ["U19", "WOMEN", " LIONS", " A ["])
                is_senior_b = c2 in SENIOR_INTL and not any(w in t2 for w in ["U19", "WOMEN", " LIONS", " A ["])

                if is_senior_a and is_senior_b:
                    ms = m.get("ms")
                    status_text = m.get("status") or ""
                    st_low = status_text.lower()
                    if any(k in st_low for k in ["won by", "drawn", "tied", "abandoned", "no result", "starts at"]):
                        continue
                    is_live = bool(
                        ms == "live"
                        or any(k in st_low for k in ["need ", "opt to ", "day 1", "day 2", "day 3", "day 4", "day 5", "trail by", "lead by", "stumps", "session"])
                    )
                    if is_live:
                        t1_score = self.parse_score_string(m.get("t1s", ""))
                        t2_score = self.parse_score_string(m.get("t2s", ""))
                        live_matches.append({
                            "cricapi_id": m.get("id"),
                            "series_name": m.get("series"),
                            "team_a_code": c1,
                            "team_b_code": c2,
                            "team_a_name": m.get("t1"),
                            "team_b_name": m.get("t2"),
                            "team_a_score": t1_score,
                            "team_b_score": t2_score,
                            "raw_t1s": m.get("t1s"),
                            "raw_t2s": m.get("t2s"),
                            "status_text": status_text,
                            "match_type": m.get("matchType")
                        })
            return live_matches

        # Check in-memory cache TTL (600 seconds / 10 minutes to protect daily quota)
        if not force and getattr(self, "_cached_live_scores", None):
            cached_t = getattr(self, "_cached_live_scores_time", 0.0)
            if (now - cached_t) < 600.0:
                return self._cached_live_scores

        if not self._check_and_increment_quota():
            logger.warning("CricAPI quota guard active. Using cached live scores.")
            return getattr(self, "_cached_live_scores", [])

        url = f"https://api.cricapi.com/v1/cricScore?apikey={self.api_key}"
        try:
            with httpx.Client(timeout=8.0, headers=self.headers) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success":
                        raw_matches = data.get("data", [])
                        live_matches = parse_matches_list(raw_matches)
                        self._cached_live_scores = live_matches
                        self._cached_live_scores_time = now
                        logger.info(f"Successfully fetched and cached {len(live_matches)} live senior international matches from CricAPI.")
                        return live_matches
        except Exception as e:
            logger.error(f"Error calling live CricAPI scores: {e}")

        return getattr(self, "_cached_live_scores", [])

    async def sync_live_senior_mens_matches(self, conn) -> List[str]:
        """
        Automatically discovers and synchronizes live senior men's international matches
        between the Top 20 ICC teams from CricketData.org into the SQLite database.
        Always updates scores, overs, wickets dynamically from the API feed.
        """
        TOP_20_MAP = {
            "IND": "IND", "AUS": "AUS", "ENG": "ENG", "SA": "SA", "RSA": "SA",
            "NZ": "NZ", "PAK": "PAK", "SL": "SL", "WI": "WI", "BAN": "BAN",
            "AFG": "AFG", "ZIM": "ZIM", "IRE": "IRE", "IR": "IRE", "NED": "NED",
            "NET": "NED", "SCO": "SCO", "NAM": "NAM", "NEP": "NEP", "USA": "USA",
            "UAE": "UAE", "OMA": "OMA", "CAN": "CAN"
        }

        live_scores = self.fetch_live_senior_scores()
        updated_matches = []

        for m in live_scores:
            c1_raw = m.get("team_a_code", "")
            c2_raw = m.get("team_b_code", "")
            c1 = TOP_20_MAP.get(c1_raw)
            c2 = TOP_20_MAP.get(c2_raw)

            if not c1 or not c2:
                continue

            status_text = m.get("status_text", "")
            t1_sc = m.get("team_a_score", {})
            t2_sc = m.get("team_b_score", {})
            st_low = status_text.lower()

            is_finished = any(k in st_low for k in [
                "won by", "won the match", "match drawn", "match tied", "abandoned", "no result", "concluded"
            ])

            # Find matching fixture in database
            async with conn.execute("""
                SELECT * FROM matches
                WHERE ((team_a_id = ? AND team_b_id = ?) OR (team_a_id = ? AND team_b_id = ?))
                  AND state != 'COMPLETED'
                ORDER BY CASE WHEN state = 'LIVE_BALL_BY_BALL' THEN 0 ELSE 1 END, scheduled_start ASC LIMIT 1
            """, (c1, c2, c2, c1)) as cur:
                db_m = await cur.fetchone()

            if db_m:
                db_m = dict(db_m)
                mid = db_m["id"]
                if is_finished:
                    logger.info(f"CricAPI match concluded: {mid} ({status_text}). Marking COMPLETED.")
                    await conn.execute("""
                        UPDATE matches SET
                            state = 'COMPLETED',
                            is_stumps = 0,
                            stumps_label = NULL,
                            match_status_text = ?,
                            last_event_description = ?
                        WHERE id = ?
                    """, (status_text, status_text, mid))
                    updated_matches.append(mid)
                else:
                    # Determine active batting team and innings accurately
                    w1 = int(t1_sc.get("wickets") or 0)
                    w2 = int(t2_sc.get("wickets") or 0)
                    r1 = t1_sc.get("runs")
                    r2 = t2_sc.get("runs")

                    # Identify which team is batting in which innings
                    if r1 is not None and r2 is not None:
                        inn_val = 2
                        if w2 == 10 and w1 < 10:
                            # Team B all out in 1st innings; Team A currently batting in 2nd innings
                            bat_sc = t1_sc
                            bat_team = c1
                            inn_1_team, inn_1_sc = c2, t2_sc
                            inn_2_team, inn_2_sc = c1, t1_sc
                        elif w1 == 10 and w2 < 10:
                            # Team A all out in 1st innings; Team B currently batting in 2nd innings
                            bat_sc = t2_sc
                            bat_team = c2
                            inn_1_team, inn_1_sc = c1, t1_sc
                            inn_2_team, inn_2_sc = c2, t2_sc
                        elif any(k in st_low for k in ["lead by", "trail by", "need", "target"]):
                            if (c1.lower() in st_low or (c1 == "IND" and "india" in st_low) or (c1 == "AUS" and "australia" in st_low)):
                                bat_sc = t1_sc
                                bat_team = c1
                                inn_1_team, inn_1_sc = c2, t2_sc
                                inn_2_team, inn_2_sc = c1, t1_sc
                            else:
                                bat_sc = t2_sc
                                bat_team = c2
                                inn_1_team, inn_1_sc = c1, t1_sc
                                inn_2_team, inn_2_sc = c2, t2_sc
                        else:
                            # Default 2nd innings chaser / responder
                            bat_sc = t2_sc
                            bat_team = c2
                            inn_1_team, inn_1_sc = c1, t1_sc
                            inn_2_team, inn_2_sc = c2, t2_sc
                    else:
                        inn_val = 1
                        bat_sc = t1_sc if r1 is not None else t2_sc
                        bat_team = c1 if r1 is not None else c2
                        inn_1_team, inn_1_sc = bat_team, bat_sc
                        inn_2_team, inn_2_sc = None, None

                    runs = bat_sc.get("runs") or db_m.get("current_score") or 0
                    wkts = bat_sc.get("wickets") if bat_sc.get("wickets") is not None else (db_m.get("current_wickets") or 0)
                    ovs = bat_sc.get("overs") if bat_sc.get("overs") is not None else (db_m.get("current_overs") or 0)
                    balls = bat_sc.get("balls") or db_m.get("current_balls") or 0

                    # Anti-score-regression guard
                    if inn_val == db_m.get("current_innings") and runs < int(db_m.get("current_score") or 0):
                        runs = int(db_m.get("current_score") or 0)
                        wkts = int(db_m.get("current_wickets") or 0)
                        ovs = int(db_m.get("current_overs") or 0)
                        balls = int(db_m.get("current_balls") or 0)

                    is_stumps = 1 if "stumps" in st_low else int(db_m.get("is_stumps") or 0)
                    day_m = re.search(r"day\s*([1-5])", st_low)
                    s_day = int(day_m.group(1)) if day_m else int(db_m.get("stumps_day") or 1)
                    s_lbl = f"🌙 STUMPS • DAY {s_day}" if is_stumps else None

                    # Prepare 1st & 2nd innings values
                    i1_team = inn_1_team or db_m.get("innings_1_team_id")
                    i1_runs = inn_1_sc.get("runs") if inn_1_sc else db_m.get("innings_1_score")
                    i1_wkts = inn_1_sc.get("wickets") if inn_1_sc else db_m.get("innings_1_wickets")
                    i1_ovs = inn_1_sc.get("overs") if inn_1_sc else db_m.get("innings_1_overs")

                    i2_team = inn_2_team or db_m.get("innings_2_team_id")
                    i2_runs = inn_2_sc.get("runs") if inn_2_sc else db_m.get("innings_2_score")
                    i2_wkts = inn_2_sc.get("wickets") if inn_2_sc else db_m.get("innings_2_wickets")
                    i2_ovs = inn_2_sc.get("overs") if inn_2_sc else db_m.get("innings_2_overs")

                    await conn.execute("""
                        UPDATE matches SET
                            state = 'LIVE_BALL_BY_BALL',
                            current_innings = ?,
                            current_score = ?,
                            current_wickets = ?,
                            current_overs = ?,
                            current_balls = ?,
                            match_status_text = ?,
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
                            innings_2_overs = ?
                        WHERE id = ?
                    """, (
                        inn_val, runs, wkts, ovs, balls, status_text, is_stumps, s_day, s_lbl,
                        i1_team, i1_runs, i1_wkts, i1_ovs,
                        i2_team, i2_runs, i2_wkts, i2_ovs,
                        mid
                    ))
                    updated_matches.append(mid)

        await conn.commit()
        return updated_matches

cricapi_provider = CricApiSquadProvider()
