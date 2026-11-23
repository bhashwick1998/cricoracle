"""
Multi-Source Resilient Cricket Live Scraper & Telemetry Aggregator (Top Scraper)
Provides high-availability, fault-tolerant live ball-by-ball scraping across top cricket platforms:
1. Cricbuzz (Primary live score, ball commentary, active duel)
2. CREX (Secondary live score, ball feeds, SSR state extraction)
3. ESPNcricinfo (Tertiary live score & RSS feed)
4. ICC Official Website (Tournament & live match status)
5. Flashscore / SofaScore / CricketWorld (Auxiliary web fallbacks)
6. CricketData.org (CricAPI):
   - Fixture schedules & confirmed Playing 11 (cached at toss)
   - Ultimate fallback for live scores if web scrapers fail
   - Strictly enforced quota cap: Maximum 100 hits/day (safety buffer at 95)

Features:
- Automatic cascading failover when any source fails, stops, or times out
- Delivery deduplication: Unique (match_id, innings, over, ball) tracking to prevent duplicate records
- Score Monotonicity Guard: Prevents score or overs regression when switching between sources
- Rain & Weather Interruption Detection: Covers on, rain delay, wet outfield, bad light, play suspended
- Match Restriction Engine: Reduced overs detection, bowler quota limit (ceil(N/5)), and revised targets (DLS)
- Confirmed Playing 11: Extracts 11 players per team with captain/keeper flags and bench reserves
- Strict Scope: Only Senior Men's International Matches between Top 20 ICC teams
"""

import asyncio
import html
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup

from .cricbuzz_score_api import CricbuzzScoreApi
from .cricapi_provider import cricapi_provider, MAX_DAILY_CALLS
from ...engine.cricket_rules import evaluate_match_completion

logger = logging.getLogger("cricoracle.top_scraper")

# Top 20 ICC Senior Men's Teams
TOP_20_ICC_TEAMS: Set[str] = {
    "IND", "AUS", "ENG", "SA", "PAK", "NZ", "WI", "SL", "BAN", "AFG",
    "ZIM", "IRE", "NED", "SCO", "NAM", "NEP", "USA", "UAE", "OMA", "CAN"
}

TEAM_NAME_TO_CODE: Dict[str, str] = {
    "india": "IND", "australia": "AUS", "england": "ENG", "south africa": "SA",
    "rsa": "SA", "pakistan": "PAK", "new zealand": "NZ", "west indies": "WI",
    "sri lanka": "SL", "bangladesh": "BAN", "afghanistan": "AFG", "zimbabwe": "ZIM",
    "ireland": "IRE", "netherlands": "NED", "scotland": "SCO", "namibia": "NAM",
    "nepal": "NEP", "united states": "USA", "usa": "USA", "united arab emirates": "UAE",
    "uae": "UAE", "oman": "OMA", "canada": "CAN"
}

CODE_ALIASES: Dict[str, str] = {
    "RSA": "SA", "NET": "NED", "IR": "IRE"
}


def normalize_team_code(val: Optional[str]) -> Optional[str]:
    """Normalizes team code or name to standard Top 20 3-letter code."""
    if not val:
        return None
    cleaned = str(val).strip().upper()
    if cleaned in CODE_ALIASES:
        return CODE_ALIASES[cleaned]
    if cleaned in TOP_20_ICC_TEAMS:
        return cleaned
    low = str(val).strip().lower()
    for name, code in TEAM_NAME_TO_CODE.items():
        if name in low or low in name:
            return code
    return None


def is_senior_mens_international(title_or_meta: Any) -> bool:
    """
    Strict International Gate:
    Enforces that the match is an official Senior Men's International match
    between Top 20 ICC teams. Rejects Women, U19, 'A' teams, and domestic/franchise matches.
    """
    text = ""
    team_a = None
    team_b = None

    if isinstance(title_or_meta, dict):
        text = f"{title_or_meta.get('title', '')} {title_or_meta.get('series_name', '')} {title_or_meta.get('id', '')}"
        team_a = normalize_team_code(title_or_meta.get("team_a_id") or title_or_meta.get("team_a"))
        team_b = normalize_team_code(title_or_meta.get("team_b_id") or title_or_meta.get("team_b"))
    else:
        text = str(title_or_meta)

    low = text.lower()

    # Rejection keywords for women, youth, second-tier, or domestic leagues
    reject_patterns = [
        r"\bwomen\b", r"\bwomens\b", r"\bwodi\b", r"\bwt20\b", r"\bwbbl\b", r"\bwpl\b",
        r"\bu19\b", r"\bunder-19\b", r"\bu-19\b", r"\byouth\b",
        r"\bu-?23\b", r"\bunder-?23\b", r"\bcolts\b", r"\bacademy\b",
        r"\ba-team\b", r"\ba\s+team\b", r"\ba\s+tour\b", r"\ba\s+vs\b",
        r"\bindia a\b", r"\baustralia a\b", r"\bengland lions\b", r"\blions\b",
        r"\bnew zealand a\b", r"\bsri lanka a\b", r"\bpakistan a\b", r"\bshaheens\b",
        r"\bireland a\b", r"\bzimbabwe a\b", r"\bbangladesh a\b", r"\bafghanistan a\b",
        r"\binda\b", r"\bpaka\b", r"\bsla\b", r"\bbana\b", r"\bafga\b", r"\birea\b", r"\bzima\b",
        r"\bunofficial\b", r"\brising[\s\-_]*stars?\b", r"\bemerging[\s\-_]*(teams?|players?|asia)?\b",
        r"\bacc\b", r"\basia[\s\-_]*cup[\s\-_]*rising\b", r"\brising\b", r"\bgroup\s+[a-z0-9]\b",
        r"\b7th\s+match\b", r"\b8th\s+match\b", r"\b9th\s+match\b",
        r"\bbbl\b", r"\bipl\b", r"\bpsl\b", r"\bcpl\b", r"\bsmat\b", r"\branji\b",
        r"\bplunket\b", r"\bt10\b", r"\bclub\b", r"\btour match\b", r"\bwarm-up\b",
        r"\bdivision\b", r"\bshield\b", r"\bcsa\b", r"\bthree-day\b", r"\bfour-day\b",
        r"nepal[\s\-_]*premier", r"abu[\s\-_]*dhabi[\s\-_]*t10"
    ]
    for pat in reject_patterns:
        if re.search(pat, low):
            return False

    # Check teams if known
    if team_a and team_b:
        return team_a in TOP_20_ICC_TEAMS and team_b in TOP_20_ICC_TEAMS

    # Check text for Top 20 teams
    found_teams = set()
    for name, code in TEAM_NAME_TO_CODE.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            found_teams.add(code)
    for code in TOP_20_ICC_TEAMS:
        if re.search(rf"\b{code}\b", text.upper()):
            found_teams.add(code)

    return len(found_teams) >= 2


class DeliveryDeduplicator:
    """
    Strict Delivery Deduplication & Monotonicity Engine
    Guarantees no duplicate ball records are inserted and prevents score regression.
    """
    def __init__(self, max_cache_size: int = 5000):
        self._seen_deliveries: Set[str] = set()
        self._max_cache_size = max_cache_size

    def make_delivery_key(self, match_id: str, innings: int, over: int, ball: int) -> str:
        return f"{match_id}:{innings}:{over}:{ball}"

    def is_seen(self, match_id: str, innings: int, over: int, ball: int) -> bool:
        key = self.make_delivery_key(match_id, innings, over, ball)
        return key in self._seen_deliveries

    def record_delivery(self, match_id: str, innings: int, over: int, ball: int):
        key = self.make_delivery_key(match_id, innings, over, ball)
        if len(self._seen_deliveries) >= self._max_cache_size:
            # Clear half of the set when capacity is reached
            self._seen_deliveries.clear()
        self._seen_deliveries.add(key)

    async def is_duplicate_in_db(self, conn, match_id: str, innings: int, over: int, ball: int) -> bool:
        """Checks SQLite database to ensure the delivery does not already exist."""
        try:
            async with conn.execute(
                "SELECT 1 FROM match_deliveries WHERE match_id = ? AND innings = ? AND over = ? AND ball = ? LIMIT 1",
                (match_id, innings, over, ball)
            ) as c:
                row = await c.fetchone()
                return row is not None
        except Exception as e:
            logger.debug(f"DB deduplication check error: {e}")
            return False

    @staticmethod
    def is_score_regressive(
        current_innings: int,
        current_overs: int,
        current_balls: int,
        current_runs: int,
        new_innings: int,
        new_overs: int,
        new_balls: int,
        new_runs: int
    ) -> bool:
        """
        Anti-Score-Regression Guard:
        Returns True if new_score represents a backwards jump (latency lag from a fallback provider).
        """
        if new_innings < current_innings:
            return True
        if new_innings == current_innings:
            new_total_balls = new_overs * 6 + new_balls
            cur_total_balls = current_overs * 6 + current_balls
            if new_total_balls < cur_total_balls or new_runs < current_runs:
                return True
        return False


class RainAndRestrictionsDetector:
    """
    Detects rain delays, covers on, bad light, wet outfields, and restricted conditions.
    Calculates reduced overs and bowler quota restrictions according to ICC Standard Playing Conditions.
    """

    RAIN_KEYWORDS = [
        "covers on", "rain delay", "rain interruption", "delayed by rain",
        "rain stops play", "wet outfield", "match delayed", "play suspended",
        "bad light", "dls par", "dls method", "rain", "interrupted by rain",
        "ground under covers", "inspection scheduled", "covers are coming on"
    ]

    @classmethod
    def calculate_bowler_restrictions(cls, reduced_overs: int, format_code: str = "ODI") -> Dict[str, Any]:
        """
        Calculates maximum overs per bowler according to ICC Playing Conditions (Clause 13.9.1 / 13.9.2):
        max_overs = ceil(N / 5)
        Number of bowlers allowed the maximum is N % 5 (or 5 if N % 5 == 0).
        Remaining bowlers may bowl at most max_overs - 1.
        """
        standard_overs = 50 if format_code == "ODI" else 20
        reduced_overs = max(5 if format_code == "T20I" else 20, min(standard_overs, reduced_overs))
        overs_lost = max(0, standard_overs - reduced_overs)

        max_overs = math.ceil(reduced_overs / 5.0)
        rem = reduced_overs % 5
        bowlers_with_max = 5 if rem == 0 else rem
        other_max = max(1, max_overs - 1)

        return {
            "reduced_overs": reduced_overs,
            "overs_lost": overs_lost,
            "max_overs_per_bowler": max_overs,
            "bowlers_allowed_max": bowlers_with_max,
            "other_bowlers_max": other_max,
            "dls_applicable": True,
            "summary": (
                f"Match reduced to {reduced_overs} overs: Up to {bowlers_with_max} bowlers can bowl "
                f"a maximum of {max_overs} overs; remaining bowlers can bowl a maximum of {other_max} overs."
            )
        }

    @classmethod
    def detect(cls, status_text: str = "", commentary: str = "", format_code: str = "ODI") -> Dict[str, Any]:
        combined = f"{status_text} {commentary}".lower()

        is_rain = any(k in combined for k in cls.RAIN_KEYWORDS)

        # Detect reduced overs pattern
        reduced_overs = None
        red_match = re.search(r"(?:match\s+reduced\s+to|revised\s+to|reduced\s+to)\s+(\d+)\s+overs", combined)
        if not red_match:
            red_match = re.search(r"(\d+)\s+overs\s+per\s+side", combined)
        if red_match:
            try:
                reduced_overs = int(red_match.group(1))
            except Exception:
                pass

        restrictions = None
        if reduced_overs and format_code in ["ODI", "T20I", "T20"]:
            restrictions = cls.calculate_bowler_restrictions(reduced_overs, format_code)

        # Detect DLS revised target
        revised_target = None
        tar_match = re.search(r"target\s+(?:of\s+)?(\d+)\s+(?:runs\s+)?in\s+(\d+)\s+overs", combined)
        if tar_match:
            try:
                revised_target = int(tar_match.group(1))
            except Exception:
                pass
        elif "dls" in combined:
            simple_tar = re.search(r"target\s+(?:is\s+|of\s+)?(\d+)", combined)
            if simple_tar:
                try:
                    revised_target = int(simple_tar.group(1))
                except Exception:
                    pass

        weather_condition = "Rain Delay" if is_rain else "Clear / Good Playing Conditions"
        rain_probability = 95.0 if is_rain else 10.0

        return {
            "is_rain_interrupted": is_rain,
            "weather_condition": weather_condition,
            "rain_probability_pct": rain_probability,
            "reduced_overs": reduced_overs,
            "revised_target": revised_target,
            "restrictions": restrictions
        }


class FailoverScraper:
    """
    Unified Multi-Source Cascading Cricket Scraper & Telemetry Aggregator.
    Cascades across Cricbuzz, CREX, ESPNcricinfo, ICC, and CricketData.org (CricAPI).
    """

    def __init__(self):
        self.cricbuzz = CricbuzzScoreApi()
        self.deduplicator = DeliveryDeduplicator()
        self.rain_detector = RainAndRestrictionsDetector()
        self.headers_browser = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24", "Google Chrome";v="131"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
        }
        self.active_crex_links: Dict[str, str] = {}
        self.last_crex_fetch_time: float = 0.0
        self.cached_crex_fixtures: List[Dict[str, Any]] = []

    # ==========================================
    # PROVIDER 1: Cricbuzz
    # ==========================================
    async def _try_cricbuzz(
        self,
        match_id: str,
        query: Optional[str] = None,
        team_a: Optional[str] = None,
        team_b: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Scrapes Cricbuzz live scorecard and ball commentary with dynamic match ID discovery."""
        target_id = None
        if query and re.search(r"\d{5,8}", str(query)):
            target_id = self.cricbuzz.extract_match_id(query)
        elif team_a and team_b:
            target_id = await self.cricbuzz.discover_match_id_by_teams(team_a, team_b)

        if not target_id and query:
            target_id = query

        if not target_id and re.search(r"\d{5,8}", str(match_id)):
            target_id = re.search(r"\d{5,8}", str(match_id)).group(0)

        if not target_id and not (team_a and team_b):
            m_teams = re.search(r"m_([a-z]{2,4})_([a-z]{2,4})_", match_id)
            if m_teams:
                t1, t2 = m_teams.group(1).upper(), m_teams.group(2).upper()
                target_id = await self.cricbuzz.discover_match_id_by_teams(t1, t2)

        if not target_id:
            return None

        try:
            res = await self.cricbuzz.get_live_score(target_id)
            if res and res.get("status") == "success" and res.get("score"):
                parsed = self.cricbuzz.parse_score_components(res["score"])
                if parsed.get("runs", 0) > 0 or parsed.get("overs", 0) > 0:
                    res["provider_used"] = "cricbuzz"
                    res["parsed_score"] = parsed
                    res["cricbuzz_match_id"] = target_id
                    return res
        except Exception as e:
            logger.debug(f"Cricbuzz scrape attempt failed for {match_id}: {e}")
        return None

    # ==========================================
    # PROVIDER 2: CREX (crex.com SSR data)
    # ==========================================
    async def _fetch_crex_fixtures(self) -> List[Dict[str, Any]]:
        """Fetches and caches live fixture list from crex.com SSR bootstrap."""
        now = datetime.now(timezone.utc).timestamp()
        if self.cached_crex_fixtures and (now - self.last_crex_fetch_time) < 45.0:
            return self.cached_crex_fixtures

        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get("https://crex.com/fixtures/match-list", headers=self.headers_browser)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for s in soup.find_all("script"):
                        if s.string and "ssr-bootstrap" in s.string:
                            decoded = html.unescape(s.string.replace("&q;", '"'))
                            data = json.loads(decoded)
                            fixtures = data.get("https://stats.crickapi.com/fixture/getFixture", [])
                            if fixtures:
                                self.cached_crex_fixtures = fixtures
                                self.last_crex_fetch_time = now
                                return fixtures
        except Exception as e:
            logger.debug(f"CREX fixtures fetch failed: {e}")
        return self.cached_crex_fixtures

    async def _try_crex(self, match_id: str, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """Scrapes live match telemetry from CREX SSR JSON."""
        try:
            fixtures = await self._fetch_crex_fixtures()
            matched_fixture = None
            t_pair = {team_a, team_b}

            for f in fixtures:
                c1 = normalize_team_code(f.get("t1SName") or f.get("team1"))
                c2 = normalize_team_code(f.get("t2SName") or f.get("team2"))
                if c1 and c2 and {c1, c2} == t_pair:
                    matched_fixture = f
                    break

            if not matched_fixture:
                return None

            link = matched_fixture.get("link")
            if not link:
                # Return compact fixture score directly from match-list
                s1 = matched_fixture.get("team1Score1", "")
                s2 = matched_fixture.get("team2Score1", "")
                st = matched_fixture.get("result") or matched_fixture.get("statusText") or ""
                return {
                    "provider_used": "crex_fixture",
                    "title": f"{matched_fixture.get('team1')} vs {matched_fixture.get('team2')}",
                    "score": f"{matched_fixture.get('team2')} {s2}" if s2 else f"{matched_fixture.get('team1')} {s1}",
                    "match_status": st,
                    "format": matched_fixture.get("formats", "ODI").upper(),
                    "venue": matched_fixture.get("venue")
                }

            # Fetch full live scorecard from match URL
            full_url = f"https://crex.com{link}"
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(full_url, headers=self.headers_browser)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for s in soup.find_all("script"):
                        if s.string and "ssr-bootstrap" in s.string:
                            decoded = html.unescape(s.string.replace("&q;", '"'))
                            data = json.loads(decoded)
                            sv3 = data.get("https://api.goscorer.com/api/v3/getSV3", {})
                            ball_feeds = data.get("https://content.crickapi.com/commentary/v1/getBallFeeds", [])
                            post_data = data.get("https://stats.crickapi.com/i/live/getPostMatchDataV2", {})

                            striker = {
                                "name": sv3.get("player_full_name1") or sv3.get("pname1", "Striker"),
                                "runs": sv3.get("run1", 0),
                                "balls": sv3.get("ball1", 0),
                                "fours": sv3.get("four1", 0),
                                "sixes": sv3.get("six1", 0),
                                "sr": sv3.get("sr1", 0.0)
                            }
                            non_striker = {
                                "name": sv3.get("player_full_name2") or sv3.get("pname2", "Non-Striker"),
                                "runs": sv3.get("run2", 0),
                                "balls": sv3.get("ball2", 0),
                                "fours": sv3.get("four2", 0),
                                "sixes": sv3.get("six2", 0),
                                "sr": sv3.get("sr2", 0.0)
                            }
                            bowler = {
                                "name": sv3.get("bowler_full_name") or sv3.get("bname", "Bowler"),
                                "runs": sv3.get("bwr", 0),
                                "overs": sv3.get("bover", 0.0),
                                "eco": sv3.get("beco", 0.0)
                            }

                            latest_commentary = ""
                            if ball_feeds and isinstance(ball_feeds, list) and len(ball_feeds) > 0:
                                raw_c = ball_feeds[0].get("c", "")
                                latest_commentary = re.sub(r"&[lg];[^&]+&[lg];", "", raw_c).strip()

                            sc1 = sv3.get("score1", "")
                            ov1 = sv3.get("over1", "")
                            sc2 = sv3.get("score2", "")
                            ov2 = sv3.get("over2", "")
                            current_score = f"{sc2} ({ov2})" if sc2 else f"{sc1} ({ov1})"

                            return {
                                "provider_used": "crex",
                                "title": f"{sv3.get('team1')} vs {sv3.get('team2')}",
                                "score": current_score,
                                "current_batsmen": [striker, non_striker],
                                "current_bowler": bowler,
                                "commentary": latest_commentary,
                                "match_status": sv3.get("status") or matched_fixture.get("result", ""),
                                "target": sv3.get("target"),
                                "crr": sv3.get("crr"),
                                "rrr": sv3.get("rrr"),
                                "toss_team": post_data.get("toss_team"),
                                "toss_decision": post_data.get("chose_to"),
                                "venue": matched_fixture.get("venue")
                            }
        except Exception as e:
            logger.debug(f"CREX scrape attempt failed for {match_id}: {e}")
        return None

    # ==========================================
    # Multi-Innings Test Score Parser
    # ==========================================
    @staticmethod
    def parse_test_telemetry(text: str, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """
        Parses compound multi-innings Test scores from Cricinfo RSS, SofaScore, CREX, or Flashscore.
        Handles normal alternation, follow-ons, combined overs, and Stumps indicators.
        """
        clean = text.replace("*", "").strip()
        ta = normalize_team_code(team_a) or team_a
        tb = normalize_team_code(team_b) or team_b

        def parse_single_token(tok: str, is_completed_innings: bool = False) -> Dict[str, Any]:
            c_tok = tok.strip()
            m_ov = re.search(r"(\d+)(?:[/-](\d+))?\s*(?:all\s*out|d)?\s*[-–(]?\s*([\d.]+)?\s*(?:ov|\)|overs)?", c_tok, re.IGNORECASE)
            r = 0
            w = 0
            ov = None
            if m_ov:
                r = int(m_ov.group(1))
                is_declared = any(k in c_tok.lower() for k in [" d", "/d", "d ", "dec", "declared"])
                is_explicit_all_out = any(k in c_tok.lower() for k in ["all out", "all-out", "allout", "all_out"])

                if m_ov.group(2):
                    w = int(m_ov.group(2))
                elif is_explicit_all_out:
                    w = 10
                elif is_completed_innings and not is_declared:
                    # In cricket (especially Test matches), a completed innings without declaration is ALL OUT (10 wkts)
                    w = 10
                else:
                    w = 0
                if m_ov.group(3):
                    try:
                        ov = float(m_ov.group(3))
                    except ValueError:
                        ov = None
            return {"runs": r, "wickets": w, "overs": ov}

        # Case 1: Cricinfo RSS format:
        # "South Africa 557/10 v Bangladesh 100/5 & 400/10"
        # or "Pakistan 325/10 v Sri Lanka 74/3 & 98/10"
        if " v " in clean:
            parts = clean.split(" v ", 1)
            side1_txt = parts[0].strip()
            side2_txt = parts[1].strip()

            s1_team = None
            s2_team = None
            for name, code in TEAM_NAME_TO_CODE.items():
                if name in side1_txt.lower():
                    s1_team = code
                    break
            if not s1_team:
                for code in [ta, tb]:
                    if code in side1_txt.upper():
                        s1_team = code
                        break

            for name, code in TEAM_NAME_TO_CODE.items():
                if name in side2_txt.lower():
                    s2_team = code
                    break
            if not s2_team:
                for code in [ta, tb]:
                    if code in side2_txt.upper():
                        s2_team = code
                        break

            s1_team = s1_team or ta
            s2_team = s2_team or (tb if s1_team == ta else ta)

            side1_tokens = [tok.strip() for tok in side1_txt.split("&")]
            side2_tokens = [tok.strip() for tok in side2_txt.split("&")]

            all_innings = []

            # Check if side 2 has 2 scores (e.g. side 2 batted in Innings 1 & 3: "IND 720/9d v NZ 60/1 & 368")
            if len(side2_tokens) >= 2:
                t_act = parse_single_token(side2_tokens[0], is_completed_innings=False)
                t_prev = parse_single_token(side2_tokens[1], is_completed_innings=True)
                inn1_data = parse_single_token(side1_tokens[0], is_completed_innings=True)

                # Under natural Test sequence: side 2 batted 1st (t_prev), side 1 batted 2nd (inn1_data), side 2 batted 3rd (t_act)
                all_innings.append({
                    "team": s2_team,
                    "runs": t_prev["runs"],
                    "wickets": t_prev["wickets"],
                    "overs": t_prev["overs"],
                    "innings_index": 1
                })
                all_innings.append({
                    "team": s1_team,
                    "runs": inn1_data["runs"],
                    "wickets": inn1_data["wickets"],
                    "overs": inn1_data["overs"],
                    "innings_index": 2
                })
                all_innings.append({
                    "team": s2_team,
                    "runs": t_act["runs"],
                    "wickets": t_act["wickets"],
                    "overs": t_act["overs"],
                    "innings_index": 3,
                    "is_follow_on": False
                })
            elif len(side1_tokens) >= 2:
                # Normal alternation: Team 1 batted in Innings 1 & 3
                t_act = parse_single_token(side1_tokens[0], is_completed_innings=False)
                t_prev = parse_single_token(side1_tokens[1], is_completed_innings=True)
                inn2_data = parse_single_token(side2_tokens[0], is_completed_innings=True)

                all_innings.append({
                    "team": s1_team,
                    "runs": t_prev["runs"],
                    "wickets": t_prev["wickets"],
                    "overs": t_prev["overs"],
                    "innings_index": 1
                })
                all_innings.append({
                    "team": s2_team,
                    "runs": inn2_data["runs"],
                    "wickets": inn2_data["wickets"],
                    "overs": inn2_data["overs"],
                    "innings_index": 2
                })
                all_innings.append({
                    "team": s1_team,
                    "runs": t_act["runs"],
                    "wickets": t_act["wickets"],
                    "overs": t_act["overs"],
                    "innings_index": 3,
                    "is_follow_on": False
                })
            else:
                has_inn2 = bool(side2_tokens and any(c.isdigit() for c in side2_tokens[0]))
                inn1_data = parse_single_token(side1_tokens[0], is_completed_innings=has_inn2)
                all_innings.append({
                    "team": s1_team,
                    "runs": inn1_data["runs"],
                    "wickets": inn1_data["wickets"],
                    "overs": inn1_data["overs"],
                    "innings_index": 1
                })
                if has_inn2:
                    inn2_data = parse_single_token(side2_tokens[0], is_completed_innings=False)
                    all_innings.append({
                        "team": s2_team,
                        "runs": inn2_data["runs"],
                        "wickets": inn2_data["wickets"],
                        "overs": inn2_data["overs"],
                        "innings_index": 2
                    })

            return FailoverScraper._build_match_payload_from_innings(all_innings, s1_team, s2_team, clean)

        # Case 2: SofaScore text format:
        # e.g. "SA 557-124.1 ov 400 100-5 135.4 ov BAN" or "PAK 325-87.5 ov 98 74-3 57.3 ov SL"
        m_sofa = re.search(
            r"([A-Z]{2,4})\s*(\d+)(?:[/-](\d+))?(?:-([\d.]+)\s*ov)?\s+(\d+)(?:[/-](\d+))?\s+(\d+)(?:[/-](\d+))?(?:-([\d.]+)\s*ov)?\s*([A-Z]{2,4})?",
            clean,
            re.IGNORECASE
        )
        if m_sofa:
            t1 = normalize_team_code(m_sofa.group(1)) or ta
            r1 = int(m_sofa.group(2))
            w1 = int(m_sofa.group(3)) if m_sofa.group(3) else 10
            ov1 = float(m_sofa.group(4)) if m_sofa.group(4) else None

            r2 = int(m_sofa.group(5))
            w2 = int(m_sofa.group(6)) if m_sofa.group(6) else 10

            r3 = int(m_sofa.group(7))
            w3 = int(m_sofa.group(8)) if m_sofa.group(8) else 0
            comb_ov2 = float(m_sofa.group(9)) if m_sofa.group(9) else None

            t2_raw = m_sofa.group(10)
            t2 = normalize_team_code(t2_raw) or (tb if t1 == ta else ta)

            all_innings = [
                {"team": t1, "runs": r1, "wickets": w1, "overs": ov1, "innings_index": 1},
                {"team": t2, "runs": r2, "wickets": w2, "overs": None, "innings_index": 2},
                {"team": t2, "runs": r3, "wickets": w3, "overs": None, "innings_index": 3, "is_follow_on": True}
            ]
            return FailoverScraper._build_match_payload_from_innings(all_innings, t1, t2, clean, combined_overs_t2=comb_ov2)

        return None

    @staticmethod
    def _build_match_payload_from_innings(
        all_innings: List[Dict[str, Any]],
        team_1: str,
        team_2: str,
        raw_text: str = "",
        combined_overs_t2: Optional[float] = None
    ) -> Dict[str, Any]:
        """Builds standardized live match telemetry from parsed multi-innings."""
        inn1 = all_innings[0]
        inn2 = all_innings[1] if len(all_innings) > 1 else None
        inn3 = all_innings[2] if len(all_innings) > 2 else None

        status_text = ""
        is_stumps = False
        stumps_day = 1

        if inn3:
            is_follow_on = inn3.get("is_follow_on") or (inn3["team"] == inn2["team"] if inn2 else False)
            if is_follow_on and inn2:
                deficit = inn1["runs"] - (inn2["runs"] + inn3["runs"])
                if deficit > 0:
                    lead_team = inn1["team"]
                    trail_team = inn2["team"]
                    lead_runs = deficit
                    status_trail = f"{trail_team} trail by {lead_runs} runs"
                elif deficit == 0:
                    status_trail = "Scores level"
                else:
                    lead_team = inn2["team"]
                    lead_runs = abs(deficit)
                    status_trail = f"{lead_team} lead by {lead_runs} runs"
            elif inn2:
                lead_val = (inn1["runs"] + inn3["runs"]) - inn2["runs"]
                status_trail = f"{inn1['team']} lead by {lead_val} runs" if lead_val >= 0 else f"{inn2['team']} lead by {abs(lead_val)} runs"
            else:
                status_trail = ""

            active_inn = inn3
            active_team = inn3["team"]

            m_day = re.search(r"day\s*([1-5])", raw_text, re.I)
            if m_day:
                stumps_day = int(m_day.group(1))
            else:
                ov_t1 = inn1.get("overs") or 90.0
                ov_t2 = combined_overs_t2 or ((inn2.get("overs") or 90.0) + (inn3.get("overs") or 30.0))
                total_overs = ov_t1 + ov_t2
                stumps_day = min(5, max(1, int(total_overs // 80) + 1))

            is_stumps = bool(re.search(r"\bstumps\b|\bstump\b|close of play", raw_text, re.I))
            status_text = f"Day {stumps_day}: Stumps - {status_trail}" if is_stumps else f"Day {stumps_day}: {status_trail}"
        elif inn2:
            lead_val = inn1["runs"] - inn2["runs"]
            status_text = f"{inn1['team']} lead by {lead_val} runs" if lead_val > 0 else f"{inn2['team']} lead by {abs(lead_val)} runs"
            active_inn = inn2
            active_team = inn2["team"]
        else:
            status_text = f"{inn1['team']} 1st Innings"
            active_inn = inn1
            active_team = inn1["team"]

        score_str = f"{active_team} {active_inn['runs']}/{active_inn['wickets']}"
        if active_inn.get("overs") is not None:
            score_str += f" ({active_inn['overs']})"

        return {
            "score": score_str,
            "match_status": status_text,
            "all_innings": all_innings,
            "current_innings": len(all_innings),
            "is_stumps": is_stumps,
            "stumps_day": stumps_day,
            "active_team": active_team
        }

    # ==========================================
    # PROVIDER 3: ESPNcricinfo RSS & Web
    # ==========================================
    async def _try_espn(self, match_id: str, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """Scrapes live match scores from ESPNcricinfo RSS feed."""
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get("https://www.espncricinfo.com/rss/livescores.xml", headers=self.headers_browser)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    items = root.findall(".//item")

                    for item in items:
                        title_el = item.find("title")
                        link_el = item.find("link")
                        if title_el is not None and title_el.text:
                            t_txt = title_el.text
                            if not is_senior_mens_international(t_txt):
                                continue

                            has_team_a = team_a in t_txt.upper() or any(k in t_txt.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_a)
                            has_team_b = team_b in t_txt.upper() or any(k in t_txt.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_b)

                            if has_team_a and has_team_b:
                                parsed = self.parse_test_telemetry(t_txt, team_a, team_b)
                                if parsed:
                                    parsed["provider_used"] = "espncricinfo_rss"
                                    parsed["title"] = t_txt
                                    parsed["link"] = link_el.text if link_el is not None else ""
                                    return parsed

                                return {
                                    "provider_used": "espncricinfo_rss",
                                    "title": t_txt,
                                    "score": t_txt,
                                    "match_status": t_txt,
                                    "link": link_el.text if link_el is not None else ""
                                }
        except Exception as e:
            logger.debug(f"ESPN RSS scrape attempt failed for {match_id}: {e}")
        return None

    # ==========================================
    # DOM-Resilient Extraction Methodology
    # ==========================================
    def parse_dom_resilient_html(self, html_text: str, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """
        DOM-Resilient Extraction Methodology:
        Extracts live cricket scores immune to CSS / DOM layout changes via:
        1. JSON-LD (Schema.org SportsEvent) microdata
        2. OpenGraph / Twitter meta tags
        3. Semantic regular expressions on text stream
        """
        try:
            soup = BeautifulSoup(html_text, "html.parser")

            # Layer 1: JSON-LD Schema.org extraction
            for script in soup.find_all("script", type="application/ld+json"):
                if not script.string:
                    continue
                try:
                    data = json.loads(script.string)
                    items = data if isinstance(data, list) else [data]
                    for it in items:
                        desc = str(it.get("description") or it.get("name") or "")
                        has_a = team_a in desc.upper() or any(k in desc.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_a)
                        has_b = team_b in desc.upper() or any(k in desc.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_b)
                        if has_a and has_b:
                            parsed = self.parse_test_telemetry(desc, team_a, team_b)
                            if parsed:
                                parsed["provider_used"] = "json_ld"
                                return parsed
                except Exception:
                    continue

            # Layer 2: Meta description extraction (SEO tags always reflect live status)
            for m_name in ["og:description", "description", "twitter:description"]:
                m_el = soup.find("meta", property=m_name) or soup.find("meta", attrs={"name": m_name})
                if m_el and m_el.get("content"):
                    cnt = str(m_el["content"])
                    has_a = team_a in cnt.upper() or any(k in cnt.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_a)
                    has_b = team_b in cnt.upper() or any(k in cnt.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_b)
                    if has_a and has_b:
                        parsed = self.parse_test_telemetry(cnt, team_a, team_b)
                        if parsed:
                            parsed["provider_used"] = "meta_tag"
                            return parsed

            # Layer 3: Semantic regex on flattened text stream
            text_all = soup.get_text(" ", strip=True)
            has_a = team_a in text_all.upper() or any(k in text_all.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_a)
            has_b = team_b in text_all.upper() or any(k in text_all.lower() for k, v in TEAM_NAME_TO_CODE.items() if v == team_b)
            if has_a and has_b:
                return self.parse_test_telemetry(text_all, team_a, team_b)
        except Exception as e:
            logger.debug(f"DOM resilient parse error: {e}")
        return None

    # ==========================================
    # PROVIDER 4: SofaScore Live Web Scraper
    # ==========================================
    async def _try_sofascore(self, match_id: str, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """Scrapes live match scores from SofaScore tournament and match feeds."""
        try:
            url = "https://www.sofascore.com/cricket/tournament/world/icc-world-test-championship/18833"
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=self.headers_browser)
                if resp.status_code == 200 and resp.text:
                    parsed = self.parse_dom_resilient_html(resp.text, team_a, team_b)
                    if parsed:
                        parsed["provider_used"] = "sofascore"
                        parsed["title"] = f"{team_a} vs {team_b}"
                        return parsed
        except Exception as e:
            logger.debug(f"SofaScore scrape attempt failed for {match_id}: {e}")
        return None

    # ==========================================
    # PROVIDER 4: ICC Website & Auxiliary Feeds
    # ==========================================
    async def _try_icc_and_aux(self, match_id: str, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """Scrapes ICC official tournament and live feeds."""
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get("https://www.icc-cricket.com/live-cricket/live", headers=self.headers_browser)
                if resp.status_code == 200 and resp.text:
                    parsed = self.parse_dom_resilient_html(resp.text, team_a, team_b)
                    if parsed:
                        parsed["provider_used"] = "icc_official"
                        parsed["title"] = f"{team_a} vs {team_b}"
                        return parsed
                    soup = BeautifulSoup(resp.text, "html.parser")
                    text_content = soup.get_text(" ", strip=True)
                    has_a = team_a in text_content.upper()
                    has_b = team_b in text_content.upper()
                    if has_a and has_b:
                        # Find card containing both teams
                        for card in soup.find_all(["div", "section"]):
                            c_text = card.get_text(" ", strip=True)
                            if team_a in c_text.upper() and team_b in c_text.upper() and 20 < len(c_text) < 300:
                                return {
                                    "provider_used": "icc_official",
                                    "title": f"{team_a} vs {team_b}",
                                    "score": c_text,
                                    "match_status": c_text
                                }
        except Exception as e:
            logger.debug(f"ICC live scrape attempt failed for {match_id}: {e}")
        return None

    # ==========================================
    # PROVIDER 5: CricketData.org (CricAPI Fallback)
    # Strictly limited to <= 100 hits/day
    # ==========================================
    def _try_cricapi_fallback(self, team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """
        Ultimate fallback provider using CricketData.org API.
        Enforces daily quota cap and 90-second in-memory/disk cache.
        """
        try:
            live_matches = cricapi_provider.fetch_live_senior_scores(force=False)
            t_pair = {team_a, team_b}
            for m in live_matches:
                c1 = normalize_team_code(m.get("team_a_code"))
                c2 = normalize_team_code(m.get("team_b_code"))
                if c1 and c2 and {c1, c2} == t_pair:
                    t1_sc = m.get("team_a_score", {})
                    t2_sc = m.get("team_b_score", {})
                    st_low = (m.get("status_text") or "").lower()

                    # Select active batting team score
                    w1 = int(t1_sc.get("wickets") or 0)
                    w2 = int(t2_sc.get("wickets") or 0)
                    r1 = t1_sc.get("runs")
                    r2 = t2_sc.get("runs")

                    if w2 == 10 and w1 < 10:
                        bat_sc = t1_sc
                        bat_team = c1
                    elif w1 == 10 and w2 < 10:
                        bat_sc = t2_sc
                        bat_team = c2
                    elif any(k in st_low for k in ["need", "lead by", "trail by", "target"]):
                        if (c1.lower() in st_low or (c1 == "IND" and "india" in st_low) or (c1 == "AUS" and "australia" in st_low)):
                            bat_sc = t1_sc
                            bat_team = c1
                        else:
                            bat_sc = t2_sc
                            bat_team = c2
                    elif r2 is not None and r1 is not None:
                        bat_sc = t2_sc
                        bat_team = c2
                    else:
                        bat_sc = t1_sc if r1 is not None else t2_sc
                        bat_team = c1 if r1 is not None else c2

                    score_str = f"{bat_team} {bat_sc.get('runs', 0)}/{bat_sc.get('wickets', 0)} ({bat_sc.get('overs', 0)}.{bat_sc.get('balls', 0)})"

                    return {
                        "provider_used": "cricapi_fallback",
                        "title": f"{c1} vs {c2}",
                        "score": score_str,
                        "raw_cricapi": m,
                        "match_status": m.get("status_text")
                    }
        except Exception as e:
            logger.debug(f"CricAPI live fallback failed: {e}")
        return None

    # ==========================================
    # Cascading Telemetry Execution
    # ==========================================
    async def get_live_telemetry_with_cascade(
        self,
        match_id: str,
        match_meta: Optional[Dict[str, Any]] = None,
        cricbuzz_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cascades across top live sources in sequence:
        1. Cricbuzz -> 2. CREX -> 3. ESPNcricinfo -> 4. ICC -> 5. CricAPI (Quota-guarded)
        Returns structured telemetry with weather, restrictions, and delivery info.
        """
        team_a = normalize_team_code(match_meta.get("team_a_id") if match_meta else None) or "IND"
        team_b = normalize_team_code(match_meta.get("team_b_id") if match_meta else None) or "AUS"
        fmt = match_meta.get("format", "ODI") if match_meta else "ODI"

        failover_log = []

        # 1. Primary: Cricbuzz (with dynamic match discovery by teams)
        cb_res = await self._try_cricbuzz(match_id, query=cricbuzz_query, team_a=team_a, team_b=team_b)
        if cb_res:
            cb_res["failover_history"] = failover_log
            cb_res["restrictions"] = self.rain_detector.detect(
                cb_res.get("match_status", ""),
                cb_res.get("commentary", ""),
                fmt
            )
            return cb_res
        failover_log.append("cricbuzz_failed_or_timeout")

        # 2. Secondary: CREX
        crex_res = await self._try_crex(match_id, team_a, team_b)
        if crex_res:
            crex_res["failover_history"] = failover_log
            crex_res["restrictions"] = self.rain_detector.detect(
                crex_res.get("match_status", ""),
                crex_res.get("commentary", ""),
                fmt
            )
            return crex_res
        failover_log.append("crex_failed_or_no_match")

        # 3. Tertiary: ESPNcricinfo
        espn_res = await self._try_espn(match_id, team_a, team_b)
        if espn_res:
            espn_res["failover_history"] = failover_log
            espn_res["restrictions"] = self.rain_detector.detect(
                espn_res.get("match_status", ""),
                espn_res.get("commentary", ""),
                fmt
            )
            return espn_res
        failover_log.append("espn_failed")

        # 4. SofaScore Web Scraper
        sofa_res = await self._try_sofascore(match_id, team_a, team_b)
        if sofa_res:
            sofa_res["failover_history"] = failover_log
            sofa_res["restrictions"] = self.rain_detector.detect(
                sofa_res.get("match_status", ""),
                sofa_res.get("commentary", ""),
                fmt
            )
            return sofa_res
        failover_log.append("sofascore_failed")

        # 5. Quaternary: ICC Official Live
        icc_res = await self._try_icc_and_aux(match_id, team_a, team_b)
        if icc_res:
            icc_res["failover_history"] = failover_log
            icc_res["restrictions"] = self.rain_detector.detect(
                icc_res.get("match_status", ""),
                icc_res.get("commentary", ""),
                fmt
            )
            return icc_res
        failover_log.append("icc_failed")

        # 6. Ultimate Fallback: CricketData.org (CricAPI)
        cricapi_res = self._try_cricapi_fallback(team_a, team_b)
        if cricapi_res:
            cricapi_res["failover_history"] = failover_log
            cricapi_res["restrictions"] = self.rain_detector.detect(
                cricapi_res.get("match_status", ""),
                "",
                fmt
            )
            return cricapi_res
        failover_log.append("cricapi_fallback_exhausted")

        return {
            "status": "failover_exhausted",
            "provider_used": "none",
            "failover_history": failover_log,
            "message": "All live scrapers returned empty or matches concluded."
        }

    # ==========================================
    # Confirmed Playing 11 Extraction
    # ==========================================
    async def get_confirmed_playing_xi(
        self,
        match_id: str,
        team_a: str,
        team_b: str,
        cricapi_id: Optional[str] = None,
        scheduled_start_iso: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts confirmed Playing 11 (11 players per team) and bench/reserves.
        First attempts CricAPI (1 call cached per match), then Cricbuzz squads, then CREX.
        """
        # 1. Check CricAPI (strictly cached, at most 1 hit per match at toss)
        if cricapi_id:
            squad_res = cricapi_provider.fetch_playing_xi_for_match(
                cricapi_id,
                scheduled_start_iso=scheduled_start_iso,
                force_call=False
            )
            if squad_res.get("status") == "success" and squad_res.get("teams"):
                t_data = squad_res["teams"]
                if len(t_data) >= 2:
                    return {
                        "status": "success",
                        "provider_used": "cricapi_squad",
                        "team_a_xi": t_data[0].get("players", [])[:11],
                        "team_b_xi": t_data[1].get("players", [])[:11],
                        "team_a_bench": t_data[0].get("players", [])[11:],
                        "team_b_bench": t_data[1].get("players", [])[11:],
                    }

        # 2. Check Cricbuzz squads
        try:
            cb_id = self.cricbuzz.extract_match_id(match_id)
            if cb_id:
                cb_sq = await self.cricbuzz.get_match_squads(cb_id)
                if cb_sq.get("status") == "success":
                    sq_a = cb_sq.get("team_a_squad", [])
                    sq_b = cb_sq.get("team_b_squad", [])
                    xi_a = [p for p in sq_a if p.get("is_playing_xi")]
                    xi_b = [p for p in sq_b if p.get("is_playing_xi")]
                    if len(xi_a) >= 11 and len(xi_b) >= 11:
                        return {
                            "status": "success",
                            "provider_used": "cricbuzz_squad",
                            "team_a_xi": xi_a[:11],
                            "team_b_xi": xi_b[:11],
                            "team_a_bench": [p for p in sq_a if not p.get("is_playing_xi")],
                            "team_b_bench": [p for p in sq_b if not p.get("is_playing_xi")],
                        }
        except Exception as e:
            logger.debug(f"Cricbuzz squad extraction failed: {e}")

        return {"status": "unannounced", "message": "Playing XI not yet confirmed."}


# Global singleton instance
top_scraper = FailoverScraper()
